# file: vidur/scheduler/global_scheduler/dqn_global_scheduler_online.py
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from vidur.entities import Replica, Request
from vidur.config import SimulationConfig
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler_with_state import (
    build_global_state,
    debug_dump_replica_state,
)

logger = init_logger(__name__)


class _ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.s: List[np.ndarray] = []
        self.a: List[int] = []
        self.r: List[float] = []
        self.s2: List[np.ndarray] = []
        self.d: List[bool] = []
        self.cap = int(capacity)

    def push(self, s, a, r, s2, d) -> None:
        if len(self.s) >= self.cap:
            self.s.pop(0); self.a.pop(0); self.r.pop(0); self.s2.pop(0); self.d.pop(0)
        self.s.append(s.copy()); self.a.append(int(a)); self.r.append(float(r))
        self.s2.append(s2.copy()); self.d.append(bool(d))

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.s))
        idx = random.sample(range(len(self.s)), n)
        s = np.stack([self.s[i] for i in idx], axis=0)
        a = np.asarray([self.a[i] for i in idx], dtype=np.int64)
        r = np.asarray([self.r[i] for i in idx], dtype=np.float32)
        s2 = np.stack([self.s2[i] for i in idx], axis=0)
        d = np.asarray([self.d[i] for i in idx], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.s)


class _RunningNormalizer:
    """Welford 标准化，避免特征尺度过大导致爆炸。"""
    def __init__(self, eps: float = 1e-6, clip: float = 5.0) -> None:
        self.eps = float(eps)
        self.clip = float(clip)
        self.count = 0
        self.mean: Optional[np.ndarray] = None
        self.m2: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float32, copy=False)
        if self.mean is None:
            self.mean = x.copy()
            self.m2 = np.zeros_like(x)
            self.count = 1
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.m2 is None or self.count < 2:
            # 冷启动：仅做简单缩放+裁剪，避免巨幅数值
            z = x / (np.linalg.norm(x) + 1.0)
        else:
            var = self.m2 / (self.count - 1)
            std = np.sqrt(np.maximum(var, self.eps))
            z = (x - self.mean) / std
        return np.clip(z, -self.clip, self.clip).astype(np.float32, copy=False)


class _SimpleDQN:
    """
    线性 Q 近似，Huber 损失，梯度裁剪 + L2 正则；平局随机 argmax；数值 NaN/Inf 清洗。
    """
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        buffer_size: int,
        batch_size: int,
        l2_weight: float = 1e-5,
        huber_delta: float = 1.0,
        grad_clip: float = 10.0,
        seed: Optional[int] = 42,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sd = int(state_dim)
        self.na = int(num_actions)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.eps = float(epsilon)
        self.eps_min = float(epsilon_min)
        self.eps_decay = float(epsilon_decay)
        self.bs = int(batch_size)
        self.l2 = float(l2_weight)
        self.huber_delta = float(huber_delta)
        self.grad_clip = float(grad_clip)

        lim = math.sqrt(6.0 / (self.sd + self.na))
        self.W = np.random.uniform(-lim, lim, size=(self.na, self.sd)).astype(np.float32)
        self.b = np.zeros((self.na,), dtype=np.float32)

        self.buf = _ReplayBuffer(buffer_size)

    @staticmethod
    def _nan_to_num(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
        return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill, copy=False)

    def q(self, s: np.ndarray) -> np.ndarray:
        # 清洗输入，避免 matmul NaN/Inf
        s = self._nan_to_num(s)
        q = (self.W @ s.T).T + self.b
        return self._nan_to_num(q)

    def act(self, s: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return int(np.random.randint(self.na))
        q = self.q(s[None, :])[0]
        # 若全为 0 或非数，fallback 随机
        if not np.isfinite(q).any():
            return int(np.random.randint(self.na))
        q = self._nan_to_num(q)
        m = np.max(q)
        best = np.flatnonzero(q == m)
        if best.size == 0:
            return int(np.random.randint(self.na))
        return int(np.random.choice(best))

    def push(self, s, a, r, s2, d) -> None:
        self.buf.push(s, a, r, s2, d)

    def _huber_grad(self, diff: np.ndarray) -> np.ndarray:
        # Huber 对 q 的梯度：clip(diff, -δ, δ)
        return np.clip(diff, -self.huber_delta, self.huber_delta)

    def train_step(self) -> Tuple[float, float]:
        if len(self.buf) == 0:
            return 0.0, 0.0
        s, a, r, s2, d = self.buf.sample(self.bs)
        q = self.q(s)          # (B, A)
        q2 = self.q(s2)        # (B, A)
        y = r + (1.0 - d) * self.gamma * np.max(q2, axis=1)

        idx = (np.arange(s.shape[0]), a)
        diff = q[idx] - y
        g = self._huber_grad(diff)  # (B,)

        # 只更新被选择的动作行
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        for i in range(s.shape[0]):
            ai = a[i]
            grad_W[ai] += g[i] * s[i]
            grad_b[ai] += g[i]

        # L2 正则
        grad_W += self.l2 * self.W

        # 梯度裁剪
        gn = float(np.sqrt(np.sum(grad_W**2) + np.sum(grad_b**2)))
        if gn > self.grad_clip and gn > 0.0:
            scale = self.grad_clip / gn
            grad_W *= scale
            grad_b *= scale

        # SGD
        n = max(1, s.shape[0])
        self.W -= self.lr * grad_W / n
        self.b -= self.lr * grad_b / n

        # 退火
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        # Huber 损失（标量）
        abs_diff = np.abs(diff)
        quad = np.minimum(abs_diff, self.huber_delta)
        lin = abs_diff - quad
        loss = 0.5 * (quad**2) + self.huber_delta * lin
        return float(np.mean(loss)), gn


class DQNGlobalSchedulerOnline(BaseGlobalScheduler):
    """
    在线 DQN 路由器：
    - 状态：拼接各副本向量 + [global_queue_len, throughput, avg_latency]。
    - 奖励（已改为瞬时/窗口化）：Δthroughput_win - λ_lat·Δlatency_win - λ_bal·balance_penalty。
      * throughput_win/latency_win 为最近 reward_window_sec 秒窗口的瞬时指标。
      * 若 MetricsStore 不支持瞬时接口，则自动回退到累计指标（向后兼容）。
    - 归一化：运行统计标准化 + 裁剪。
    - 训练输出：每步打印 step/epsilon/action/tp/lat/delta/bal/reward/loss/q_vals/grad_norm。
    """
    TYPE = "globalscheduleonline"

    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]) -> None:
        super().__init__(config, replicas)
        self._replica_ids: List[int] = sorted(list(replicas.keys()))

        gcfg = getattr(getattr(config, "cluster_config", None), "global_scheduler_config", None)
        def _get(k, d): return getattr(gcfg, k, d) if gcfg is not None else d

        self._lr = float(_get("lr", 2e-3))    # 稍降 LR，稳定性更好
        self._gamma = float(_get("gamma", 0.95))
        self._epsilon = float(_get("epsilon", 0.2))
        self._epsilon_min = float(_get("epsilon_min", 0.02))
        self._epsilon_decay = float(_get("epsilon_decay", 0.995))
        self._buffer_size = int(_get("buffer_size", 20000))
        self._batch_size = int(_get("batch_size", 64))
        self._reward_latency_weight = float(_get("reward_latency_weight", 1.0))
        self._balance_penalty_weight = float(_get("balance_penalty_weight", 0.0))  # 0 表示不启用
        self._max_queue_requests = int(_get("max_queue_requests_per_replica", 4))
        self._debug_dump = bool(_get("debug_dump_global_state", False))

        # 关键：瞬时奖励窗口（秒）；若未在 config dataclass 暴露 CLI，这里仍可读取默认值
        self._reward_window_sec = float(_get("reward_window_sec", 1.0))

        # DQN & 归一化
        self._dqn: Optional[_SimpleDQN] = None
        self._norm = _RunningNormalizer(eps=1e-6, clip=5.0)

        # 训练缓存
        self._last_state: Optional[np.ndarray] = None
        self._last_state_norm: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_tp: float = 0.0     # 缓存：窗口化瞬时 tp
        self._last_lat: float = 0.0    # 缓存：窗口化瞬时 latency
        self._step: int = 0

    # ---- 新增：窗口化瞬时指标读取（带回退） ----
    def _metrics_window(self) -> Tuple[float, float]:
        """返回窗口化瞬时吞吐/时延；若无瞬时接口则回退到累计接口。"""
        if getattr(self, "_metric_store", None) is None:
            return 0.0, 0.0

        # 优先使用瞬时接口
        has_inst = (
            hasattr(self._metric_store, "get_instant_throughput")
            and hasattr(self._metric_store, "get_instant_latency")
        )
        if has_inst:
            try:
                tp = float(self._metric_store.get_instant_throughput(self._reward_window_sec, self._current_time))
            except Exception:
                tp = 0.0
            try:
                lat = float(self._metric_store.get_instant_latency(self._reward_window_sec, self._current_time))
            except Exception:
                lat = 0.0
            return tp, lat

        # 回退：累计指标（兼容老版本）
        try:
            tp = float(self._metric_store.get_throughput(self._current_time))
        except Exception:
            tp = 0.0
        try:
            lat = float(self._metric_store.get_average_latency())
        except Exception:
            lat = 0.0
        return tp, lat

    def schedule(self) -> List[Tuple[int, Request]]:
        if not self._request_queue:
            return []

        if self._debug_dump:
            for rid in self._replica_ids:
                rs = self.get_replica_scheduler(rid)
                debug_dump_replica_state(rid, rs)

        # 原始状态
        s = build_global_state(
            self._replicas,
            self.get_replica_scheduler,
            self._current_time,
            self._metric_store,
            max_queue_requests=self._max_queue_requests,
        )
        # 更新归一化器并标准化
        self._norm.update(s)
        s_norm = self._norm.normalize(s)

        # 初始化 DQN
        if self._dqn is None:
            self._dqn = _SimpleDQN(
                state_dim=int(s_norm.shape[0]),
                num_actions=len(self._replica_ids),
                lr=self._lr,
                gamma=self._gamma,
                epsilon=self._epsilon,
                epsilon_min=self._epsilon_min,
                epsilon_decay=self._epsilon_decay,
                buffer_size=self._buffer_size,
                batch_size=self._batch_size,
                l2_weight=1e-5,
                huber_delta=1.0,
                grad_clip=10.0,
                seed=42,
            )
            # 初始化缓存：使用“窗口化瞬时指标”
            self._last_tp, self._last_lat = self._metrics_window()
            logger.info(
                f"[DQN:init] state_dim={s_norm.shape[0]}, actions={len(self._replica_ids)}, "
                f"lr={self._lr}, gamma={self._gamma}, eps0={self._epsilon}, reward_window_sec={self._reward_window_sec}"
            )

        # 训练一步（如果有上一步）
        if self._last_state_norm is not None and self._last_action is not None:
            # 使用“窗口化瞬时指标”的差分，形成“优势式”奖励
            tp_now, lat_now = self._metrics_window()
            d_tp = tp_now - self._last_tp
            d_lat = lat_now - self._last_lat
            bal = self._balance_penalty()
            reward = d_tp - self._reward_latency_weight * d_lat - self._balance_penalty_weight * bal
            # 奖励裁剪（避免极端值）
            reward = float(np.clip(reward, -1.0, 1.0))

            self._dqn.push(self._last_state_norm, int(self._last_action), reward, s_norm, False)
            loss, grad_norm = self._dqn.train_step()

            if self._debug_dump:
                q_vals = self._dqn.q(self._last_state_norm[None, :])[0]
                q_print = np.nan_to_num(q_vals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                logger.info(
                    "[DQN:step=%d] eps=%.4f act=%d "
                    "inst_tp_prev=%.5f inst_tp_now=%.5f d_tp=%.5f "
                    "inst_lat_prev=%.5f inst_lat_now=%.5f d_lat=%.5f "
                    "bal=%.5f reward=%.6f loss=%.6f grad_norm=%.4f q=%s",
                    self._step,
                    self._dqn.eps,
                    self._last_action,
                    self._last_tp, tp_now, d_tp,
                    self._last_lat, lat_now, d_lat,
                    bal,
                    reward,
                    loss,
                    grad_norm,
                    np.array2string(q_print, precision=4, floatmode="fixed"),
                )

        # 选动作（数值安全）
        a = self._dqn.act(s_norm)
        rid = self._replica_ids[a]

        # 出队 1 个请求
        self.sort_requests()
        req = self._request_queue.pop(0)
        mapping = [(rid, req)]

        # 缓存
        self._last_state = s
        self._last_state_norm = s_norm
        self._last_action = a

        self._last_tp, self._last_lat = self._metrics_window()
        self._step += 1

        return mapping

    # （仍保留：如 state 里需要累计 tp/lat 或外部用得到，可继续使用）
    def _metrics(self) -> Tuple[float, float]:
        if getattr(self, "_metric_store", None) is None:
            return 0.0, 0.0
        try:
            tp = float(self._metric_store.get_throughput(self._current_time))
        except Exception:
            tp = 0.0
        try:
            lat = float(self._metric_store.get_average_latency())
        except Exception:
            lat = 0.0
        return tp, lat

    def _balance_penalty(self) -> float:
        """利用率极差（鼓励分流）。"""
        utils: List[float] = []
        for rid in self._replica_ids:
            rs = self.get_replica_scheduler(rid)
            num_alloc = float(getattr(rs, "_num_allocated_blocks", 0))
            cfg = getattr(rs, "_config", None)
            num_blocks = float(getattr(cfg, "num_blocks", 1) if cfg else 1)
            num_blocks = max(1.0, num_blocks)
            utils.append(num_alloc / num_blocks)
        return float(max(utils) - min(utils)) if utils else 0.0


# 可选注册（若已在 registry 显式注册，可忽略）
try:
    from vidur.scheduler.global_scheduler.global_scheduler_registry import GlobalSchedulerRegistry
    from vidur.types import GlobalSchedulerType
    GlobalSchedulerRegistry.register(GlobalSchedulerType.GLOBALSCHEDULEONLINE, DQNGlobalSchedulerOnline)
except Exception:
    pass



