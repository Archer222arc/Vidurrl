# file: vidur/scheduler/global_scheduler/ppo_global_scheduler_online.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from vidur.entities import Replica, Request
from vidur.config import SimulationConfig
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler_with_state import (
    build_global_state,
    debug_dump_replica_state,
)

logger = init_logger(__name__)


# ---------- Utils ----------
class _RunningNormalizer:
    """Welford 归一化器 + 裁剪，数值更稳。"""
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
            z = x / (np.linalg.norm(x) + 1.0)
        else:
            var = self.m2 / (self.count - 1)
            std = np.sqrt(np.maximum(var, self.eps))
            z = (x - self.mean) / std
        return np.clip(z, -self.clip, self.clip).astype(np.float32, copy=False)


def _init_layer(layer: nn.Module, gain: float = 1.0, use_orthogonal: bool = True):
    if isinstance(layer, nn.Linear):
        if use_orthogonal:
            nn.init.orthogonal_(layer.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


# ---------- Actor-Critic (MLP + GRU) ----------
class _ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        layer_N: int = 2,      # 额外 MLP 层数（总 3 层）
        gru_layers: int = 2,
        use_orthogonal: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.layer_N = layer_N
        self.gru_layers = gru_layers

        # MLP encoder: Linear -> (LayerNorm + ReLU) x (1 + layer_N)
        mlp = []
        mlp.append(nn.Linear(state_dim, hidden_size))
        self.mlp_ln0 = nn.LayerNorm(hidden_size)
        _init_layer(mlp[-1], gain=math.sqrt(2), use_orthogonal=use_orthogonal)

        self.mlp_h = nn.ModuleList()
        self.mlp_ln = nn.ModuleList()
        for _ in range(layer_N):
            self.mlp_h.append(nn.Linear(hidden_size, hidden_size))
            self.mlp_ln.append(nn.LayerNorm(hidden_size))
            _init_layer(self.mlp_h[-1], gain=math.sqrt(2), use_orthogonal=use_orthogonal)
        self.mlp_in = nn.Sequential(*mlp)

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=False,   # use (T,N,H)
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        self.gru_ln = nn.LayerNorm(hidden_size)

        # Heads
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        _init_layer(self.actor, gain=0.01, use_orthogonal=use_orthogonal)
        _init_layer(self.critic, gain=1.0, use_orthogonal=use_orthogonal)

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_in(x)
        x = F.relu(self.mlp_ln0(x))
        for i in range(self.layer_N):
            x = F.relu(self.mlp_ln[i](self.mlp_h[i](x)))
        return x

    def forward_gru(
        self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, H) -> (T=1, N, H)
        x = x.unsqueeze(0)
        # masks: (N, 1) with 0 indicates reset hidden
        if hxs is not None and masks is not None:
            inv_masks = (1.0 - masks).view(1, -1, 1)
            hxs = hxs * masks.view(1, -1, 1)
            hxs = hxs + torch.zeros_like(hxs) * inv_masks
        out, hxs = self.gru(x, hxs)
        out = out.squeeze(0)  # (N,H)
        out = self.gru_ln(out)
        return out, hxs

    def act_value(
        self, s: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.forward_mlp(s)
        z, hxs = self.forward_gru(z, hxs, masks)
        logits = self.actor(z)
        v = self.critic(z).squeeze(-1)  # (N,)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v, hxs

    def evaluate_actions(
        self, s: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.forward_mlp(s)
        z, hxs = self.forward_gru(z, hxs, masks)
        logits = self.actor(z)
        v = self.critic(z).squeeze(-1)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(a)
        entropy = dist.entropy().mean()
        return logp, entropy, v, hxs


# ---------- PPO buffer & trainer ----------
class _RolloutBuffer:
    def __init__(self, state_dim: int, rollout_len: int, gamma: float, gae_lambda: float, device: str = "cpu"):
        self.rollout_len = rollout_len
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.device = device
        self.reset(state_dim)

    def reset(self, state_dim: int):
        self.s = []
        self.a = []
        self.logp = []
        self.v = []
        self.r = []
        self.masks = []  # 1: not done, 0: reset hidden (float)
        self.ptr = 0

    def add_step(self, s, a, logp, v, r, mask):
        self.s.append(s)
        self.a.append(a)                 # Tensor 标量 (1,) 或 0-D
        self.logp.append(logp)           # (1,)
        # 价值存为 (1,)
        if torch.is_tensor(v):
            self.v.append(v.view(-1)[0:1])
        else:
            self.v.append(torch.tensor([float(v)], dtype=torch.float32))
        self.r.append(r)
        # mask 统一成 float
        self.masks.append(float(mask.item() if torch.is_tensor(mask) else mask))
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.rollout_len

    def compute_gae(self, last_v: torch.Tensor):
        s = torch.stack(self.s, dim=0).to(self.device)                        # (T, D)
        a = torch.stack(self.a, dim=0).view(-1).to(torch.long).to(self.device)  # (T,)
        logp = torch.stack(self.logp, dim=0).view(-1).to(self.device)         # (T,)
        v = torch.stack(self.v, dim=0).to(self.device)                        # (T,1) -> (T,)
        if v.dim() == 2 and v.size(-1) == 1:
            v = v.squeeze(-1)
        r = torch.tensor(self.r, dtype=torch.float32, device=self.device)     # (T,)
        m = torch.tensor(self.masks, dtype=torch.float32, device=self.device) # (T,)

        # last_v -> 标量
        if torch.is_tensor(last_v):
            last_v = last_v.view(-1)[0]

        T = v.shape[0]
        adv = torch.zeros(T, device=self.device)
        last_gae = torch.tensor(0.0, device=self.device)
        for t in reversed(range(T)):
            v_next = last_v if t == T - 1 else v[t + 1]
            delta = r[t] + self.gamma * v_next * m[t] - v[t]
            last_gae = delta + self.gamma * self.lmbda * m[t] * last_gae
            adv[t] = last_gae
        ret = adv + v
        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return s, a, logp, v, ret, adv


class _PPOTrainer:
    def __init__(
        self,
        policy: _ActorCritic,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, s, a, logp_old, v_old, ret, adv, masks, hxs_init) -> Dict[str, float]:
        N = s.shape[0]
        idx = np.arange(N)

        pi_losses, vf_losses, entropies = [], [], []
        kls, clipfracs, gradnorms, evs = [], [], [], []

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for i0 in range(0, N, self.minibatch_size):
                j = idx[i0 : i0 + self.minibatch_size]
                bs = s[j]
                ba = a[j].view(-1).to(torch.long)         # 保证 1D 整型
                blogp = logp_old[j]
                bret = ret[j]
                badv = adv[j]
                bm = masks[j]

                # 简化：每个样本都从相同初始 hxs 开始（在线调度足够）
                with torch.no_grad():
                    hxs = hxs_init.clone().detach()

                new_logp, entropy, v_pred, _ = self.policy.evaluate_actions(bs, hxs, bm.unsqueeze(-1), ba)

                ratio = torch.exp(new_logp - blogp)
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * badv
                pi_loss = -torch.min(surr1, surr2).mean()

                v_clipped = v_old[j] + (v_pred - v_old[j]).clamp(-self.clip_ratio, self.clip_ratio)
                vf_loss1 = (v_pred - bret).pow(2)
                vf_loss2 = (v_clipped - bret).pow(2)
                vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                loss = pi_loss + self.value_coef * vf_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()

                # 反传后（裁剪前）梯度范数
                total_sq = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        g = p.grad.data
                        total_sq += float(torch.sum(g * g))
                grad_norm = float(math.sqrt(total_sq)) if total_sq > 0 else 0.0

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

                # 统计
                with torch.no_grad():
                    approx_kl = torch.mean(blogp - new_logp).clamp(min=0).item()
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_ratio).float()).item()
                    var_y = torch.var(bret, unbiased=False)
                    ev = (1.0 - torch.var(bret - v_pred, unbiased=False) / (var_y + 1e-8)).item()

                pi_losses.append(pi_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.item())
                kls.append(approx_kl)
                clipfracs.append(clipfrac)
                gradnorms.append(grad_norm)
                evs.append(ev)

        lr = self.opt.param_groups[0]["lr"]
        stats = {
            "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "vf_loss": float(np.mean(vf_losses)) if vf_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "pg_grad_norm": float(np.mean(gradnorms)) if gradnorms else 0.0,
            "explained_var": float(np.mean(evs)) if evs else 0.0,
            "lr": float(lr),
        }
        return stats


# ---------- Scheduler (PPO online) ----------
class PPOGlobalSchedulerOnline(BaseGlobalScheduler):
    """
    在线 PPO 路由器（MLP+GRU）：
      支持两种奖励模式（通过 Enum 或字符串传入）：
        1) delta   : r = Δtp - λ_lat * Δlat - λ_bal * balance_penalty
        2) instant : r = tp  - λ_lat *  lat - λ_bal * balance_penalty
    """
    TYPE = "ppoonline"

    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]) -> None:
        super().__init__(config, replicas)
        self._replica_ids: List[int] = sorted(list(replicas.keys()))

        gcfg = getattr(getattr(config, "cluster_config", None), "global_scheduler_config", None)
        def _get(k, d): return getattr(gcfg, k, d) if gcfg is not None else d

        # 超参
        self._hidden_size    = int(_get("hidden_size", 128))
        self._layer_N        = int(_get("layer_N", 2))
        self._gru_layers     = int(_get("gru_layers", 2))
        self._lr             = float(_get("lr", 3e-4))
        self._gamma          = float(_get("gamma", 0.95))
        self._gae_lambda     = float(_get("gae_lambda", 0.95))
        self._clip_ratio     = float(_get("clip_ratio", 0.2))
        self._entropy_coef   = float(_get("entropy_coef", 0.01))
        self._value_coef     = float(_get("value_coef", 0.5))
        self._epochs         = int(_get("epochs", 4))
        self._rollout_len    = int(_get("rollout_len", 64))
        self._minibatch_size = int(_get("minibatch_size", 64))
        self._max_grad_norm  = float(_get("max_grad_norm", 0.5))

        # 与 DQN 保持一致的权重命名
        self._reward_latency_weight   = float(_get("reward_latency_weight", 1.0))
        self._balance_penalty_weight  = float(_get("balance_penalty_weight", 0.0))
        self._max_queue_requests      = int(_get("max_queue_requests_per_replica", 4))
        self._debug_dump              = bool(_get("debug_dump_global_state", False))

        # 奖励模式：兼容 Enum 或 str
        mode = _get("reward_mode", "delta")
        if hasattr(mode, "value"):  # Enum 转字符串
            mode = mode.value
        self._reward_mode = str(mode).lower()
        if self._reward_mode not in ("delta", "instant"):
            logger.warning("Unknown reward_mode=%s, fallback to 'delta'", self._reward_mode)
            self._reward_mode = "delta"

        self._device = "cpu"
        self._norm = _RunningNormalizer(eps=1e-6, clip=5.0)

        # 用一份初始状态确定维度
        s0 = build_global_state(
            self._replicas, self.get_replica_scheduler, 0.0, None,
            max_queue_requests=self._max_queue_requests
        )
        self._norm.update(s0)
        s0n = self._norm.normalize(s0)
        state_dim = int(s0n.shape[0])
        action_dim = len(self._replica_ids)

        self._ac = _ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=self._hidden_size,
            layer_N=self._layer_N,
            gru_layers=self._gru_layers,
            use_orthogonal=True,
        ).to(self._device)

        self._ppo = _PPOTrainer(
            self._ac,
            lr=self._lr,
            clip_ratio=self._clip_ratio,
            entropy_coef=self._entropy_coef,
            value_coef=self._value_coef,
            epochs=self._epochs,
            minibatch_size=self._minibatch_size,
            max_grad_norm=self._max_grad_norm,
            device=self._device,
        )

        self._buf = _RolloutBuffer(
            state_dim=state_dim,
            rollout_len=self._rollout_len,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            device=self._device,
        )

        # GRU hidden (L, N(=1), H)
        self._hxs = torch.zeros(self._gru_layers, 1, self._hidden_size, device=self._device)

        # 训练缓存（上一时刻全局度量）
        self._last_tp: float = 0.0
        self._last_lat: float = 0.0
        self._step: int = 0

        logger.info(
            "[PPO:init] mode=%s | h=%d L=%d GRU=%d | lr=%.2e gamma=%.3f lam=%.3f "
            "clip=%.2f ent=%.3f vf=%.3f epochs=%d roll=%d mb=%d gnorm=%.2f | lat_w=%.3f bal_w=%.3f",
            self._reward_mode, self._hidden_size, self._layer_N, self._gru_layers,
            self._lr, self._gamma, self._gae_lambda, self._clip_ratio,
            self._entropy_coef, self._value_coef, self._epochs, self._rollout_len,
            self._minibatch_size, self._max_grad_norm, self._reward_latency_weight, self._balance_penalty_weight
        )
        logger.info(
            "[PPO:init] state_dim=%d sample(min/mean/max)=%.4f/%.4f/%.4f",
            s0.shape[0], float(np.min(s0)), float(np.mean(s0)), float(np.max(s0))
)

    # ------- helpers -------
    def _metrics_now(self) -> Tuple[float, float]:
        """当前全局吞吐与平均时延（与 DQN 相同接口）。"""
        tp = 0.0
        lat = 0.0
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
        utils: List[float] = []
        for rid in self._replica_ids:
            rs = self.get_replica_scheduler(rid)
            num_alloc = float(getattr(rs, "_num_allocated_blocks", 0))
            cfg = getattr(rs, "_config", None)
            num_blocks = float(getattr(cfg, "num_blocks", 1) if cfg else 1)
            num_blocks = max(1.0, num_blocks)
            utils.append(num_alloc / num_blocks)
        return float(max(utils) - min(utils)) if utils else 0.0

    def _compute_reward(self, tp_now: float, lat_now: float) -> Tuple[float, float, float, float]:
        """根据 reward_mode 计算奖励。返回 (r, x_tp, x_lat, bal)；x_* 用于日志展示。"""
        bal = self._balance_penalty()
        if self._reward_mode == "delta":
            d_tp  = tp_now - self._last_tp
            d_lat = lat_now - self._last_lat
            r = d_tp - self._reward_latency_weight * d_lat - self._balance_penalty_weight * bal
            r = float(np.clip(r, -1.0, 1.0))
            return r, d_tp, d_lat, bal
        # instant
        r = tp_now - self._reward_latency_weight * lat_now - self._balance_penalty_weight * bal
        r = float(np.tanh(r / 10.0))  
        return r, tp_now, lat_now, bal

    # ------- main -------
    def schedule(self) -> List[Tuple[int, Request]]:
        if not self._request_queue:
            return []

        if self._debug_dump:
            for rid in self._replica_ids:
                rs = self.get_replica_scheduler(rid)
                debug_dump_replica_state(rid, rs)

        # 1) 采状态 + 归一化
        s_np = build_global_state(
            self._replicas,
            self.get_replica_scheduler,
            self._current_time,
            self._metric_store,
            max_queue_requests=self._max_queue_requests,
        )
        self._norm.update(s_np)
        s_norm = self._norm.normalize(s_np)
        s = torch.from_numpy(s_norm).float().unsqueeze(0).to(self._device)  # (1, D)
        if self._debug_dump:
            # ---- 精准打印当前副本的 state 切片，校验每个请求 7 个字段 ----
            try:
                if self._debug_dump:
                    RAW = s_np  # 用未归一化的
                    BASE_LEN = 11
                    REQ_FEATS = 7
                    K = int(self._max_queue_requests)
                    stride = BASE_LEN + K * REQ_FEATS

                    # state 拼接时用的是排序后的 rid 顺序
                    for rid_idx, rid in enumerate(self._replica_ids):
                        base_start = rid_idx * stride
                        base_slice = RAW[base_start : base_start + BASE_LEN]
                        logger.info(
                            "[STATE:rid=%d] base(%d)=%s",
                            rid, BASE_LEN,
                            np.array2string(base_slice, formatter={"float_kind": lambda x: f"{x: .3e}"})
                        )

                        # 打印该副本队列前2条（或 <=K）
                        rs = self.get_replica_scheduler(rid)
                        rq = getattr(rs, "_request_queue", [])
                        for j, req in enumerate(rq[: min(2, K)]):
                            off = base_start + BASE_LEN + j * REQ_FEATS
                            req_slice = RAW[off : off + REQ_FEATS]
                            age, prefill, processed, prefill_done, completed, decode, prio_state = req_slice.tolist()
                            prio_actual = float(getattr(req, "priority", 0.0) or 0.0)
                            logger.info(
                                "[STATE:rid=%d req#%d id=%s] "
                                "[age,prefill,processed,prefill_done,completed,decode,priority]=%s | priority(req)=%.3f",
                                rid, j, getattr(req, "id", None),
                                np.array2string(req_slice, formatter={"float_kind": lambda x: f"{x: .3e}"}),
                                prio_actual,
                            )
                            # 一致性校验
                            if abs(prio_state - prio_actual) > 1e-6:
                                logger.warning(
                                    "STATE priority mismatch! rid=%d j=%d state=%.6f actual=%.6f",
                                    rid, j, prio_state, prio_actual
                                )
            except Exception:
                pass


        # 2) 基于“上一动作”的奖励（delta/instant）
        tp_now, lat_now = self._metrics_now()
        r, x1, x2, bal = self._compute_reward(tp_now, lat_now)

        # mask：当系统明显 idle 时重置 GRU（这里基本总是 1，因为有请求才会进入）
        idle = 1 if (len(self._request_queue) == 0) else 0
        mask = torch.tensor([1.0 - float(idle)], dtype=torch.float32, device=self._device).unsqueeze(-1)  # (1,1)

        # 回写上一步的 reward/mask
        if self._buf.ptr > 0 and self._buf.ptr <= self._rollout_len:
            self._buf.r[-1] = float(r)
            self._buf.masks[-1] = float(mask.item())

        # 3) 策略前向，采样动作 + 价值
        with torch.no_grad():
            a, logp, v, self._hxs = self._ac.act_value(s, self._hxs, mask)
        a_i = int(a.item())
        rid = self._replica_ids[a_i]

        # 4) 存入 buffer（r 先占位 0，下一步 schedule 再用新指标回填）
        self._buf.add_step(
            s.squeeze(0),                          # s: (D,)
            a.detach(),                            # a
            logp.detach(),                         # logp
            v.detach(),                            # v  (1,)
            0.0,                                   # r placeholder
            mask.squeeze(-1).detach(),             # mask: (1,)
        )

        # 5) 触发 PPO 更新
        if self._buf.is_full():
            with torch.no_grad():
                # Bootstrap 也走 GRU，保持一致性
                z = self._ac.forward_mlp(s)                 # (1,H)
                z, _ = self._ac.forward_gru(z, self._hxs, mask)
                last_v = self._ac.critic(z).squeeze(-1)     # (1,)
                last_v = last_v.detach()

            s_t, a_t, logp_t, v_t, ret_t, adv_t = self._buf.compute_gae(last_v)
            m_tensor = torch.tensor(self._buf.masks, dtype=torch.float32, device=self._device)  # (T,)

            stats = self._ppo.update(
                s_t, a_t, logp_t, v_t, ret_t, adv_t,
                masks=m_tensor,
                hxs_init=self._hxs.detach(),
            )

            # ------ rollout 统计 + 打点（每次更新都打印） ------
            with torch.no_grad():
                r_hist = np.asarray(self._buf.r, dtype=np.float32)
                T = r_hist.shape[0]
                R_mean, R_std = float(np.mean(r_hist)), float(np.std(r_hist))
                R_min, R_max = float(np.min(r_hist)), float(np.max(r_hist))
                Adv_std = float(adv_t.std(unbiased=False).item())
                V_mean = float(v_t.mean().item())
                Ret_mean = float(ret_t.mean().item())
                mask_mean = float(m_tensor.mean().item())
                # 动作分布（确保 1D int64）
                a_np = a_t.view(-1).to(torch.int64).cpu().numpy()
                act_hist = np.bincount(a_np, minlength=len(self._replica_ids)).tolist()

            logger.info(
                "[PPO:update] step=%d len=%d "
                "pi=%.6f vf=%.6f ent=%.4f kl=%.5f clip=%.3f ev=%.3f gnorm=%.3f lr=%.2e "
                "R(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f Adv_std=%.4f V/Ret=%.4f/%.4f mask=%.3f act=%s",
                self._step, T,
                stats["pi_loss"], stats["vf_loss"], stats["entropy"],
                stats["approx_kl"], stats["clipfrac"], stats["explained_var"],
                stats["pg_grad_norm"], stats["lr"],
                R_mean, R_std, R_min, R_max, Adv_std, V_mean, Ret_mean, mask_mean,
                act_hist,
            )

            self._buf.reset(state_dim=s.shape[-1])

        # 6) 选择请求出队
        self.sort_requests()
        req = self._request_queue.pop(0)
        mapping = [(rid, req)]

        # 7) 调试日志（打印指标）
        if self._debug_dump:
            if self._reward_mode == "delta":
                logger.info(
                    "[PPO:step=%d] mode=delta act=%d d_tp=%.5f d_lat=%.5f bal=%.5f reward=%.6f",
                    self._step, a_i, x1, x2, bal, r
                )
            else:
                logger.info(
                    "[PPO:step=%d] mode=instant act=%d tp=%.5f lat=%.5f bal=%.5f reward=%.6f",
                    self._step, a_i, x1, x2, bal, r
                )

        # 7.1) 轻量进度打点：每 10 步看一次 rollout 进度
        try:
            if (self._step % 10) == 0:
                last_r = float(self._buf.r[-1]) if len(self._buf.r) > 0 else float("nan")
                logger.info(
                    "[PPO:rollout] step=%d buf_ptr=%d/%d last_r=%.6f",
                    self._step, self._buf.ptr, self._rollout_len, last_r
                )
        except Exception:
            pass

        # 8) 更新“上一时刻全局度量”
        self._last_tp, self._last_lat = tp_now, lat_now
        self._step += 1
        return mapping


try:
    from vidur.scheduler.global_scheduler.global_scheduler_registry import GlobalSchedulerRegistry
    from vidur.types import GlobalSchedulerType
    # 与 DQN 共存，给 PPO 一个新类型
    GlobalSchedulerRegistry.register(GlobalSchedulerType.PPOONLINE, PPOGlobalSchedulerOnline)


except Exception:
    pass




