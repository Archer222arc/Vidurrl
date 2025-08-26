# file: vidur/scheduler/global_scheduler/random_global_scheduler_with_state.py
from __future__ import annotations

from random import choice
from typing import Any, Dict, List, Tuple

import numpy as np

from vidur.entities import Replica, Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


def _safe_get(obj: Any, name: str, default: Any) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _get_replica_queue(rs: Any) -> List[Request]:
    """优先走公开 API，其次访问内部字段，最后回退空列表。"""
    get_q = _safe_get(rs, "get_queue", None)
    if callable(get_q):
        try:
            return list(get_q())
        except Exception:
            pass
    return list(_safe_get(rs, "_request_queue", [])) or []


def _normalize_priority(p: Any) -> float:
    """尽量把 priority 归一为 float：支持 None/int/float/Enum/str。"""
    if p is None:
        return 0.0
    # Enum(value) 或普通数字
    try:
        v = getattr(p, "value", p)
        return float(v)
    except Exception:
        pass
    # 常见字符串兜底
    try:
        s = str(p).strip().lower()
        table = {
            "low": 1.0, "normal": 2.0, "med": 2.0, "medium": 2.0,
            "high": 3.0, "urgent": 4.0, "critical": 5.0,
        }
        return table.get(s, 0.0)
    except Exception:
        return 0.0


def dump_request_like(r: Request) -> str:
    """仅打印关注字段，避免属性缺失报错。"""
    fields = [
        "id",
        "arrived_at",
        "num_prefill_tokens",
        "num_processed_tokens",
        "is_prefill_complete",
        "completed",
        "num_decode_tokens",
        "priority",
    ]
    kv = ", ".join(f"{k}={getattr(r, k, None)}" for k in fields)
    return "    " + kv


def debug_dump_replica_state(replica_id: int, rs: Any) -> None:
    """与副本调度器端打印对齐，便于排查。"""
    rq: List[Request] = _get_replica_queue(rs)
    num_alloc: int = int(_safe_get(rs, "_num_allocated_blocks", 0))
    cfg = _safe_get(rs, "_config", None)
    num_blocks: int = int(getattr(cfg, "num_blocks", 0)) if cfg else 0
    preempted: List[Request] = _safe_get(rs, "_preempted_requests", [])
    alloc_map = _safe_get(rs, "_allocation_map", {})

    print(f"[Replica {replica_id}] 调度前状态：")
    print(f"  请求队列长度: {len(rq)}")
    print(f"  已分配blocks: {num_alloc}/{num_blocks}")
    print(f"  预占用请求数: {len(preempted)}")
    print(f"  allocation_map: {alloc_map}")
    print(f"[Replica {replica_id}] 队列明细（len={len(rq)}）:")
    for req in rq:
        print(dump_request_like(req))
    print(f"[Replica {replica_id}] _preempted_requests（len={len(preempted)}）:")
    for r in preempted:
        print(dump_request_like(r))


def build_replica_state(
    replica_id: int, rs: Any, current_time: float, max_queue_requests: int
) -> List[float]:
    """
    单副本状态向量：
      - 资源/运行时基础特征 11 维
      - 队列头 K 的请求特征，每个请求 **7 维**（与现有日志解码一致）：
          [age, num_prefill, num_processed, remaining_prefill, completed, num_decode, priority]
    """
    vec: List[float] = []
    rq: List[Request] = _get_replica_queue(rs)
    num_alloc: int = int(_safe_get(rs, "_num_allocated_blocks", 0))
    num_running_batches: int = int(_safe_get(rs, "_num_running_batches", 0))
    preempted: List[Request] = _safe_get(rs, "_preempted_requests", [])
    alloc_map = _safe_get(rs, "_allocation_map", {})

    cfg = _safe_get(rs, "_config", None)
    num_blocks: int = int(getattr(cfg, "num_blocks", 1)) if cfg else 1
    num_blocks = max(1, num_blocks)
    block_size: int = int(getattr(cfg, "block_size", 1)) if cfg else 1
    batch_cap: int = int(getattr(cfg, "batch_size_cap", 1)) if cfg else 1
    num_stages: int = int(_safe_get(rs, "_num_stages", 0))

    alloc_frac = float(num_alloc) / float(num_blocks)
    avail_frac = float(max(num_blocks - num_alloc, 0)) / float(num_blocks)

    # 基础特征（与历史保持顺序）
    vec.extend(
        [
            float(len(rq)),            # 队列长度
            float(num_alloc),          # 已分配 blocks
            float(num_blocks),         # 总 blocks
            float(alloc_frac),         # 利用率
            float(avail_frac),         # 空闲率
            float(num_running_batches),
            float(len(preempted)),
            float(len(alloc_map)),
            float(batch_cap),
            float(block_size),
            float(num_stages),
        ]
    )

    # 每请求特征（7 维）
    K = int(max_queue_requests)
    for i in range(K):
        if i < len(rq):
            req = rq[i]
            arrived_at = float(getattr(req, "arrived_at", getattr(req, "_arrived_at", 0.0)) or 0.0)
            age = max(0.0, float(current_time) - arrived_at)

            num_prefill = float(getattr(req, "num_prefill_tokens", 0) or 0.0)
            num_processed = float(getattr(req, "num_processed_tokens", 0) or 0.0)
            remaining_prefill = max(0.0, num_prefill - num_processed)

            completed = 1.0 if bool(getattr(req, "completed", False)) else 0.0
            num_decode = float(getattr(req, "num_decode_tokens", 0) or 0.0)
            priority = _normalize_priority(getattr(req, "priority", None))

            vec.extend(
                [
                    age,
                    num_prefill,
                    num_processed,
                    remaining_prefill,   # 对应日志里的 prefill_done 一列
                    completed,
                    num_decode,
                    priority,            # 放在最后一列，避免错读
                ]
            )
        else:
            # 7 维补零
            vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return vec


def build_global_state(
    replicas: Dict[int, Replica],
    get_rs,
    current_time: float,
    metric_store,
    max_queue_requests: int,
) -> np.ndarray:
    """
    全局状态 = 各副本状态拼接 + [global_queue_len, throughput(t), latency(avg)]
    """
    parts: List[float] = []

    for rid in sorted(replicas.keys()):
        rs = get_rs(rid)
        parts.extend(build_replica_state(rid, rs, current_time, max_queue_requests))

    # 全局特征：全局队列长度（尽量从 scheduler 实例读取，失败回退 0）
    global_queue_len = 0.0
    try:
        sched = getattr(get_rs, "__self__", None)
        if sched is not None:
            global_queue_len = float(len(getattr(sched, "_request_queue", []) or []))
    except Exception:
        global_queue_len = 0.0

    # 指标
    def _metrics_pair():
        if metric_store is None:
            return (0.0, 0.0)
        try:
            tp = float(metric_store.get_throughput(current_time))
        except Exception:
            tp = 0.0
        try:
            lat = float(metric_store.get_average_latency())
        except Exception:
            lat = 0.0
        return (tp, lat)

    tp, lat = _metrics_pair()
    parts.extend([global_queue_len, tp, lat])

    return np.asarray(parts, dtype=np.float32)


class RandomGlobalSchedulerWithState(BaseGlobalScheduler):
    """随机路由 + 可选调试打印 + 构建全局状态向量。"""

    TYPE = "random_with_state"

    def __init__(self, config, replicas: Dict[int, Replica]) -> None:
        super().__init__(config, replicas)
        self._replica_ids: List[int] = sorted(list(replicas.keys()))
        # 关键：从 cluster_config 取全局调度器配置
        gcfg = getattr(getattr(config, "cluster_config", None), "global_scheduler_config", None)
        self._debug_dump = bool(getattr(gcfg, "debug_dump_global_state", False))
        self._max_queue_requests = int(getattr(gcfg, "max_queue_requests_per_replica", 4))

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        if self._debug_dump:
            for rid in self._replica_ids:
                rs = self.get_replica_scheduler(rid)
                debug_dump_replica_state(rid, rs)

        # 构造一次全局 state（便于调试或日志）
        _ = build_global_state(
            self._replicas,
            self.get_replica_scheduler,
            self._current_time,
            self._metric_store,
            max_queue_requests=self._max_queue_requests,
        )

        mapping: List[Tuple[int, Request]] = []
        while self._request_queue:
            req = self._request_queue.pop(0)
            rid = choice(self._replica_ids)
            mapping.append((rid, req))
        return mapping


