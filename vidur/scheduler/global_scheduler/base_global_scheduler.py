from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

        self._num_replicas = len(self._replicas)

        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=execution_time_predictor,
            )
            for replica_id, replica in replicas.items()
        }
        self._request_queue = []
        self._metric_store = None
        self._current_time = 0.0


    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )
    
    def set_runtime_context(self, current_time, metric_store):
        """把当前仿真时间和 MetricsStore 下发给所有副本调度器"""
        self._current_time = float(current_time)
        self._metric_store = metric_store

        if hasattr(self, "_replica_schedulers"):
            # 关键：遍历 values() 才能拿到各副本调度器实例
            for rs in self._replica_schedulers.values():
                if hasattr(rs, "set_runtime_context"):
                    rs.set_runtime_context(self._current_time, self._metric_store)
             # 打点：确认全局调度器本身被调用 & 向下转发了多少个
        #print(f"[ctx→global] t={self._current_time}, ms={'ok' if metric_store else 'None'}, fanout={fanout}")



    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
