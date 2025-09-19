from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from vidur.types import GlobalSchedulerType
from vidur.utils.base_registry import BaseRegistry
from vidur.scheduler.global_scheduler.random_global_scheduler_with_state import RandomGlobalSchedulerWithState

from vidur.scheduler.global_scheduler.dqn_global_scheduler_online import (
    DQNGlobalSchedulerOnline,
)
from vidur.scheduler.global_scheduler.ppo_scheduler_modular import PPOGlobalSchedulerModular

class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
# 新注册（关键）
GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM_WITH_STATE, RandomGlobalSchedulerWithState)
GlobalSchedulerRegistry.register(GlobalSchedulerType.GLOBALSCHEDULEONLINE, DQNGlobalSchedulerOnline)
# 原版PPO使用原始实现（梯度问题解决方案）
from vidur.scheduler.global_scheduler.ppo_global_scheduler_online import PPOGlobalSchedulerOnline
GlobalSchedulerRegistry.register(GlobalSchedulerType.PPOONLINE, PPOGlobalSchedulerOnline)
GlobalSchedulerRegistry.register(GlobalSchedulerType.PPO_MODULAR, PPOGlobalSchedulerModular)