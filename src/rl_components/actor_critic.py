"""兼容性重定向 - actor_critic已迁移到src.core.models"""
import warnings
warnings.warn("Use 'from src.core.models.actor_critic import ActorCritic' instead", DeprecationWarning)
from ..core.models.actor_critic import *