"""兼容性重定向 - ppo_trainer已迁移到src.core.algorithms"""
import warnings
warnings.warn("Use 'from src.core.algorithms.ppo_trainer import PPOTrainer' instead", DeprecationWarning)
from ..core.algorithms.ppo_trainer import *