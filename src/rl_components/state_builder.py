"""兼容性重定向 - state_builder已迁移到src.core.models"""
import warnings
warnings.warn("Use 'from src.core.models.state_builder import StateBuilder' instead", DeprecationWarning)
from ..core.models.state_builder import *