"""兼容性重定向 - normalizers已迁移到src.core.utils"""
import warnings
warnings.warn("Use 'from src.core.utils.normalizers import RunningNormalizer' instead", DeprecationWarning)
from ..core.utils.normalizers import *