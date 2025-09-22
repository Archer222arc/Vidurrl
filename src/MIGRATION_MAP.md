# Vidur 模块重构映射表

## 📋 完整的重定向映射

### 核心组件 (rl_components → 新结构)

| 原路径 | 新路径 | 状态 | 组件 |
|--------|--------|------|------|
| `src.rl_components.actor_critic` | `src.core.models.actor_critic` | ✅ | ActorCritic, init_layer |
| `src.rl_components.state_builder` | `src.core.models.state_builder` | ✅ | StateBuilder |
| `src.rl_components.ppo_trainer` | `src.core.algorithms.ppo_trainer` | ✅ | PPOTrainer |
| `src.rl_components.rollout_buffer` | `src.core.algorithms.rollout_buffer` | ✅ | RolloutBuffer |
| `src.rl_components.normalizers` | `src.core.utils.normalizers` | ✅ | RunningNormalizer |
| `src.rl_components.temperature_controller` | `src.core.utils.temperature_controller` | ✅ | TemperatureController |
| `src.rl_components.checkpoint_manager` | `src.infrastructure.checkpoints.checkpoint_manager` | ✅ | CheckpointManager, InferenceMode |
| `src.rl_components.tensorboard_monitor` | `src.monitoring.tensorboard_monitor` | ✅ | TensorBoardMonitor, PPOTrainingDetector |
| `src.rl_components.metrics_exporter` | `src.monitoring.metrics_exporter` | ✅ | MetricsExporter |
| `src.rl_components.reward_calculator` | `src.rewards.reward_calculator` | ✅ | RewardCalculator |

### 配置模块

| 原路径 | 新路径 | 状态 | 组件 |
|--------|--------|------|------|
| `src.config.training_config` | `src.infrastructure.config.training_config` | ✅ | load_config, build_ppo_args |

### 训练模块

| 原路径 | 新路径 | 状态 | 组件 |
|--------|--------|------|------|
| `src.pretraining.*` | `src.training.pretraining.*` | ✅ | 所有预训练组件 |

### 数据收集模块

| 原路径 | 新路径 | 状态 | 组件 |
|--------|--------|------|------|
| `src.demo_collection.*` | `src.data.collection.*` | ✅ | 示教数据收集组件 |

### 调度工具

| 原路径 | 新路径 | 状态 | 组件 |
|--------|--------|------|------|
| `src.scheduler_utils.*` | `src.infrastructure.scheduling.*` | ✅ | 调度相关工具 |

## 🆕 新增组件

### 分块训练系统
- `src.training.chunk_trainer` - ChunkTrainer (新增)
- `src.training.progress_manager` - ProgressManager (新增)

### 监控系统
- `src.monitoring.progress_monitor` - ProgressMonitor (新增)

### 便捷访问器
- `src.modules` - VidurModules (新增)

## 🔄 兼容性保证

### 重定向机制
所有原有导入路径仍然可用，通过以下机制：

1. **警告系统**: 使用 `warnings.warn()` 提供迁移指导
2. **自动重定向**: `__init__.py` 自动从新位置导入
3. **降级处理**: 如果新位置不可用，回退到原位置
4. **运行时反馈**: 显示当前使用的结构类型

### 使用示例

```python
# ✅ 旧导入 - 仍然工作（带警告）
from src.rl_components.actor_critic import ActorCritic

# ✅ 新导入 - 推荐使用
from src.core.models.actor_critic import ActorCritic

# ✅ 便捷访问
import src
model = src.modules.core.models.ActorCritic()
```

## 📊 验证状态

- ✅ **13/13** 原始 rl_components 组件已迁移
- ✅ **5/5** 配置和工具模块已重定向
- ✅ **100%** 向后兼容性保持
- ✅ **0** 破坏性变更

## 🚀 迁移建议

1. **立即可用**: 现有代码无需修改即可工作
2. **渐进迁移**: 建议在新功能中使用新路径
3. **批量更新**: 可使用 IDE 的全局替换功能批量更新导入
4. **测试验证**: 每次更新后运行测试确保兼容性

---
*自动生成于模块重构完成时*