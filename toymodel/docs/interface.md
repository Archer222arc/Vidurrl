# Toymodel 模块接口文档

## 📋 模块概述

**模块名称**: Toymodel
**功能定位**: M/M/1排队系统仿真，用于验证PPO路由策略
**维护状态**: ✅ 活跃开发
**创建时间**: 2025-10-01
**最后更新**: 2025-10-01

## 🎯 核心功能

Toymodel是一个简化的M/M/1排队仿真系统，用于：
1. 验证PPO算法能否学习最优路由策略
2. 对比不同调度策略（Oracle, Random, Round-Robin, Shortest Queue）的性能
3. 通过TensorBoard实时监控仿真指标

## 📁 模块结构

```
toymodel/
├── __init__.py              # 模块导出
├── config.py                # 配置管理
├── entities.py              # 核心实体（Request, Replica）
├── environment.py           # M/M/1仿真环境
├── request_generator.py     # Poisson到达过程生成器
├── tensorboard_monitor.py   # TensorBoard监控
├── run_simulation.py        # 仿真主程序
├── compare_schedulers.py    # Scheduler对比工具
├── schedulers/              # 调度策略
│   ├── __init__.py
│   ├── base.py             # 基础调度器
│   ├── oracle.py           # 最优策略
│   └── baselines.py        # 基线策略
└── scripts/                 # 运行脚本
    ├── run_toymodel.sh
    └── compare_schedulers.sh
```

## 🔌 主要接口

### 1. QueueEnvironment

**作用**: M/M/1排队环境仿真

**初始化参数**:
```python
QueueEnvironment(
    num_replicas: int,              # Replica数量（固定为2）
    service_rates: dict,            # 服务率 {replica_id: {req_type: rate}}
    arrival_rates: dict,            # 到达率 {req_type: rate}
    max_time: float,                # 最大仿真时间
    seed: int = None,               # 随机种子
    tensorboard_enabled: bool = False,  # TensorBoard监控
    tensorboard_log_dir: str = ...  # TensorBoard日志目录
)
```

**核心方法**:
```python
def run_simulation(
    routing_policy: Callable[[Request, list[Replica]], int]
) -> list[Request]:
    """运行完整仿真，返回完成的请求列表"""
```

**使用示例**:
```python
from toymodel import QueueEnvironment, load_config
from toymodel.schedulers import OracleScheduler

config = load_config('configs/toymodel/config.json')
env = QueueEnvironment(
    num_replicas=config.environment.num_replicas,
    service_rates=config.environment.service_rates,
    arrival_rates=config.environment.arrival_rates,
    max_time=config.environment.max_time,
    seed=config.experiment.seed,
    tensorboard_enabled=config.tensorboard.enabled,
)

scheduler = OracleScheduler(num_replicas=2)
completed = env.run_simulation(
    lambda req, reps: scheduler.schedule(req, reps)
)
```

### 2. 配置管理

**作用**: 从JSON加载和验证配置

**主要接口**:
```python
def load_config(config_path: str) -> ToyModelConfig:
    """加载并验证配置文件"""

@dataclass
class ToyModelConfig:
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    scheduler: SchedulerConfig
    metrics: MetricsConfig
    tensorboard: TensorBoardConfig
```

**配置结构**:
```json
{
  "experiment": {
    "name": "toymodel_experiment",
    "description": "...",
    "seed": 42
  },
  "environment": {
    "num_replicas": 2,
    "max_time": 500.0,
    "service_rates": {
      "0": {"0": 10.0, "1": 5.0},
      "1": {"0": 5.0, "1": 10.0}
    },
    "arrival_rates": {"0": 6.0, "1": 6.0}
  },
  "scheduler": {
    "type": "oracle",
    "options": {}
  },
  "tensorboard": {
    "enabled": true,
    "log_dir": "outputs/toymodel/tensorboard",
    "port": 6006,
    "clean_previous_runs": true
  }
}
```

### 3. Scheduler接口

**基类**: `BaseScheduler`

**必须实现的方法**:
```python
@abstractmethod
def schedule(self, request: Request, replicas: list[Replica]) -> int:
    """返回目标replica ID"""
```

**内置Scheduler**:
- `OracleScheduler`: 最优策略（Type A→Replica 0, Type B→Replica 1）
- `RandomScheduler`: 随机分配
- `RoundRobinScheduler`: 轮询分配
- `ShortestQueueScheduler`: 最短队列优先

### 4. TensorBoard监控

**作用**: 实时可视化仿真指标

**监控指标分组**:

**Routing/** - 路由指标
- `Type_{X}_to_Replica_{Y}_Ratio`: 路由分配比例
- `Type_{X}_Accuracy`: 路由准确率

**Latency/** - 时延指标
- `Replica_{X}_Mean`: 各replica平均时延
- `Replica_{X}_P99`: 各replica P99时延

**Queue/** - 队列指标
- `Replica_{X}_Length`: 队列长度
- `Replica_{X}_Utilization`: 利用率

**Metrics/** - 汇总指标
- `MeanLatency`, `P50Latency`, `P99Latency`
- `RoutingAccuracy`, `TotalCompleted`

**System/** - 系统指标
- `Time`, `CompletedRequests`, `RequestsInSystem`

## 🚀 快速开始

### 运行仿真

```bash
# 方式1: 使用脚本
./toymodel/scripts/run_toymodel.sh

# 方式2: 直接调用Python模块
python -m toymodel.run_simulation

# 比较所有scheduler
./toymodel/scripts/compare_schedulers.sh
```

### 修改配置

编辑 `configs/toymodel/config.json`:
- `arrival_rates`: 调整负载（如 `{"0": 7.0, "1": 7.0}` 测试高负载）
- `scheduler.type`: 切换策略（`oracle`/`random`/`round_robin`/`shortest_queue`）
- `max_time`: 调整仿真时长
- `tensorboard.enabled`: 启用/禁用监控

### 查看监控

```bash
# 自动启动（如果tensorboard.enabled=true）
# 访问 http://localhost:6006

# 手动停止
lsof -ti :6006 | xargs kill -9
```

## 📊 性能指标

**典型性能**（λ_A=λ_B=6.0, max_time=500）:

| Scheduler | Mean Latency | Routing Accuracy | 特点 |
|-----------|-------------|------------------|------|
| Oracle | 0.25 | 100% | 最优策略 |
| Shortest Queue | 0.91 | 51% | 次优 |
| Round-Robin | 1.05 | 50% | 均匀分配 |
| Random | 1.34 | 48% | 基线 |

## 🔄 扩展接口

### 添加新Scheduler

1. 继承 `BaseScheduler`
2. 实现 `schedule()` 方法
3. 在 `run_simulation.py` 的 `scheduler_map` 中注册

```python
# toymodel/schedulers/my_scheduler.py
from toymodel.schedulers.base import BaseScheduler

class MyScheduler(BaseScheduler):
    def schedule(self, request: Request, replicas: list[Replica]) -> int:
        # 自定义路由逻辑
        return selected_replica_id
```

### 自定义指标记录

```python
# 在environment中添加自定义指标
if self.tb_monitor and self.tb_monitor.writer:
    self.tb_monitor.writer.add_scalar(
        "Custom/MyMetric",
        value,
        step
    )
```

## ⚠️ 注意事项

1. **Replica数量固定为2**: 当前版本仅支持2个replica
2. **TensorBoard端口冲突**: 默认6006，可在config中修改
3. **Oracle的routing ratio是直线**: 这是正常现象，因为是确定性策略
4. **日志自动清理**: `clean_previous_runs=true` 会删除旧日志

## 🐛 已知问题

无

## 📚 相关文档

- **设计文档**: `docs/toymodel_ppo_routing_design.md`
- **规范文档**: `.claude/TOYMODEL.md`
- **演进历史**: `docs/modules/evolution/toymodel_evolution.md`
- **使用说明**: `toymodel/README.md`

## 📝 更新日志

### v1.0.0 (2025-10-01)
- ✅ M/M/1仿真环境
- ✅ Oracle/Random/RoundRobin/ShortestQueue scheduler
- ✅ TensorBoard实时监控
- ✅ JSON配置管理
- ✅ 细粒度指标记录（按type和replica）

---
**维护者**: Claude
**最后审核**: 2025-10-01
**下次审核**: 需要时
