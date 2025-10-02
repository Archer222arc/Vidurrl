# Toy Model - M/M/1 Queue Simulation

简化的排队系统仿真，用于验证PPO路由策略。

## 快速开始

### 运行仿真

```bash
# 使用默认配置运行
./toymodel/scripts/run_toymodel.sh

# 比较所有scheduler
./toymodel/scripts/compare_schedulers.sh
```

### 目录结构

```
toymodel/
├── src/                 # 核心代码
├── schedulers/          # 调度策略
├── scripts/             # 运行脚本
├── configs/             # 配置文件
├── outputs/             # 输出结果
├── tests/               # 单元测试
├── demo/                # 示例代码
└── docs/                # 模块文档
```

详细说明见 `README_STRUCTURE.md`

### 配置参数

编辑 `toymodel/configs/config.json` 修改参数：

```json
{
  "experiment": {
    "name": "toymodel_experiment",
    "seed": 42
  },
  "environment": {
    "num_replicas": 2,
    "max_time": 100.0,
    "service_rates": {
      "0": {"0": 10.0, "1": 5.0},
      "1": {"0": 5.0, "1": 10.0}
    },
    "arrival_rates": {
      "0": 6.0,
      "1": 6.0
    }
  },
  "scheduler": {
    "type": "oracle"
  },
  "tensorboard": {
    "enabled": true,
    "log_dir": "toymodel/outputs/tensorboard",
    "port": 6006
  }
}
```

## TensorBoard 监控

### 启用监控

在配置文件中设置 `tensorboard.enabled: true`，运行仿真时会自动：
1. Kill旧的TensorBoard进程（如果存在）
2. 启动新的TensorBoard服务器（后台运行）
3. 记录实时指标
4. 仿真结束后保持服务器运行
5. 访问 http://localhost:6006 查看

**手动停止TensorBoard**:
```bash
lsof -ti :6006 | xargs kill -9
```

### TensorBoard 显示优化

**调整曲线可见性**：
- **Smoothing**: 左下角滑块调到0可看原始数据（默认0.6会平滑）
- **查看直线数据**: 对于Oracle等确定性策略，routing ratio会是稳定的直线（100%），这是正常的
  - 悬停在图表上可以看到精确数值
  - 使用Random scheduler可以看到明显的波动曲线

**对比不同scheduler**：
1. 修改config.json中的scheduler type
2. 运行仿真（会自动清理旧日志）
3. 在TensorBoard中观察指标差异

### 监控指标

**队列指标** (`Queue/`):
- `Replica_X_Length` - 队列长度
- `Replica_X_Utilization` - 利用率

**请求指标** (`Request/`):
- `Type_X_QueueTime` - 排队时间
- `Type_X_ServiceTime` - 服务时间
- `Type_X_TotalTime` - 总时延

**路由指标** (`Routing/`):
- `Type_X_Optimal` - 路由准确率

**系统指标** (`System/`):
- `Time` - 仿真时间
- `CompletedRequests` - 完成请求数
- `RequestsInSystem` - 系统内请求数

**汇总指标** (`Metrics/`):
- `MeanLatency` - 平均时延
- `P50Latency` / `P99Latency` - 时延分位数
- `RoutingAccuracy` - 路由准确率

## Scheduler 类型

- `oracle` - 最优策略（Type A→Replica 0, Type B→Replica 1）
- `random` - 随机路由
- `round_robin` - 轮询路由
- `shortest_queue` - 最短队列优先

## 输出

- **Metrics CSV**: `toymodel/outputs/metrics/experiment_metrics.csv`
- **TensorBoard logs**: `toymodel/outputs/tensorboard/`

## 更多文档

- **接口文档**: `docs/interface.md`
- **演进历史**: `docs/evolution.md`
- **PPO设计**: `docs/ppo_design.md`
