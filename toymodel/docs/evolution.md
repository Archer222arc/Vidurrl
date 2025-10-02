# Toymodel 模块演进历史

## 📅 版本时间线

### v1.0.0 - 2025-10-01
**状态**: ✅ 当前版本
**标签**: 初始版本

---

## 🎯 v1.0.0 - 初始实现 (2025-10-01)

### 新增功能

#### 1. 核心仿真环境
- **M/M/1排队系统**: 事件驱动仿真
  - Poisson到达过程（指数到达间隔）
  - 指数服务时间
  - FIFO队列
  - 2个replica，每个处理不同类型请求的速率不同

- **关键类**:
  - `Request`: 请求实体（跟踪arrival_time, queue_time, service_time等）
  - `Replica`: Replica实体（管理队列和服务状态）
  - `QueueEnvironment`: 仿真环境（事件循环、路由控制）

#### 2. 调度策略
- `OracleScheduler`: 最优策略（Type A→Replica 0, Type B→Replica 1）
- `RandomScheduler`: 随机路由
- `RoundRobinScheduler`: 轮询路由
- `ShortestQueueScheduler`: 最短队列优先

#### 3. 配置管理
- JSON配置文件：`configs/toymodel/config.json`
- Dataclass结构化配置加载
- 参数验证（num_replicas=2, 正值rates等）
- 支持的配置项：
  - `experiment`: 实验元信息（name, seed, description）
  - `environment`: 仿真参数（replicas, rates, time）
  - `scheduler`: 调度器类型和选项
  - `metrics`: CSV导出配置
  - `tensorboard`: 监控配置

#### 4. TensorBoard监控
- **自动化流程**:
  - 启动时自动kill旧进程
  - 后台运行（detached process）
  - 自动清理旧日志（可配置）
  - 仿真结束保持运行

- **监控指标**（细粒度）:
  - **Routing**: 按type记录到各replica的分配比例
    - `Type_{X}_to_Replica_{Y}_Ratio`
    - `Type_{X}_Accuracy`
  - **Latency**: 按replica追踪时延
    - `Replica_{X}_Mean`, `Replica_{X}_P99`
  - **Queue**: 队列长度和利用率
    - `Replica_{X}_Length`, `Replica_{X}_Utilization`
  - **System**: 系统整体状态
    - `Time`, `CompletedRequests`, `RequestsInSystem`

- **记录频率**:
  - 汇总指标: 每20个请求
  - 队列指标: 每5个请求
  - TensorBoard刷新: 5秒

#### 5. 运行工具
- `run_simulation.py`: 仿真主程序（支持命令行参数）
- `compare_schedulers.py`: Scheduler性能对比
- Bash wrapper scripts:
  - `run_toymodel.sh`
  - `compare_schedulers.sh`

### 设计决策

#### 为什么选择M/M/1模型？
- **简化验证**: 相比完整LLM serving系统，M/M/1可解析、可预测
- **清晰的最优策略**: Oracle策略作为ground truth
- **快速迭代**: 仿真时间短，便于调试PPO

#### 为什么固定2个replica？
- **最小化复杂度**: 2个replica足以验证路由策略学习
- **对称设计**: 两个replica处理能力互补（Type A快慢相反）
- **扩展性**: 架构支持扩展到N个replica（当前配置限制为2）

#### TensorBoard集成策略
- **后台持续运行**: 避免每次仿真重启服务器
- **自动清理旧日志**: 防止多run叠加导致图表混乱
- **细粒度指标**: 按type和replica分解，便于诊断问题

### 性能表现

**测试配置**: λ_A=λ_B=6.0, max_time=500, ~6000个请求

| Scheduler | Mean Latency | P99 Latency | Routing Accuracy |
|-----------|-------------|-------------|------------------|
| Oracle | 0.252 | 1.240 | 100.0% |
| Shortest Queue | 0.913 | 3.245 | 50.9% |
| Round-Robin | 1.050 | 3.132 | 50.3% |
| Random | 1.340 | 3.639 | 48.4% |

**关键观察**:
- Oracle达到理论最优性能
- 非Oracle策略与Oracle有~3-5倍latency差距
- Shortest Queue优于Round-Robin和Random

### 文件结构变化

#### 新增文件
```
toymodel/
├── __init__.py
├── config.py
├── entities.py
├── environment.py
├── request_generator.py
├── tensorboard_monitor.py
├── run_simulation.py
├── compare_schedulers.py
├── README.md
├── schedulers/
│   ├── __init__.py
│   ├── base.py
│   ├── oracle.py
│   └── baselines.py
└── scripts/
    ├── run_toymodel.sh
    └── compare_schedulers.sh

configs/toymodel/
└── config.json

docs/
├── modules/
│   ├── toymodel.md
│   └── evolution/
│       └── toymodel_evolution.md
└── toymodel_ppo_routing_design.md

tests/toymodel/
├── test_entities.py
├── test_environment.py
└── test_schedulers.py

demo/
└── demo_environment.py
```

### 未来规划

#### Phase 2 - PPO集成（待实现）
- [ ] State builder（6维状态向量）
- [ ] Reward calculator（基于latency）
- [ ] PPOScheduler实现
- [ ] 训练循环
- [ ] Warm-start from demonstrations

#### 可能的增强
- [ ] 支持可变到达率（动态负载）
- [ ] 支持N个replica（当前固定为2）
- [ ] 更多baseline策略（JSQ, Power-of-Two等）
- [ ] 离线分析工具（对比报告生成）

### 技术债务

无

### 已知限制

1. **Replica数量固定**: 当前验证逻辑hardcode为2
2. **同步仿真**: 单线程事件循环（不影响正确性）
3. **内存存储**: 所有完成请求保存在内存（长时间仿真可能OOM）

---

**维护说明**:
- 每个重大功能更新都应在此文档记录
- 设计决策应包含"为什么"的解释
- 性能数据应定期更新
- 技术债务应及时标记和跟踪

**文档版本**: v1.0
**最后更新**: 2025-10-01
**维护者**: Claude
