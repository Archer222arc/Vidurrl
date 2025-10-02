# Toymodel 目录结构

```
toymodel/
├── README.md                    # 模块使用说明
├── README_STRUCTURE.md          # 本文件：目录结构说明
├── __init__.py                  # 模块导出
│
├── src/                         # 核心代码
│   ├── __init__.py
│   ├── entities.py              # 核心实体（Request, Replica）
│   ├── environment.py           # M/M/1仿真环境
│   ├── request_generator.py     # Poisson到达过程生成器
│   ├── config.py                # 配置管理
│   └── tensorboard_monitor.py   # TensorBoard监控
│
├── schedulers/                  # 调度策略
│   ├── __init__.py
│   ├── base.py                  # 基础调度器
│   ├── oracle.py                # 最优策略
│   └── baselines.py             # 基线策略
│
├── scripts/                     # 运行脚本
│   ├── run_simulation.py        # 仿真主程序
│   ├── compare_schedulers.py    # Scheduler对比工具
│   ├── run_toymodel.sh          # Bash wrapper
│   └── compare_schedulers.sh    # Bash wrapper
│
├── configs/                     # 配置文件
│   ├── config.json              # 默认配置
│   ├── high_load.json           # 高负载场景
│   └── balanced_load.json       # 均衡负载场景
│
├── outputs/                     # 输出结果
│   ├── metrics/                 # CSV导出
│   └── tensorboard/             # TensorBoard日志
│
├── tests/                       # 单元测试
│   ├── __init__.py
│   ├── test_entities.py
│   ├── test_environment.py
│   └── test_schedulers.py
│
├── demo/                        # 示例代码
│   └── demo_environment.py      # 环境使用示例
│
└── docs/                        # 模块文档
    ├── interface.md             # 接口文档
    ├── evolution.md             # 演进历史
    └── ppo_design.md            # PPO设计文档

```

## 目录说明

### src/ - 核心代码
所有核心实现代码，便于复用和测试。

### schedulers/ - 调度策略
所有路由策略实现，支持自定义扩展。

### scripts/ - 运行脚本
可执行脚本，调用 `src/` 中的核心代码。

### configs/ - 配置文件
JSON配置文件，参数化控制仿真行为。

### outputs/ - 输出结果
- `metrics/`: CSV格式的实验指标
- `tensorboard/`: TensorBoard日志文件

### tests/ - 单元测试
pytest测试脚本，覆盖核心功能。

### demo/ - 示例代码
使用示例和快速入门代码。

### docs/ - 模块文档
模块接口文档、演进历史、设计文档。

## 文件迁移映射

**当前结构 → 目标结构**：

```
toymodel/entities.py              → toymodel/src/entities.py
toymodel/environment.py           → toymodel/src/environment.py
toymodel/request_generator.py     → toymodel/src/request_generator.py
toymodel/config.py                → toymodel/src/config.py
toymodel/tensorboard_monitor.py   → toymodel/src/tensorboard_monitor.py

toymodel/run_simulation.py        → toymodel/scripts/run_simulation.py
toymodel/compare_schedulers.py    → toymodel/scripts/compare_schedulers.py
toymodel/scripts/*.sh             → toymodel/scripts/*.sh

configs/toymodel/config.json      → toymodel/configs/config.json

outputs/toymodel/metrics/         → toymodel/outputs/metrics/
outputs/toymodel/tensorboard/     → toymodel/outputs/tensorboard/

tests/toymodel/*                  → toymodel/tests/*

demo/demo_environment.py          → toymodel/demo/demo_environment.py

docs/modules/toymodel.md          → toymodel/docs/interface.md
docs/modules/evolution/toymodel_evolution.md → toymodel/docs/evolution.md
docs/toymodel_ppo_routing_design.md → toymodel/docs/ppo_design.md
```

## 维护原则

1. **src/** - 核心代码，模块化设计，便于复用
2. **scripts/** - 运行脚本，仅做调用和流程控制
3. **configs/** - 参数化配置，便于实验复现
4. **outputs/** - 实验结果，git忽略但目录保留
5. **tests/** - 单元测试，覆盖核心功能
6. **docs/** - 文档集中管理，便于查阅

---

**维护者**: Claude
**创建时间**: 2025-10-01
**最后更新**: 2025-10-01
