# 调度器性能对比指南

## 概述

本文档提供了多种调度器（PPO、Random、Round Robin、LOR）性能对比的完整指南，包括测试脚本、结果分析和解读说明。

## 对比调度器

1. **PPO** - 强化学习调度器（使用训练的checkpoint）
2. **Random** - 随机调度器
3. **Round Robin** - 轮询调度器
4. **LOR** - 最少请求调度器（Least Outstanding Requests）

## 测试脚本

### 快速对比脚本 (推荐)

适用于快速验证和日常测试：

```bash
# 运行快速对比
bash scripts/quick_scheduler_comparison.sh
```

**特点**：
- 测试时间短（200请求）
- 自动提取核心指标
- 生成简洁的对比报告
- 包含超时保护（120秒）

### 完整对比脚本

适用于详细评测和研究分析：

```bash
# 运行完整对比
bash scripts/scheduler_comparison.sh
```

**特点**：
- 更多测试请求（500请求）
- 详细的结果分析
- 生成Markdown格式报告
- 包含错误诊断信息

## 使用方法

### 前置条件

1. **PPO Checkpoint**: 确保存在训练好的PPO模型
   ```bash
   # 检查checkpoint是否存在
   ls outputs/checkpoints/latest.pt
   ```

2. **执行权限**: 确保脚本有执行权限
   ```bash
   chmod +x scripts/quick_scheduler_comparison.sh
   chmod +x scripts/scheduler_comparison.sh
   ```

### 执行步骤

1. **运行快速对比**：
   ```bash
   cd /path/to/Vidur_arc2
   bash scripts/quick_scheduler_comparison.sh
   ```

2. **查看结果**：
   ```bash
   # 查看控制台输出的对比表格
   # 或查看保存的结果文件
   cat outputs/runs/quick_comparison_*/comparison_summary.txt
   ```

## 结果解读

### 输出格式

```
调度器      状态       平均延迟(s)      吞吐量(req/s)
------------------------------------------------------
PPO        ✅         2.3456         8.5234
Random     ✅         3.2100         6.2100
RoundRobin ✅         2.8900         7.1200
LOR        ✅         2.6700         7.8900
```

### 状态说明

- **✅** - 测试成功，指标有效
- **❌** - 测试失败或超时
- **⚠️** - 测试完成但未找到指标
- **❓** - 未知状态

### 性能指标

1. **平均延迟** (越低越好)
   - 单位：秒
   - 衡量请求处理的响应时间

2. **吞吐量** (越高越好)
   - 单位：请求/秒
   - 衡量系统处理请求的能力

### 性能排名

脚本会自动生成两个排名：

1. **延迟排名**: 从低到高排序（越低越好）
2. **吞吐量排名**: 从高到低排序（越高越好）

## 自定义配置

### 修改测试参数

编辑脚本开头的配置变量：

```bash
# 在 quick_scheduler_comparison.sh 中
NUM_REQUESTS=200    # 请求数量
QPS=2              # 每秒请求数
NUM_REPLICAS=4     # 副本数量
```

### 修改Checkpoint路径

```bash
# 修改PPO模型路径
CHECKPOINT_PATH="/your/path/to/checkpoint.pt"
```

### 添加新调度器

在脚本中添加新的调度器测试：

```bash
# 添加新调度器测试
run_scheduler_test "NewScheduler" "new_scheduler_type" "
    --new_scheduler_param value
"
```

## 故障排查

### 常见问题

1. **PPO Checkpoint不存在**
   ```bash
   # 确保checkpoint文件存在
   ls -la outputs/checkpoints/latest.pt
   ```

2. **权限问题**
   ```bash
   chmod +x scripts/*.sh
   ```

3. **Python模块问题**
   ```bash
   # 确保在正确的环境中
   python -m vidur.main --help
   ```

4. **超时问题**
   - 增加超时时间（默认120秒）
   - 减少请求数量

### 日志检查

每个调度器的详细日志保存在：
```
outputs/runs/quick_comparison_YYYYMMDD_HHMMSS/
├── PPO_test.log
├── Random_test.log
├── RoundRobin_test.log
├── LOR_test.log
└── comparison_summary.txt
```

查看失败日志：
```bash
# 查看特定调度器的错误日志
tail -n 20 outputs/runs/quick_comparison_*/PPO_test.log
```

## 结果分析建议

### 性能评估维度

1. **绝对性能**
   - PPO vs 基线调度器的性能差异
   - 是否达到预期改进目标

2. **稳定性**
   - 多次运行结果的一致性
   - 是否有明显的性能波动

3. **适用场景**
   - 高QPS vs 低QPS场景
   - 不同副本数下的表现

### 基准期望

基于一般经验，期望的性能排序：

1. **延迟**: LOR ≤ PPO < Round Robin < Random
2. **吞吐量**: PPO ≥ LOR > Round Robin > Random

PPO调度器应该在延迟和吞吐量之间找到更好的平衡点。

## 扩展使用

### 批量测试

```bash
# 多次运行取平均值
for i in {1..5}; do
    echo "Run $i:"
    bash scripts/quick_scheduler_comparison.sh
    sleep 10
done
```

### 不同配置测试

```bash
# 测试不同QPS设置
for qps in 1 2 4 8; do
    sed -i "s/QPS=.*/QPS=$qps/" scripts/quick_scheduler_comparison.sh
    bash scripts/quick_scheduler_comparison.sh
done
```

---

**相关文档**:
- `test_commands.md` - 单个调度器测试命令
- `ppo_optimization_2025_09_18.md` - PPO优化技术详情
- `config_changes_summary.md` - 配置参数说明