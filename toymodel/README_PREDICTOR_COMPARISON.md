# Latency Predictor Performance Comparison

本文档说明如何训练和对比两种延迟预测器（Simple vs System-Aware）的性能。

## 快速开始

运行完整的训练和对比流程：

```bash
./toymodel/scripts/train_and_compare_predictors.sh
```

该脚本会自动完成：
1. 使用Simple Predictor训练PPO模型（1000 episodes）
2. 使用System-Aware Predictor训练PPO模型（1000 episodes）
3. 对比两个模型的性能并生成报告

## 预测器差异

### Simple Predictor

**计算公式**：
```
predicted_latency = Σ(queue_processing_times) + self_processing_time
```

**特点**：
- 只考虑当前请求的完成时间
- 不考虑对系统中其他请求的影响
- 优化目标：单个请求延迟最小化

**实现位置**：`toymodel/src/predictors/simple_predictor.py`

### System-Aware Predictor

**计算公式**：
```
combined_latency = self_latency + impact_on_others * impact_weight

其中：
- self_latency = Σ(queue_processing_times) + self_processing_time
- impact_on_others = self_processing_time × queue_length
```

**特点**：
- 考虑当前请求的完成时间
- 考虑对队列中其他请求造成的延迟影响
- 优化目标：系统整体延迟最小化

**实现位置**：`toymodel/src/predictors/system_aware_predictor.py`

**关键修复**（2025-10-03）：
```python
# 修复前（错误）
impact_on_others = avg_processing_time  # 只计算单个请求延迟

# 修复后（正确）
impact_on_others = avg_processing_time * queue_length  # 计算所有队列请求的总延迟
```

## 配置文件

### Simple Predictor配置

文件：`toymodel/configs/ppo_config_simple.json`

关键参数：
```json
{
  "ppo": {
    "predictor_type": "simple",
    "use_prediction": true,
    "prediction_weight": 1,
    "impact_weight": 0
  },
  "output": {
    "dir": "toymodel/outputs/simple_predictor"
  }
}
```

### System-Aware Predictor配置

文件：`toymodel/configs/ppo_config_system_aware.json`

关键参数：
```json
{
  "ppo": {
    "predictor_type": "system_aware",
    "use_prediction": true,
    "prediction_weight": 1,
    "impact_weight": 1
  },
  "output": {
    "dir": "toymodel/outputs/system_aware_predictor"
  }
}
```

## 手动运行

### 1. 训练Simple Predictor

```bash
python toymodel/scripts/train_ppo.py \
  --config toymodel/configs/ppo_config_simple.json \
  --device cpu
```

### 2. 训练System-Aware Predictor

```bash
python toymodel/scripts/train_ppo.py \
  --config toymodel/configs/ppo_config_system_aware.json \
  --device cpu
```

### 3. 对比性能

```bash
python toymodel/scripts/compare_predictors.py \
  --simple-config toymodel/configs/ppo_config_simple.json \
  --simple-model toymodel/outputs/simple_predictor/models/ppo_model_latest.pt \
  --system-aware-config toymodel/configs/ppo_config_system_aware.json \
  --system-aware-model toymodel/outputs/system_aware_predictor/models/ppo_model_latest.pt \
  --num-episodes 10 \
  --output-dir toymodel/outputs/comparison
```

## 输出文件结构

```
toymodel/outputs/
├── simple_predictor/
│   ├── models/
│   │   ├── ppo_model_latest.pt          # 最新模型
│   │   ├── ppo_model_episode_50.pt      # 检查点
│   │   └── ...
│   ├── tensorboard/                      # TensorBoard日志
│   ├── eval/                             # 评估结果
│   └── training.log                      # 训练日志
├── system_aware_predictor/
│   ├── models/
│   │   ├── ppo_model_latest.pt
│   │   └── ...
│   ├── tensorboard/
│   ├── eval/
│   └── training.log
└── comparison/
    ├── comparison_results.json           # 性能对比数据
    ├── predictor_comparison.png          # 可视化对比图
    └── comparison.log                    # 对比日志
```

## 性能指标

对比脚本会计算并对比以下指标：

1. **延迟指标**：
   - Mean Latency（平均延迟）
   - P50 Latency（中位数延迟）
   - P95 Latency（95分位延迟）
   - P99 Latency（99分位延迟）

2. **路由准确性**：
   - Routing Accuracy（将请求路由到最优replica的比例）

3. **改进幅度**：
   - System-Aware相对于Simple的性能提升百分比

## TensorBoard监控

### Simple Predictor

```bash
tensorboard --logdir toymodel/outputs/simple_predictor/tensorboard --port 6006
```

访问：http://localhost:6006

### System-Aware Predictor

```bash
tensorboard --logdir toymodel/outputs/system_aware_predictor/tensorboard --port 6007
```

访问：http://localhost:6007

## 预期结果

基于理论分析，System-Aware Predictor应该表现出：

1. **更低的系统平均延迟**：通过避免将请求调度到长队列
2. **更低的P99延迟**：减少极端情况下的延迟峰值
3. **更高的系统吞吐量**：更均衡的负载分配

## 故障排查

### PyTorch版本问题

如果遇到`weights_only`相关错误：
```
WeightsUnpickler error: Unsupported global...
```

确保compare_predictors.py中使用：
```python
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
```

### 配置对象缺少属性

如果遇到`'ToyModelConfig' object has no attribute 'output'`错误：

确保config.py中包含OutputConfig和PPOConfig类，并且from_json方法正确解析这些字段。

### 模型未保存到正确目录

确保train_ppo.py使用配置文件中的输出目录：
```python
self.output_dir = config.output.dir
```

## 版本历史

- **2025-10-03**:
  - 修复system_aware_predictor的impact计算（乘以queue_length）
  - 添加OutputConfig和PPOConfig到配置系统
  - 创建自动化训练和对比脚本
  - 修复PyTorch 2.6 weights_only兼容性问题
