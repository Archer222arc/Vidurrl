# Claude Code 项目规范 - Vidur智能调度器项目

## 📁 文件夹管理规范 - 最高优先级

### 核心原则
- **简洁性优先**：避免过度嵌套和冗余目录
- **功能导向**：按功能模块组织，不按技术栈分类
- **版本控制友好**：结构稳定，方便git跟踪和协作

### 标准目录结构（推荐重构目标）
```
$PROJECT_NAME/                 # 项目根目录
├── demo/                      # 示例代码（获取数据、代码规范、工具代码等）
├── src/                       # 核心代码（模块化设计，参数化输入，便于复用）
├── configs/                   # 配置文件（支持 json/yaml/toml，供 src/scripts 使用）
├── data/                      # 数据存储（需处理或存取的数据）
├── experiments/               # 实验配置（历史复现实验的参数）
├── outputs/                   # 输出结果（实验输出存放处）
├── logs/                      # 日志文件
├── tests/                     # 测试脚本（pytest 单元测试与集成测试）
├── scripts/                   # 脚本代码（运行脚本、测试脚本等）
├── tmp/                       # 临时文件（临时测试脚本，用完即删）
├── docs/                      # 自定义文档（说明文档等）
└── notebooks/                 # Jupyter Notebook 格式的可视化报告
```
### 文件管理策略
1. **遵循标准结构**：严格按照标准目录组织新文件
2. **渐进式迁移**：逐步将现有文件迁移到标准位置
3. **配置参数化**：所有硬编码路径改为配置驱动
4. **实验可复现**：每个实验在experiments/中保存完整配置

### 目录功能说明
- `src/`：核心代码，模块化设计，便于复用和测试
- `scripts/`：运行脚本，包括训练、测试、数据处理等
- `configs/`：所有配置文件，支持json/yaml格式
- `data/`：原始数据、处理后数据、训练数据集
- `experiments/`：历史实验的配置参数，便于复现
- `outputs/`：实验输出结果，包括模型、报告、对比结果
- `logs/`：日志文件，按模块和时间组织
- `tests/`：单元测试和集成测试脚本
- `tmp/`：临时文件存储，包括临时测试脚本（用完即删）
- `demo/`：示例代码和使用说明
- `notebooks/`：Jupyter分析报告和可视化

## 🚨 核心编程规范 - 严格执行

### 八荣八耻编程基本原则

1. **以暗猜接口为耻，以认真查阅为荣** - 禁止臆测API行为，必须查阅文档和代码确认
2. **以模糊执行为耻，以寻求确认为荣** - 不确定的实现必须先向用户确认，避免模糊操作
3. **以默认忽略为耻，以主动报告为荣** - 遇到异常、警告、错误必须主动报告，不得静默忽略
4. **以隐式假设为耻，以显式验证为荣** - 所有假设必须通过代码验证，禁止隐式依赖
5. **以随意修改为耻，以谨慎调试为荣** - 修改前必须理解原理，禁止试错式编程
6. **以表面应付为耻，以深入理解为荣** - 解决问题必须找到根本原因，禁止表面修补
7. **以复制粘贴为耻，以原创思考为荣** - 理解每行代码含义，禁止盲目复制
8. **以孤立开发为耻，以协同沟通为荣** - 主动汇报进度和问题，寻求指导和反馈
9. **以测试验证为耻，以代码阅读为荣** - 验证接口/函数/配置功能时必须阅读代码确认，禁止创建测试脚本验证

### 🧪 测试脚本管理规范 - 严格执行

**功能验证方法**：
- ✅ **代码阅读优先** - 验证接口/函数/配置功能时，必须通过阅读源代码确认，不得创建测试脚本
- ❌ **禁止测试脚本验证** - 严禁为验证功能正确性而创建临时测试代码
- ✅ **直接调用验证** - 如需运行验证，直接在命令行或现有入口点调用

**临时测试脚本规范**（仅限必要情况）：
- 📁 **统一存储位置** - 所有临时测试脚本必须存储在 `tmp/` 文件夹中
- 🗑️ **及时清理** - 测试完成后立即删除临时脚本，不得留存
- 📝 **明确标识** - 临时脚本必须以 `test_` 或 `temp_` 前缀命名
- ⏰ **生命周期管理** - 临时脚本仅在当前会话有效，不得提交到版本控制

**示例对比**：
```bash
# ❌ 错误做法：创建测试脚本验证
echo "# 测试reward函数" > test_reward.py
python test_reward.py

# ✅ 正确做法：直接阅读源码
# 查看 src/core/reward.py 确认函数接口和实现逻辑
# 直接运行：python -c "from src.core.reward import calculate_reward; print(calculate_reward.__doc__)"

# 🆘 必要时的临时测试（立即删除）
mkdir -p tmp/
echo "import sys; print(sys.path)" > tmp/test_imports.py
python tmp/test_imports.py
rm tmp/test_imports.py
```

### 🔥 文件命名规范 - 严格禁止

**禁用前缀后缀列表**：
- ❌ `enhanced_*` / `*_enhanced` - 禁止enhanced前缀后缀
- ❌ `integrated_*` / `*_integrated` - 禁止integrated前缀后缀
- ❌ `cleaned_*` / `*_cleaned` / `*_clean` - 禁止clean相关命名
- ❌ `improved_*` / `*_improved` - 禁止improved前缀后缀
- ❌ `optimized_*` / `*_optimized` - 禁止optimized前缀后缀（项目目录名除外）
- ❌ `advanced_*` / `*_advanced` - 禁止advanced前缀后缀
- ❌ `*_v2` / `*_new` / `*_old` / `*_temp` - 禁止版本和临时标识符

**正确命名原则**：
- ✅ **功能导向命名** - 直接描述文件功能：`reward_system.py`、`sac_trainer.py`
- ✅ **模块化命名** - 按模块组织：`scheduler/`、`metrics/`、`config/`
- ✅ **简洁明确** - 避免冗余形容词，直接表达核心功能
- ✅ **统一风格** - 使用下划线分隔，全小写字母

**命名示例对比**：
```bash
# ❌ 错误命名
enhanced_sac_training_metrics.py   →  # ✅ sac_training_metrics.py
train_sac_integrated.py            →  # ✅ train_sac.py
restart_tensorboard_clean.sh       →  # ✅ restart_tensorboard.sh
enhanced_reward.py                 →  # ✅ reward_system.py
train_gru_sac_old.sh              →  # ✅ 直接删除，保留train_gru_sac.sh
```

**违规处理**：
- 发现违规文件名立即重命名或删除
- 代码审查时强制执行此规范
- 新文件创建前必须检查命名合规性

### 🛡️ 错误处理强制规范

```python
# ❌ 严格禁止的fallback模式
try:
    result = complex_operation()
except Exception:
    result = fallback_operation()  # 禁止！

# ❌ 严格禁止的属性检查fallback
if hasattr(obj, 'attribute'):
    return obj.attribute
else:
    return default_value  # 禁止！

# ✅ 正确的错误处理方式
result = complex_operation()  # 让错误自然抛出
required_attribute = obj.attribute  # 直接访问，缺失时报错
```

**核心要求**：
- 🔥 **禁止使用try except** - 碰见错误直接显示traceback并退出终止运行程序
- 🔥 **禁止采用fallback方案** - 如缺少属性直接报错返回，不允许降级处理
- ✅ **让错误自然抛出** - 便于从本质上解决问题，而非掩盖问题


### 🔧 脚本组织和模块化规范

**脚本复杂度控制**：
- ✅ **简单脚本**: 直接在scripts/中实现，最多50行
- ✅ **复杂逻辑**: 必须分离到src/模块中，脚本仅做调用
- ❌ **禁止内嵌**: 严禁在脚本中写大段Python代码或函数
- ❌ **禁止重复**: 相同逻辑不得在多个脚本中重复实现

**模块化分离原则**：
```bash
# ❌ 错误做法：在脚本中内嵌复杂逻辑
train_model.sh:
    python -c "
    import complex_logic
    # 50行复杂代码...
    "

# ✅ 正确做法：分离到模块
src/training/trainer.py:     # 复杂逻辑在独立模块
    class ModelTrainer: ...

scripts/train_model.sh:     # 脚本仅做调用
    python -m src.training.trainer --config $1
```

**脚本职责边界**：
- **scripts/**: 参数传递、流程控制、状态检查
- **src/**: 核心算法、数据处理、复杂逻辑
- **configs/**: 参数配置、超参数设定

**集成vs分离决策标准**：
- **集成条件**: 功能高度相关且参数配置一致
- **分离条件**: 独立功能模块或可复用组件
- **重构时机**: 脚本超过50行或出现重复逻辑时

### 监控体系

**三层监控架构**：
1. **实时监控**: TensorBoard (`http://localhost:6006`)
   - 训练过程实时指标监控
   - Loss/Reward连续曲线
   - 超参数记录和对比

2. **数据导出**: CSV结构化数据
   - 实验配置和元信息记录
   - 各训练阶段详细指标
   - FQE/OPE评估结果

## 🔌 新功能接入标准化流程 - 严格执行

### 核心原则
遵循**配置-架构-main接口-代码-验证五步法**，确保新功能能正确集成到训练系统中，避免静默失效。

### 强制检查清单

#### 第一步：配置层面验证 ✅
1. **JSON配置结构检查**
   ```json
   # ✅ 正确：扁平结构，直接映射config类字段
   "new_feature_enable": true,
   "new_feature_param1": 0.01,
   "new_feature_param2": 100

   # ❌ 错误：嵌套结构，映射可能失败
   "new_feature": {
     "enable": true,
     "param1": 0.01,
     "param2": 100
   }
   ```

2. **三层配置结构要求 - 严格遵循！**
   ```json
   {
     // 第一层：基本训练参数（训练脚本直接读取）
     "training": {
       "num_replicas": 4,
       "ppo_requests": 5000,
       "qps": 3,
       "max_steps": 1000
     },

     // 第二层：算法参数（PPO算法配置）
     "ppo_config": {
       "lr": 0.00025,
       "gamma": 0.99,
       "clip_ratio": 0.2,
       "epochs": 4,
       "rollout_len": 256,
       // ... 其他PPO算法参数
     },

     // 第三层：Scheduler和新功能参数
     "cluster_config": {
       "num_replicas": 4,  // 可以重复，确保兼容性
       "global_scheduler_config": {
         "scheduler_type": "ppo_global_scheduler_modular",
         "new_feature_enable": true,
         "new_feature_param1": 0.01
       }
     },

     // 监控配置（如果需要）
     "monitoring": {
       "metrics_subsamples": 1000
     }
   }
   ```

   **关键配置层级说明**：
   - `training`: 训练脚本通过 `config["training"]["field"]` 读取
   - `ppo_config`: 训练脚本通过 `config["ppo_config"]["field"]` 读取PPO算法参数
   - `cluster_config.global_scheduler_config`: vidur.main通过配置链路传递到Scheduler类

3. **Config类字段完整性**
   - 必须在`vidur/config/config.py`中对应的Config类添加所有字段
   - 字段名必须与JSON配置**完全一致**
   - 必须设置合理的default值和metadata

4. **训练脚本兼容性检查**
   - 检查训练脚本是否使用传统字段路径：`config["training"]["field_name"]`
   - 确认是否需要同时维护传统和标准两套配置
   - 验证训练脚本能正确读取所有必需字段

#### 第二步：配置架构验证 🏗️
1. **Scheduler类型配置**
   - **关键**：必须在JSON中指定正确的scheduler类型
   ```json
   {
     "cluster_config": {
       "global_scheduler_config": {
         "scheduler_type": "ppo_global_scheduler_modular",
         "new_feature_enable": true,
         "new_feature_param1": 0.01
       }
     }
   }
   ```

2. **配置类继承验证**
   - 确认新功能字段在正确的Config子类中（如PPOGlobalSchedulerModularConfig）
   - 验证默认scheduler类型是否包含新功能字段

#### 第二步B：training_config.py配置转换 🔄
**CRITICAL STEP - 经常被遗漏的环节！**

1. **配置转换脚本更新**
   ```python
   # src/core/utils/infrastructure/config/training_config.py
   # 必须添加新功能参数的处理逻辑

   def build_ppo_args(config: Dict, output_dir: str) -> List[str]:
       # ... 现有代码 ...

       # 新功能参数处理（必须添加！）
       if "cluster_config" in config and "global_scheduler_config" in config["cluster_config"]:
           scheduler_cfg = config["cluster_config"]["global_scheduler_config"]

           if scheduler_cfg.get("new_feature_enable", False):
               args.extend([
                   f"{ppo_prefix}new_feature_enable",
                   f"{ppo_prefix}new_feature_param1", str(scheduler_cfg.get("new_feature_param1", 0.01)),
                   f"{ppo_prefix}new_feature_param2", str(scheduler_cfg.get("new_feature_param2", 100))
               ])
   ```

2. **验证配置转换正确性**
   ```bash
   # 测试配置转换是否包含新功能参数
   python src/core/utils/infrastructure/config/training_config.py configs/your_config.json /tmp/test
   # 输出应包含：--p_p_o_global_scheduler_modular_config_new_feature_enable
   ```

3. **training_config.py完整传递链**
   ```
   JSON配置 → training_config.py → 命令行参数 → vidur.main → SimulationConfig → Scheduler
   ```

   **如果这一步遗漏，新功能参数永远不会传递给Scheduler，功能静默失效！**

#### 第三步：vidur.main接口更新 🔧
**重要提醒**：还要在vidur.main中也添加对应接口才行！

1. **Main入口配置传递验证**
   ```python
   # vidur/main.py中的配置加载链路
   config: SimulationConfig = SimulationConfig.create_from_cli_args()
   # 必须确保新功能字段能正确传递到这里
   ```

2. **配置扁平化处理**
   - `create_flat_dataclass()` 必须能正确处理新功能字段
   - `reconstruct_original_dataclass()` 必须保持字段完整性
   - 可能需要更新配置加载逻辑以支持新字段

3. **完整传递链路检查**
   ```
   JSON配置 -> SimulationConfig.create_from_cli_args() ->
   create_flat_dataclass() -> reconstruct_original_dataclass() ->
   ClusterConfig -> GlobalSchedulerConfig -> Scheduler实例
   ```

#### 第四步：代码集成验证 🔧
1. **Scheduler初始化检查**
   ```python
   # 在对应的Scheduler类中
   self._new_feature_enable = bool(gcfg.new_feature_enable)

   if self._new_feature_enable:
       self._new_feature.initialize(...)
   ```

2. **统计字段集成**
   ```python
   # 在PPO trainer的stats方法中
   if self.new_feature is not None:
       stats.update({"new_feature_value": self.new_feature.get_current_value()})
   else:
       stats.update({"new_feature_value": 0.0})  # fallback值
   ```

#### 第五步：运行验证 🔍
1. **配置加载完整性验证**
   ```bash
   # 验证配置加载链路
   python -c "
   from vidur.config import SimulationConfig
   config = SimulationConfig.create_from_cli_args()
   scheduler_config = config.cluster_config.global_scheduler_config
   print('Scheduler type:', type(scheduler_config).__name__)
   print('Has new feature:', hasattr(scheduler_config, 'new_feature_enable'))
   "
   ```

2. **训练验证**
   - 检查日志中scheduler类型和新功能初始化
   - CSV字段包含新功能数据且非空
   - 确认训练行为确实发生变化

### 常见错误模式及解决方案 ⚠️

#### 错误1：配置层级结构错误
**症状**：出现 `Error: 'training'`、`Error: 'ppo_config'` 或类似字段缺失错误
**原因**：违反了三层配置结构要求，将参数放在错误的层级中
**解决**：严格按照三层结构组织配置
```json
{
  // 第一层：基本训练参数
  "training": {
    "num_replicas": 4,
    "ppo_requests": 5000,
    "qps": 3
  },
  // 第二层：PPO算法参数（不要放在global_scheduler_config中！）
  "ppo_config": {
    "lr": 0.00025,
    "gamma": 0.99,
    "clip_ratio": 0.2
  },
  // 第三层：新功能和scheduler配置
  "cluster_config": {
    "global_scheduler_config": {
      "scheduler_type": "ppo_global_scheduler_modular",
      "new_feature_enable": true  // 新功能参数放这里
    }
  }
}
```

#### 错误2：Scheduler类型不匹配
**症状**：新功能字段存在但从不被使用
**原因**：JSON未指定scheduler_type，使用了默认的RoundRobinGlobalSchedulerConfig
**解决**：必须在JSON中明确指定scheduler类型

#### 错误3：vidur.main配置传递失败
**症状**：Config类有字段，JSON有配置，但传递到Scheduler时丢失
**原因**：main入口的配置加载过程中字段被过滤或转换失败
**解决**：检查并更新vidur.main中的配置处理逻辑

#### 错误4：扁平化配置映射错误
**症状**：嵌套JSON结构无法正确映射到扁平Config字段
**原因**：create_flat_dataclass处理嵌套结构时失败
**解决**：使用扁平JSON结构，避免嵌套配置

#### 错误5：配置字段重复或冲突
**症状**：同一参数在不同配置结构中有不同值
**原因**：为了兼容性同时维护传统和标准配置，但值不一致
**解决**：确保重复字段保持相同值，或明确哪个优先

#### 错误6：training_config.py遗漏新功能参数 🚨
**症状**：JSON配置正确，Config类有字段，但训练时新功能静默失效
**原因**：`src/core/utils/infrastructure/config/training_config.py`中未添加新功能参数处理
**解决**：在training_config.py的build_ppo_args函数中添加参数转换逻辑
**验证**：运行 `python src/core/utils/infrastructure/config/training_config.py configs/your_config.json /tmp/test` 确保输出包含新功能参数

### 必须检查的关键点 🎯

1. **JSON配置结构正确性**
   - 扁平字段结构
   - 正确的scheduler_type指定

2. **Config类字段完整性**
   - 字段名与JSON完全匹配
   - 合理的default值

3. **training_config.py转换完整性** 🚨
   - build_ppo_args函数包含新功能参数处理
   - 验证命令行参数生成正确：`python src/core/utils/infrastructure/config/training_config.py configs/your_config.json /tmp/test | grep new_feature`

4. **vidur.main接口兼容性**
   - 配置加载链路完整
   - 字段传递无丢失

5. **Scheduler集成正确性**
   - 参数读取正确
   - 初始化条件明确

6. **运行时验证**
   - 日志确认初始化
   - CSV数据非空
   - 训练行为改变

### 调试命令序列 🛠️

```bash
# 1. JSON语法和三层配置结构验证
python -c "
import json
config = json.load(open('configs/config.json'))
print('=== 第一层：基本训练参数 ===')
print('training.num_replicas:', config.get('training', {}).get('num_replicas', 'NOT_FOUND'))
print('training.ppo_requests:', config.get('training', {}).get('ppo_requests', 'NOT_FOUND'))
print('training.qps:', config.get('training', {}).get('qps', 'NOT_FOUND'))
print()
print('=== 第二层：PPO算法参数 ===')
print('ppo_config.lr:', config.get('ppo_config', {}).get('lr', 'NOT_FOUND'))
print('ppo_config.gamma:', config.get('ppo_config', {}).get('gamma', 'NOT_FOUND'))
print('ppo_config.clip_ratio:', config.get('ppo_config', {}).get('clip_ratio', 'NOT_FOUND'))
print()
print('=== 第三层：Scheduler和新功能 ===')
print('scheduler_type:', config.get('cluster_config', {}).get('global_scheduler_config', {}).get('scheduler_type', 'NOT_FOUND'))
print('new_feature_enable:', config.get('cluster_config', {}).get('global_scheduler_config', {}).get('new_feature_enable', 'NOT_FOUND'))
"

# 2. Config类字段验证
python -c "from vidur.config.config import PPOGlobalSchedulerModularConfig; print('Has field:', hasattr(PPOGlobalSchedulerModularConfig(), 'new_feature_enable'))"

# 3. 训练脚本配置读取验证
python -c "
import json
from src.core.utils.infrastructure.config.training_config import get_training_args
config = json.load(open('configs/config.json'))
try:
    args = get_training_args(config, 'test_output')
    print('训练脚本配置读取成功')
    print('参数数量:', len(args))
except Exception as e:
    print('训练脚本配置读取失败:', e)
"

# 4. Main入口配置加载测试
python -c "from vidur.config import SimulationConfig; config=SimulationConfig.create_from_cli_args(); print('Config loaded successfully')"

# 5. 完整训练测试
timeout 30s python vidur/simulator.py --config configs/config.json --num_requests 10

# 6. 结果验证
grep -i "new_feature" training.log
head -1 metrics.csv | grep new_feature
```

**关键提醒**：
1. **三层配置结构严格遵循**：`training` → `ppo_config` → `cluster_config.global_scheduler_config`，参数放错层级会导致 `Error: 'field_name'` 错误
2. **vidur.main接口兼容性**：还要在vidur.main中也添加对应接口才行！
3. **完整验证链路**：必须验证从JSON配置到Scheduler实例的完整传递链路
4. **训练行为确认**：确保新功能确实改变训练行为，而不仅仅是"运行无错误"

**配置层级记忆口诀**：
- 第一层 `training`: 训练脚本基础参数（replicas, requests, qps）
- 第二层 `ppo_config`: PPO算法参数（lr, gamma, clip等）
- 第三层 `cluster_config`: 新功能和scheduler配置（新功能参数放这里）
