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

## 📋 模块文档管理规范 - 强制执行

### 核心原则
- **变更可追溯**：每个模块必须维护完整的演进历史
- **废弃功能保留**：所有deprecated功能统一管理，不得直接删除
- **版本记录标准化**：使用统一格式记录变更历史

### 强制文档结构

#### 1. 模块接口文档 (docs/modules/) - 长期稳定
每个重要模块必须维护完整的接口和功能说明文档：
```
docs/modules/
├── memory_management.md               # 内存管理模块完整接口文档
├── reward_system.md                   # 奖励系统模块完整接口文档
├── ppo_training_system.md            # PPO训练系统模块完整接口文档
├── metrics_collection.md              # 指标收集模块完整接口文档
└── evolution/                         # 演进历史子目录
    ├── memory_management_evolution.md      # 内存管理模块演进史
    ├── reward_system_evolution.md          # 奖励系统演进史
    ├── ppo_training_system_evolution.md    # PPO训练系统演进史
    └── metrics_collection_evolution.md     # 指标收集演进史
```

#### 2. 废弃功能文档 (docs/deprecated/)
与模块文档一一对应，分模块管理废弃功能：
```
docs/deprecated/
├── DEPRECATED_COMPONENTS.md                    # 废弃组件总览索引
├── deprecated_memory_management.md             # 内存管理模块废弃功能
├── deprecated_reward_system.md                 # 奖励系统模块废弃功能
├── deprecated_ppo_training.md                  # PPO训练系统废弃功能
└── deprecated_metrics_collection.md            # 指标收集模块废弃功能
```

#### 3. 超参数调优文档 (docs/) - 实验记录
**位置**: `docs/hyperparameter_tuning_log.md`

**职责**: 记录所有超参数实验、失败案例、经验总结
- 每次重要实验的完整配置
- 训练结果和关键指标
- 失败原因分析和教训
- 成功配置和最佳实践
- 监控指标理想范围
- 紧急修复方案

**与演进文档的区别**:
- **hyperparameter_tuning_log.md**: 实验导向，记录尝试过程和经验
- **evolution/*.md**: 版本导向，记录最终采用的变更

**更新时机**:
- 每次重要训练实验后
- 发现新的超参数问题时
- 积累新的调参经验时
```

#### 4. 文档职责分工

**模块接口文档** (docs/modules/*.md): 长期稳定，完整描述当前功能
- 模块概述和核心功能
- 完整的类/函数接口说明
- 配置参数详细说明
- 使用示例和最佳实践
- 常见问题和排查指南

**演进文档** (docs/modules/evolution/*.md): 滚动维护，记录变更历史
- 版本变更记录（按时间线组织）
- 新功能添加历史
- 重构和优化记录
- 性能改进追踪
- 包含文件位置和具体代码变更

**废弃文档** (docs/deprecated/*.md): 历史保存，追溯已移除功能
- 废弃功能的原始实现
- 废弃原因和替代方案
- 迁移指南和兼容性说明

**调优文档** (docs/hyperparameter_tuning_log.md): 实验记录，知识积累
- 所有超参数实验配置和结果
- 失败案例的根本原因分析
- 成功经验和最佳实践总结
- 监控指标范围和告警阈值
- 紧急修复方案和debug路径

#### 3. 变更记录格式标准
每次模块更新必须在对应演进文档中添加记录：

```markdown
## [版本日期] - [变更类型]

### 变更概述
- **改动原因**：[详细说明为什么做这个改动]
- **影响范围**：[列出受影响的文件和模块]
- **向后兼容性**：[说明是否破坏兼容性]

### 具体变更
1. **新增功能**：
   - 功能描述
   - 文件位置：`src/path/to/file.py:123-456`
   - 配置参数：`config.new_param`

2. **修改功能**：
   - 修改前行为：[原有功能描述]
   - 修改后行为：[新功能描述]
   - 文件位置：`src/path/to/file.py:123-456`

3. **废弃功能**：
   - 废弃原因：[说明为什么废弃]
   - 替代方案：[新的实现方式]
   - 迁移路径：[如何从旧版本迁移]
   - 保留位置：`docs/deprecated/component_name.md`

### 测试验证
- [x] 功能测试通过
- [x] 配置兼容性验证
- [x] 性能无明显回退

### 相关Issue/PR
- Issue: #123
- PR: #456
```

### 强制执行规则

#### 1. 模块更新检查清单

**代码变更时**:
- [ ] **更新模块接口文档** (docs/modules/[module].md)
  - [ ] 同步最新的类/函数接口
  - [ ] 更新配置参数说明
  - [ ] 更新使用示例
  - [ ] 确保接口文档完整准确
- [ ] **更新演进文档** (docs/modules/evolution/[module]_evolution.md)
  - [ ] 添加版本记录（包含日期、变更类型）
  - [ ] 记录具体变更和文件位置
  - [ ] 说明改动原因和影响范围
- [ ] **处理废弃功能** (如有)
  - [ ] 将废弃功能移至docs/deprecated/
  - [ ] 记录废弃原因和替代方案
- [ ] **配置文件变更** (如有)
  - [ ] 更新配置参数说明
  - [ ] 在配置中添加_comment注释
  - [ ] 更新文件位置索引
- [ ] **兼容性检查**
  - [ ] 验证向后兼容性
  - [ ] 更新迁移指南（如不兼容）
- [ ] **文档归档检查**
  - [ ] 演进文档是否超过10个版本或800行
  - [ ] 废弃文档是否超过5个功能记录
  - [ ] 如需归档，执行pop操作并更新索引

**训练实验时**:
- [ ] **更新调优文档** (docs/hyperparameter_tuning_log.md)
  - [ ] 记录实验配置（完整JSON片段）
  - [ ] 记录训练结果和关键指标
  - [ ] 分析失败原因（如失败）
  - [ ] 总结经验教训
  - [ ] 更新监控指标范围（如发现新阈值）
- [ ] **更新演进文档** (如配置变更被正式采用)
  - [ ] 将成功的配置变更记录为版本更新
  - [ ] 关联调优文档中的实验编号

#### 2. 废弃功能处理流程
1. **标记阶段**：在代码中添加`@deprecated`注释
2. **文档化**：在`docs/deprecated/`中创建详细文档
3. **保留期**：至少保留2个版本周期
4. **最终移除**：移至`archieve/`目录

#### 3. 文档长度管理规范
**文档长度限制**：
- 演进文档：最多保留**最近10个版本**的变更记录
- 废弃文档：最多保留**最近5个废弃功能**的详细记录
- 单个文档：不超过**800行**或**50KB**

**文档分割策略**：
```
docs/modules/
├── memory_management_evolution.md           # 当前活跃演进文档
└── archive/
    ├── memory_management_evolution_2024.md  # 2024年历史记录
    └── memory_management_evolution_2023.md  # 2023年历史记录

docs/deprecated/
├── deprecated_memory_management.md          # 当前废弃功能
└── archive/
    ├── deprecated_memory_management_2024.md # 2024年废弃功能
    └── deprecated_memory_management_2023.md # 2023年废弃功能
```

**滚动归档流程**：
1. **触发条件**: 文档超过长度限制或版本数限制
2. **分割原则**: 按年份或版本数分割，从旧到新进行pop操作
3. **归档位置**: 移至对应的`archive/`子目录
4. **索引更新**: 在当前文档中添加历史文档链接

**归档示例**：
```markdown
# 内存管理模块演进史

## 📚 历史文档索引
- [2024年演进记录](./archive/memory_management_evolution_2024.md)
- [2023年演进记录](./archive/memory_management_evolution_2023.md)

## 当前活跃版本 (2025年)

### [2025-09-27] - 配置统一化重构
...（最近10个版本的记录）
```

#### 5. 文档更新触发条件

**必须更新演进文档**:
- 新增核心模块或重要功能
- 修改现有模块的核心逻辑
- 废弃任何功能组件
- 修改配置参数结构
- 重构代码架构
- **文档长度超限需要归档**

**必须更新调优文档**:
- 完成重要的训练实验（无论成功失败）
- 发现新的超参数问题或解决方案
- 训练出现异常需要记录教训
- 找到新的最佳实践配置
- 更新监控指标阈值

**必须同时更新两者**:
- 配置文件的重大变更（需要在演进文档记录变更，在调优文档记录实验）
- 训练系统架构调整（演进文档记录架构，调优文档更新实验方法）

### 示例1：模块接口文档 + 演进文档

**模块接口文档** (docs/modules/ppo_training_system.md):
```markdown
# PPO训练系统模块接口文档

## 核心组件
- 配置文件: `configs/revolutionary_collapse_prevention.json`
- 训练入口: `vidur/simulator.py`
- PPO Scheduler: `vidur/scheduler/global_scheduler/ppo_scheduler_modular.py`

## 关键参数
- `lr`: 学习率，推荐范围1e-4到4e-4
- `clip_ratio`: PPO clip系数，标准值0.2-0.3
- `expert_guidance_weight`: Expert guidance权重，建议≤0.5

## 使用示例
python vidur/simulator.py --config configs/revolutionary_collapse_prevention.json --training_steps 20000
```

**对应演进文档** (docs/modules/evolution/ppo_training_system_evolution.md):
```markdown
# PPO训练系统演进史

## [2025-10-01] - 参数优化: 修复训练失败问题

### 变更概述
- **改动原因**: v3.0.0训练完全失败 (Entropy=1.14, ExplainedVar=0.12)
- **影响范围**: `configs/revolutionary_collapse_prevention.json` (所有PPO参数)
- **向后兼容性**: 直接修改配置,保持架构不变

### 具体变更
1. **PPO Hyperparameters优化**:
   - lr: 0.0003 → 0.0004 (+33%)
   - clip_ratio: 0.2 → 0.3 (+50%)
   - epochs: 8 → 12 (+50%)

2. **Forced-random机制降低**:
   - expert_guidance_weight: 1.5 → 0.5 (-67%)
   - 文件位置: `configs/revolutionary_collapse_prevention.json:125-126`
```

### 示例2：调优文档

**调优文档** (docs/hyperparameter_tuning_log.md):
```markdown
## 最新调参记录

### ❌ 实验6: v3.0.0 Revolutionary配置训练完全失败 (2025-09-30)

**配置**:
{
  "lr": 0.0003,
  "clip_ratio": 0.2,
  "expert_guidance_weight": 1.5
}

**结果**:
- ❌ Entropy = 1.14: policy未收敛
- ❌ ClipFraction = 0.88: 学习效率极低

**教训**:
- ⚠️ 不要同时使用多个forced-random机制
- ⚠️ History shortcut > 0.7会bypass学习

### ✅ 实验7: v3.1.0 优化配置 (2025-10-01)
**配置**: 降低所有forced-random权重，提高PPO超参数
**预期**: Entropy自然收敛, ExplainedVar>0.6
```

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
遵循**表达式解析-配置-架构-main接口-代码-验证六步法**，确保新功能能正确集成到训练系统中，避免静默失效。

### 配置系统架构概述

系统使用**动态表达式配置**机制，支持参数间的引用和计算：

```
完整配置流程:
第0步: 表达式解析    → 将 "#{...}" 表达式计算为实际数值
第1步: 配置层面验证  → 检查JSON结构和三层配置要求
第2步: 配置架构验证  → 验证Scheduler类型和Config类继承
第2B步: CLI参数转换  → training_config.py将JSON转为命令行参数
第3步: vidur.main接口 → CLI参数解析为SimulationConfig对象
第4步: 代码集成验证  → Scheduler初始化并使用参数
第5步: 运行验证      → 确认功能实际生效
```

### 强制检查清单

#### 第零步：配置表达式解析 🔢 (最先执行!)

**作用**: 在配置传递给任何组件前，先解析所有动态表达式，实现参数间的自动计算和同步。

**文件**: `vidur/config/config_expression_parser.py`

1. **表达式语法规则**
   ```json
   {
     "training_schedule": {
       "total_steps": 50000,
       "rollout_length": 256,
       "training_phases": {
         "exploration_ratio": 0.3,
         "balance_ratio": 0.5,
         "convergence_ratio": 0.2
       },
       "analysis_windows": {
         "short_term": 0.2,
         "medium_term": 0.8,
         "immediate": 0.04
       }
     },

     "ppo_config": {
       // ✅ 引用单个参数
       "rollout_len": "#{training_schedule.rollout_length}",

       // ✅ 支持数学运算
       "learning_rate_schedule": {
         "warmup_steps": "#{training_schedule.total_steps * training_schedule.training_phases.exploration_ratio}"
       }
     },

     "cluster_config": {
       "global_scheduler_config": {
         // ✅ 复合表达式
         "action_balance_window": "#{training_schedule.rollout_length * training_schedule.analysis_windows.short_term}",

         // ✅ 多级运算
         "detection_window": "#{training_schedule.rollout_length * training_schedule.analysis_windows.immediate}"
       }
     }
   }
   ```

2. **表达式解析流程**
   ```python
   # src/core/utils/infrastructure/config/training_config.py:26-37
   def load_config(config_path: str) -> Dict:
       """训练脚本首先调用此函数，表达式在这里被解析"""
       from vidur.config.config_expression_parser import load_config_with_expressions

       # 🔥 关键: 表达式解析在配置传递前完成
       config = load_config_with_expressions(config_path)
       return config

   # vidur/config/config_expression_parser.py:271-293
   def load_config_with_expressions(config_path: str) -> Dict[str, Any]:
       # 1. 加载原始JSON
       with open(config_path, 'r') as f:
           config = json.load(f)

       # 2. 验证training_schedule完整性
       TrainingScheduleValidator.validate_training_schedule(config['training_schedule'])

       # 3. 递归解析所有 "#{...}" 表达式
       parser = ConfigExpressionParser()
       resolved_config = parser.resolve_config_expressions(config)

       return resolved_config
   ```

3. **解析示例**
   ```json
   原始配置:
   {
     "training_schedule": {
       "total_steps": 50000,
       "rollout_length": 256,
       "training_phases": {"exploration_ratio": 0.3}
     },
     "ppo_config": {
       "rollout_len": "#{training_schedule.rollout_length}",
       "learning_rate_schedule": {
         "warmup_steps": "#{training_schedule.total_steps * training_schedule.training_phases.exploration_ratio}"
       }
     }
   }

   解析后配置:
   {
     "training_schedule": {...},  // 保持不变
     "ppo_config": {
       "rollout_len": 256,         // ✅ 表达式 → 数值
       "learning_rate_schedule": {
         "warmup_steps": 15000     // ✅ 50000 * 0.3 = 15000 (自动取整)
       }
     }
   }
   ```

4. **支持的数学运算**
   - 基本运算: `+`, `-`, `*`, `/`, `%`, `**`
   - 嵌套属性访问: `training_schedule.training_phases.exploration_ratio`
   - 复合表达式: `total_steps * 0.8`, `rollout_length * analysis_windows.short_term`
   - 自动类型推断: 包含`steps`、`window`等关键词的表达式结果自动取整

5. **表达式解析验证**
   ```bash
   # 验证表达式解析正确性
   python -c "
   from vidur.config.config_expression_parser import load_config_with_expressions
   config = load_config_with_expressions('configs/revolutionary_collapse_prevention.json')
   print('rollout_len:', config['ppo_config']['rollout_len'])
   print('warmup_steps:', config['ppo_config']['learning_rate_schedule']['warmup_steps'])
   print('action_balance_window:', config['cluster_config']['global_scheduler_config']['action_balance_window'])
   "
   ```

6. **表达式系统优势**
   - ✅ **统一配置源**: 所有时间窗口、步数从`training_schedule`统一计算
   - ✅ **自动同步**: 修改`total_steps`时，所有依赖参数自动重新计算
   - ✅ **配置可读**: 表达式直接展示参数关系，如`warmup_steps = total_steps * 0.3`
   - ✅ **避免不一致**: 消除手动计算导致的配置不一致问题

7. **新功能配置注意事项**
   - 如果新功能参数依赖`training_schedule`，**必须使用表达式语法**
   - 如果新功能参数是固定值，直接写数值即可
   - 表达式解析发生在所有配置处理之前，后续步骤看到的都是解析后的数值

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

0. **表达式解析正确性** 🔢 (最先检查!)
   - 表达式语法正确: `"#{training_schedule.field}"`
   - training_schedule完整性验证通过
   - 表达式计算结果符合预期(步数参数自动取整)
   - 验证命令: `python -c "from vidur.config.config_expression_parser import load_config_with_expressions; config = load_config_with_expressions('configs/config.json'); print(config['ppo_config']['rollout_len'])"`

1. **JSON配置结构正确性**
   - 扁平字段结构
   - 正确的scheduler_type指定
   - 三层配置结构遵循: training → ppo_config → cluster_config.global_scheduler_config

2. **Config类字段完整性**
   - 字段名与JSON完全匹配
   - 合理的default值

3. **training_config.py转换完整性** 🚨
   - build_ppo_args函数包含新功能参数处理
   - 表达式已在此步骤前解析完成(看到的是数值而非表达式)
   - 验证命令行参数生成正确：`python src/core/utils/infrastructure/config/training_config.py configs/your_config.json /tmp/test | grep new_feature`

4. **vidur.main接口兼容性**
   - 配置加载链路完整
   - 字段传递无丢失

5. **Scheduler集成正确性**
   - 参数读取正确(getattr字段名与JSON一致)
   - 初始化条件明确

6. **运行时验证**
   - 日志确认初始化
   - CSV数据非空
   - 训练行为改变

### 调试命令序列 🛠️

```bash
# 0. 表达式解析验证 (最先检查!)
python -c "
from vidur.config.config_expression_parser import load_config_with_expressions
config = load_config_with_expressions('configs/config.json')
print('=== 表达式解析结果 ===')
print('rollout_len:', config.get('ppo_config', {}).get('rollout_len', 'NOT_FOUND'))
print('warmup_steps:', config.get('ppo_config', {}).get('learning_rate_schedule', {}).get('warmup_steps', 'NOT_FOUND'))
print('action_balance_window:', config.get('cluster_config', {}).get('global_scheduler_config', {}).get('action_balance_window', 'NOT_FOUND'))
print()
print('✅ 检查点: 所有表达式应已被解析为数值，不应有 \"#{...}\" 字符串')
"

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
0. **表达式解析优先** 🔢：配置加载时首先解析所有 `"#{...}"` 表达式，后续步骤看到的都是解析后的数值
1. **三层配置结构严格遵循**：`training` → `ppo_config` → `cluster_config.global_scheduler_config`，参数放错层级会导致 `Error: 'field_name'` 错误
2. **vidur.main接口兼容性**：还要在vidur.main中也添加对应接口才行！
3. **完整验证链路**：必须验证从表达式解析 → JSON配置 → CLI参数 → Scheduler实例的完整传递链路
4. **训练行为确认**：确保新功能确实改变训练行为，而不仅仅是"运行无错误"

**配置完整流程记忆**：
```
第0步: 表达式解析 → "#{training_schedule.rollout_length}" 变为 256
第1步: training层 → 训练脚本基础参数（replicas, requests, qps）
第2步: ppo_config层 → PPO算法参数（lr, gamma, clip, rollout_len等）
第3步: cluster_config层 → 新功能和scheduler配置（新功能参数放这里）
第4步: CLI转换 → training_config.py将JSON转为--参数
第5步: 对象解析 → vidur.main解析CLI参数为Config对象
第6步: 组件初始化 → Scheduler读取Config并初始化组件
```

**表达式系统使用建议**：
- ✅ 依赖training_schedule的参数使用表达式: `"rollout_len": "#{training_schedule.rollout_length}"`
- ✅ 相对比例计算使用表达式: `"warmup_steps": "#{training_schedule.total_steps * 0.3}"`
- ✅ 复合计算使用表达式: `"window": "#{training_schedule.rollout_length * training_schedule.analysis_windows.short_term}"`
- ❌ 独立固定值不使用表达式: `"latency_weight": 1.0` (直接数值即可)
