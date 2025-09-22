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
