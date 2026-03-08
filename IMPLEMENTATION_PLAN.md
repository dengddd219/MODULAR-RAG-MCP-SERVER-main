# 模型管理与性能监控实施计划

## 📋 改动总览

本次改动旨在实现：
1. **快速切换不同大语言模型** - 支持运行时动态切换，无需重启
2. **模型性能监控UI** - Dashboard展示成本、延迟、Token使用等指标
3. **小模型+大模型混合策略** - 根据意图复杂度自动路由
4. **测评指标集成** - 质量评估与性能指标结合

---

## 🗂️ 改动文件清单

### 核心功能层（数据追踪）
- ✅ `src/libs/llm/model_evaluator.py` - **已存在**，需要增强
- ✅ `src/libs/llm/model_manager.py` - **已存在**，需要增强
- ⚠️ `src/libs/llm/base_llm.py` - **需要修改**：集成追踪
- ⚠️ `src/core/response/rag_generator.py` - **需要修改**：集成ModelManager和意图路由

### UI展示层（Dashboard）
- ⚠️ `src/observability/dashboard/pages/model_benchmark.py` - **新增**：模型性能监控页面
- ⚠️ `src/observability/dashboard/pages/chat_interface.py` - **需要修改**：添加模型选择器
- ⚠️ `src/observability/dashboard/app.py` - **需要修改**：注册新页面

### 配置层
- ⚠️ `config/settings.yaml` - **需要修改**：添加模型路由配置
- ⚠️ `config/models.yaml` - **新增**：模型列表配置（可选）

### 测试层
- ⚠️ `tests/unit/test_model_manager.py` - **新增**：ModelManager测试
- ⚠️ `tests/unit/test_model_evaluator.py` - **新增**：ModelEvaluator测试

---

## 🔄 实施流程（分阶段）

### 阶段1：基础数据追踪增强 ⭐ 优先级最高

**目标**：确保所有LLM调用都被自动追踪

#### 1.1 增强 BaseLLM 追踪能力
**文件**：`src/libs/llm/base_llm.py`

**改动**：
- 在 `BaseLLM.chat()` 方法中添加可选的 `ModelEvaluator` 参数
- 如果提供了 `evaluator`，自动记录：
  - 调用开始时间
  - 调用结束时间（计算延迟）
  - Token使用（从 `ChatResponse.usage` 提取）
  - 成本计算

**设计决策**：
- 使用装饰器模式还是直接集成？
  - **选择**：直接集成（更简单，性能更好）
- 如何传递 `ModelEvaluator` 实例？
  - **选择**：通过 `__init__` 或 `chat()` 的 `**kwargs` 传递（灵活）

#### 1.2 在 RAGGenerator 中集成追踪
**文件**：`src/core/response/rag_generator.py`

**改动**：
- 在 `__init__` 中初始化 `ModelEvaluator` 和 `ModelManager`
- 在 `generate()` 方法中：
  ```python
  # 获取当前模型ID
  model_id = self.model_manager.get_current_model_id()
  
  # 使用追踪上下文
  with self.model_manager.track_call(model_id, query) as metrics:
      response = self.llm.chat(messages, trace=trace)
      # 更新metrics
      metrics.prompt_tokens = response.usage.get("prompt_tokens", 0)
      metrics.completion_tokens = response.usage.get("completion_tokens", 0)
  ```

**依赖关系**：
- 需要先完成 1.1（BaseLLM 支持追踪）

---

### 阶段2：模型管理增强 ⭐ 优先级高

**目标**：支持多模型注册和快速切换

#### 2.1 增强 ModelManager
**文件**：`src/libs/llm/model_manager.py`

**改动**：
- 添加从配置文件加载模型列表的方法
- 支持运行时动态注册模型
- 增强 `get_llm()` 方法，支持模型切换时自动清理缓存

#### 2.2 创建模型配置文件
**文件**：`config/models.yaml`（新增）

**内容示例**：
```yaml
models:
  - model_id: "openai-gpt-4o-mini"
    provider: "openai"
    model_name: "gpt-4o-mini"
    display_name: "GPT-4o Mini"
    description: "快速且经济的模型，适合简单查询"
    is_small_model: true
    config_override:
      temperature: 0.2
      max_tokens: 512
  
  - model_id: "openai-gpt-4o"
    provider: "openai"
    model_name: "gpt-4o"
    display_name: "GPT-4o"
    description: "高性能模型，适合复杂推理"
    is_small_model: false
    config_override:
      temperature: 0.3
      max_tokens: 1024
```

#### 2.3 在 RAGGenerator 中集成 ModelManager
**文件**：`src/core/response/rag_generator.py`

**改动**：
- 修改 `__init__` 方法，接受 `model_manager` 参数
- 修改 `generate()` 方法，使用 `ModelManager.get_llm()` 而不是 `LLMFactory.create()`
- 支持通过参数指定使用的模型ID

**依赖关系**：
- 需要先完成 2.1 和 2.2

---

### 阶段3：UI界面开发 ⭐ 优先级高

**目标**：在Dashboard中展示模型性能和切换功能

#### 3.1 创建模型性能监控页面
**文件**：`src/observability/dashboard/pages/model_benchmark.py`（新增）

**功能模块**：
1. **模型对比表**
   - 读取 `ModelEvaluator` 的统计数据
   - 展示：模型名、调用次数、成功率、平均延迟、P95延迟、总Token、总成本
   - 支持排序和筛选

2. **性能趋势图**
   - 使用 `st.line_chart` 或 `plotly` 展示时间序列
   - 支持多模型对比

3. **成本分析**
   - 总成本、平均成本/次
   - Token成本分布（输入 vs 输出）

4. **延迟分析**
   - 延迟分布直方图
   - P50/P95/P99 分位数

**实现要点**：
- 从 `ModelEvaluator.load_metrics()` 读取数据
- 使用 `ModelEvaluator.get_stats()` 获取聚合统计
- 使用 Streamlit 的图表组件展示

#### 3.2 在Chat界面添加模型选择器
**文件**：`src/observability/dashboard/pages/chat_interface.py`

**改动**：
- 在页面顶部添加模型选择下拉框
- 使用 `st.selectbox` 或 `st.radio` 展示可用模型
- 选择后更新 `ModelManager.current_model_id`
- 在聊天过程中显示当前使用的模型

**实现要点**：
```python
# 获取ModelManager实例
model_manager = get_model_manager()  # 需要实现单例或session state

# 显示模型选择器
selected_model = st.selectbox(
    "选择模型",
    options=[m.model_id for m in model_manager.list_models()],
    format_func=lambda x: model_manager.get_model_config(x).display_name
)

# 更新当前模型
if selected_model != model_manager.get_current_model_id():
    model_manager.set_current_model(selected_model)
```

#### 3.3 注册新页面到Dashboard
**文件**：`src/observability/dashboard/app.py`

**改动**：
- 添加 `_page_model_benchmark()` 函数
- 在 `pages` 列表中注册新页面

**依赖关系**：
- 需要先完成 3.1

---

### 阶段4：小模型+大模型混合策略 ⭐ 优先级中

**目标**：根据意图复杂度自动选择模型

#### 4.1 在配置中添加路由规则
**文件**：`config/settings.yaml`

**改动**：
```yaml
llm_routing:
  enabled: true
  small_model: "openai-gpt-4o-mini"
  large_model: "openai-gpt-4o"
  simple_intents: ["fabric_care", "faq", "greeting"]
  complexity_threshold: 0.7  # 意图置信度阈值
```

#### 4.2 在 RAGGenerator 中实现路由逻辑
**文件**：`src/core/response/rag_generator.py`

**改动**：
- 在 `generate()` 方法开始处：
  ```python
  # 1. 使用IntentRouter判断意图复杂度
  routing_result = self.intent_router.route(query)
  
  # 2. 根据意图选择模型
  if routing_result.intent in self.simple_intents:
      model_id = self.small_model_id
  else:
      model_id = self.large_model_id
  
  # 3. 使用选定的模型
  llm = self.model_manager.get_llm(model_id)
  ```

**依赖关系**：
- 需要先完成阶段2（ModelManager）
- 需要 `IntentRouter` 已存在（✅ 已存在）

---

### 阶段5：测评指标集成 ⭐ 优先级中

**目标**：将质量评估指标与性能指标结合

#### 5.1 扩展 ModelMetrics
**文件**：`src/libs/llm/model_evaluator.py`

**改动**：
- 在 `ModelMetrics` 中添加 `quality_metrics` 字段
- 支持记录：faithfulness、answer_relevancy、context_relevance 等

#### 5.2 在 RAGGenerator 中集成评估
**文件**：`src/core/response/rag_generator.py`

**改动**：
- 在生成回答后，调用 `RagasEvaluator` 进行评估
- 将评估结果记录到 `ModelMetrics.quality_metrics`

**依赖关系**：
- 需要 `RagasEvaluator` 已存在（✅ 已存在）

---

## 📊 数据流图

```
用户查询
  ↓
Chat界面（选择模型）
  ↓
RAGGenerator.generate()
  ↓
IntentRouter.route() → 判断意图复杂度
  ↓
ModelManager.get_llm(model_id) → 获取LLM实例
  ↓
BaseLLM.chat() → 调用LLM
  ↓
ModelEvaluator.track_call() → 记录指标
  ↓
返回结果 + 更新Dashboard数据
```

---

## ✅ 验收标准

### 阶段1验收
- [ ] 所有LLM调用都自动记录到 `data/metrics/metrics_cache.jsonl`
- [ ] 记录包含：延迟、Token使用、成本

### 阶段2验收
- [ ] 可以从配置文件加载多个模型
- [ ] 可以在运行时切换模型
- [ ] 切换后立即生效

### 阶段3验收
- [ ] Dashboard显示"Model Benchmark"页面
- [ ] 页面展示所有模型的性能对比
- [ ] Chat界面可以切换模型

### 阶段4验收
- [ ] 简单意图自动使用小模型
- [ ] 复杂意图自动使用大模型
- [ ] 在Dashboard中显示模型使用统计

### 阶段5验收
- [ ] 质量指标记录到metrics
- [ ] Dashboard展示质量指标

---

## 🚀 实施顺序建议

**第一周**：
1. 阶段1：基础数据追踪增强
2. 阶段2：模型管理增强

**第二周**：
3. 阶段3：UI界面开发

**第三周**：
4. 阶段4：混合策略
5. 阶段5：测评指标集成

---

## 🔍 关键技术决策

### 决策1：追踪方式
- **选择**：在 `BaseLLM.chat()` 中直接集成，而不是装饰器
- **理由**：更简单，性能更好，易于维护

### 决策2：模型配置存储
- **选择**：使用 `config/models.yaml` 单独文件
- **理由**：与 `settings.yaml` 分离，更清晰，易于管理

### 决策3：UI框架
- **选择**：继续使用 Streamlit
- **理由**：已有Dashboard基础，保持一致性

### 决策4：混合策略触发点
- **选择**：在 `RAGGenerator.generate()` 中实现
- **理由**：这是生成阶段的入口，逻辑集中

---

## 📝 注意事项

1. **向后兼容**：确保现有代码不受影响
2. **性能**：追踪不应显著影响响应时间
3. **错误处理**：模型切换失败时的降级策略
4. **数据持久化**：metrics数据需要定期清理或归档







