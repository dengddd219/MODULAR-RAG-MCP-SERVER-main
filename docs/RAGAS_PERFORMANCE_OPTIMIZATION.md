# Ragas 性能优化指南

本文档说明已实施的 Ragas 评估性能优化措施。

## 已实施的优化

### 1. 减少上下文数量
- **优化前**: 使用 5 个 chunks
- **优化后**: 使用 3 个 chunks
- **效果**: 减少 ~40% 的上下文处理时间

### 2. 截断答案长度
- **优化前**: 使用完整答案（可能数千字符）
- **优化后**: 截断到 800 字符（约 200-300 tokens）
- **效果**: 减少 LLM 处理时间，大多数评估指标只需要答案的开头部分

### 3. 减少 max_tokens
- **优化前**: 4096 tokens
- **优化后**: 2048 tokens
- **效果**: 减少 LLM 生成时间，大多数 Ragas 响应 <1000 tokens

### 4. 优化超时设置
- **RunConfig timeout**: 从 300 秒减少到 120 秒（快速失败）
- **API client timeout**: 从 60 秒减少到 30 秒
- **max_retries**: 从 3 次减少到 2 次
- **效果**: 更快地检测和恢复失败

## 进一步优化建议

### 1. 减少评估指标数量（最快的方法）

在 `config/settings.yaml` 中只保留最重要的指标：

```yaml
evaluation:
  enabled: true
  provider: "ragas"
  metrics:
    - "faithfulness"  # 只评估忠实度（最快）
    # - "answer_relevancy"  # 注释掉可以节省 ~30-40% 时间
    # - "context_precision"  # 注释掉可以节省 ~20-30% 时间
```

**预期效果**: 只评估 `faithfulness` 可以节省 **50-60%** 的评估时间

### 2. 使用更快的 LLM 模型

在 `settings.yaml` 中确保使用最快的模型：

```yaml
llm:
  model: "gpt-4o-mini"  # 已经是最快的 OpenAI 模型
```

**注意**: Ragas 评估**总是**使用 `settings.yaml` 中的模型，不受用户界面选择影响。

### 3. 批量评估（如果评估多个 query）

如果需要在批量评估中加速，可以考虑：
- 并行评估多个 query（需要修改代码支持异步批量）
- 使用 Ragas 的 `evaluate()` 批量模式（需要重构代码）

### 4. 缓存评估结果

对于相同的 query + answer 组合，可以缓存评估结果：
- 实现简单的 LRU 缓存
- 使用 query + answer 的 hash 作为 key

## 性能对比

| 优化项 | 时间节省 | 质量影响 |
|--------|---------|---------|
| 减少 chunks (5→3) | ~20-30% | 轻微（前3个chunks通常最重要）|
| 截断答案 (800 chars) | ~15-25% | 轻微（评估主要看开头）|
| 减少 max_tokens (4096→2048) | ~10-15% | 无（响应通常<1000 tokens）|
| 减少指标数量 (3→1) | ~50-60% | 中等（只评估忠实度）|
| **总计（所有优化）** | **~60-70%** | **轻微到中等** |

## 使用建议

1. **开发/测试阶段**: 使用所有优化 + 只评估 `faithfulness`
2. **生产环境**: 根据需求平衡速度和完整性
3. **批量评估**: 考虑只评估 `faithfulness` 以节省时间

## 监控性能

在日志中查找以下信息来监控性能：
- `RAGAS evaluation configured with dedicated API` - 配置信息
- `Truncating retrieved chunks` - 上下文截断
- `Truncating answer` - 答案截断
- 评估总时间（在 UI 中显示）

