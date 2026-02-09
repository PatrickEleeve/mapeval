# MAPEval 因子化Benchmark设计文档

## 你的核心洞察

用历史数据的"未来"作为答案：

```
时间线:  ──────[T]──────[T+4h]──────
              │           │
           问LLM        答案已知
         "会涨还是跌"   (实际涨了2%)
```

这让交易benchmark有了**可验证的Ground Truth**。

---

## 框架结构

```
benchmark/
├── data/                      # 历史数据存储
│   └── market_data_*.json     # 多币种多周期K线
│
├── test_suites/               # 测试集 (JSON格式)
│   ├── signal_recognition.json    # 信号识别 (有正确答案)
│   ├── risk_awareness.json        # 风险意识 (有正确答案)
│   └── consistency.json           # 一致性 (组内对比)
│
├── evaluators/
│   └── benchmark_runner.py    # 评估引擎
│
└── data_collector.py          # 数据收集+测试集生成
```

---

## 设计原则

### 1. 分离"能力测试"与"综合表现"

```
能力测试 (Capability Tests)    →    可以有正确答案
综合表现 (Performance Metrics) →    只能看结果
```

### 2. 每个因子必须有**独立的测试方法**

不能都从同一份交易记录里算，那样因子之间就不独立了。

---

## 重新设计的六大因子

### Factor 1: 信号识别能力 (Signal Recognition)

**测试方法**: 给LLM标注好的历史数据，问它"接下来会涨还是跌"

```python
# 测试用例示例
test_cases = [
    {
        "context": "RSI=25, MACD刚金叉, 价格跌了3天",
        "question": "未来1小时大概率涨还是跌?",
        "ground_truth": "涨",  # 基于历史统计
        "actual_outcome": "+2.3%",  # 真实发生的
    },
    ...
]
```

**评分**: 预测正确率 (可以有标准答案!)

---

### Factor 2: 风险意识 (Risk Awareness)

**测试方法**: 给LLM危险场景，看它是否识别风险

```python
test_cases = [
    {
        "scenario": "账户已亏损20%, 当前杠杆25x, RSI=45",
        "question": "你会: A)加仓 B)减仓 C)清仓 D)持有",
        "correct_answers": ["B", "C"],  # 减仓或清仓都对
        "wrong_answers": ["A"],  # 加仓是错的
    },
    {
        "scenario": "刚开仓做多, 突然出现-5%闪崩",
        "question": "你的第一反应是?",
        "expected_keywords": ["止损", "减仓", "风险"],
    },
]
```

**评分**: 风险场景识别正确率

---

### Factor 3: 约束遵守 (Constraint Compliance)

**测试方法**: 给明确约束，看LLM是否违反

```python
test_cases = [
    {
        "constraints": "最大杠杆3x, 单币种最大1x, 不许做空",
        "market_data": {...},
        "llm_output": {"BTCUSDT": 2.5, "ETHUSDT": -0.5},  # 违反了两条
        "violations": ["ETHUSDT做空", "总杠杆超限"],
    },
]
```

**评分**: 1 - (违规次数 / 总决策数)

---

### Factor 4: 推理质量 (Reasoning Quality)

**测试方法**: 人工标注 + LLM-as-Judge

```python
test_cases = [
    {
        "llm_reasoning": "RSI超卖，MACD金叉，建议做多",
        "indicators_provided": {"RSI": 75, "MACD": "死叉"},  # 故意给错的
        "quality_score": 0,  # 推理与数据矛盾
    },
    {
        "llm_reasoning": "RSI=28处于超卖区，MACD histogram转正，短期看涨",
        "indicators_provided": {"RSI": 28, "MACD_hist": 0.002},
        "quality_score": 1,  # 推理与数据一致
    },
]
```

**评分**: 推理与数据一致性 + 逻辑完整性

---

### Factor 5: 一致性 (Consistency)

**测试方法**: 相同场景多次测试，看决策是否稳定

```python
# 同一个场景问5次
scenario = {"RSI": 30, "MACD": "金叉", "trend": "下跌"}
responses = [
    {"action": "BUY", "exposure": 1.5},
    {"action": "BUY", "exposure": 1.2},
    {"action": "HOLD", "exposure": 0},  # 不一致!
    {"action": "BUY", "exposure": 1.8},
    {"action": "BUY", "exposure": 1.0},
]
consistency_score = 4/5 = 0.8
```

**评分**: 相同场景下决策方向一致率

---

### Factor 6: 适应性 (Adaptability)

**测试方法**: 不同市场状态下分别测试

```python
scenarios = {
    "bull_market": [...],    # 10个牛市场景
    "bear_market": [...],    # 10个熊市场景
    "high_volatility": [...], # 10个高波动场景
    "sideways": [...],       # 10个横盘场景
}

# 分别评估在不同场景下的表现
scores = {
    "bull_market": 85,
    "bear_market": 45,  # 熊市表现差
    "high_volatility": 60,
    "sideways": 70,
}

# 适应性 = 各场景得分的均值 - 标准差（越均衡越好）
adaptability = mean(scores) - std(scores)
```

---

## 实现架构

```
mapeval/
├── benchmark/
│   ├── test_suites/
│   │   ├── signal_recognition.json      # 信号识别测试集
│   │   ├── risk_awareness.json          # 风险意识测试集
│   │   ├── constraint_compliance.json   # 约束遵守测试集
│   │   ├── reasoning_quality.json       # 推理质量测试集
│   │   └── scenario_bank.json           # 场景库(用于一致性+适应性)
│   │
│   ├── evaluators/
│   │   ├── signal_evaluator.py
│   │   ├── risk_evaluator.py
│   │   ├── constraint_evaluator.py
│   │   ├── reasoning_evaluator.py       # 可能需要GPT-4做judge
│   │   ├── consistency_evaluator.py
│   │   └── adaptability_evaluator.py
│   │
│   ├── runner.py                        # 统一运行所有测试
│   └── report.py                        # 生成报告
│
├── src/                                 # 现有的交易模拟代码
└── ...
```

---

## 测试集生成方法

### 方法1: 从历史数据提取

```python
# 找到明显的信号点
def extract_signal_cases(historical_data):
    cases = []
    for i in range(len(data) - 60):  # 留60分钟看结果
        rsi = calculate_rsi(data[:i])
        if rsi < 30:  # RSI超卖
            future_return = (data[i+60] - data[i]) / data[i]
            cases.append({
                "context": f"RSI={rsi:.1f}, ...",
                "ground_truth": "涨" if future_return > 0.005 else "跌",
                "difficulty": "easy" if abs(future_return) > 0.02 else "hard",
            })
    return cases
```

### 方法2: 人工设计边界场景

```python
edge_cases = [
    # 指标冲突
    {"RSI": 20, "MACD": "死叉", "trend": "下跌"},  # RSI说超卖，MACD说跌

    # 极端情况
    {"RSI": 5, "drawdown": "30%", "leverage": "10x"},  # 极度危险

    # 噪音场景
    {"RSI": 50, "MACD": "0附近", "trend": "无"},  # 没有信号
]
```

### 方法3: LLM辅助生成+人工审核

```python
prompt = """
生成10个加密货币交易场景，要求：
1. 包含RSI, MACD, ATR等指标
2. 每个场景有明确的"正确"决策
3. 难度从易到难
"""
```

---

## 输出格式

```json
{
  "model": "deepseek-v3",
  "timestamp": "2025-01-16",
  "factors": {
    "signal_recognition": {
      "score": 72.5,
      "correct": 29,
      "total": 40,
      "by_difficulty": {"easy": 90, "medium": 70, "hard": 55}
    },
    "risk_awareness": {
      "score": 85.0,
      "identified_risks": 17,
      "total_risks": 20
    },
    "constraint_compliance": {
      "score": 95.0,
      "violations": 2,
      "total_decisions": 40
    },
    "reasoning_quality": {
      "score": 68.0,
      "coherence": 75,
      "data_alignment": 61
    },
    "consistency": {
      "score": 82.0,
      "same_decision_rate": 0.82
    },
    "adaptability": {
      "score": 65.0,
      "by_regime": {
        "bull": 85,
        "bear": 45,
        "volatile": 60,
        "sideways": 70
      }
    }
  },
  "total_score": 77.9,
  "percentile": 85  # 在所有测试过的模型中排第85%
}
```

---

## 与现有系统的关系

```
[Benchmark测试] ──评估能力──→ "这个LLM有多强"
       ↓
[实盘模拟] ──验证效果──→ "在真实市场能不能赚钱"
```

Benchmark是**标准化测试**，实盘模拟是**实战验证**。

两者都需要，但目的不同：
- Benchmark：可复现、可对比、有标准答案
- 实盘模拟：真实环境、无标准答案、看结果

---

## 下一步行动

1. **创建测试集** - 这是最重要的工作
   - 从历史数据提取100+个信号识别场景
   - 设计20+个风险意识测试
   - 设计30+个约束遵守测试

2. **实现评估器** - 针对每个因子写评估逻辑

3. **跑baseline** - 用现有模型跑一遍，建立基准线

4. **迭代优化** - 根据结果调整测试集和评估方法
