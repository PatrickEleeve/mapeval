"""
Benchmark Runner - 统一运行所有测试并评估LLM

核心流程：
1. 加载测试用例
2. 向LLM发送问题
3. 解析LLM回答
4. 与Ground Truth对比打分
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class EvaluationResult:
    """单个测试用例的评估结果"""
    case_id: str
    category: str
    passed: bool
    score: float  # 0-1
    llm_response: str
    parsed_answer: Any
    ground_truth: Any
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkScore:
    """完整的Benchmark评分"""
    model: str
    provider: str
    timestamp: str
    total_score: float
    factor_scores: Dict[str, float]
    detailed_results: List[EvaluationResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "total_score": round(self.total_score, 2),
            "factor_scores": {k: round(v, 2) for k, v in self.factor_scores.items()},
            "summary": {
                "total_cases": len(self.detailed_results),
                "passed": sum(1 for r in self.detailed_results if r.passed),
                "failed": sum(1 for r in self.detailed_results if not r.passed),
            },
            "by_category": self._group_by_category(),
            "metadata": self.metadata,
        }

    def _group_by_category(self) -> Dict:
        groups = {}
        for result in self.detailed_results:
            if result.category not in groups:
                groups[result.category] = {"total": 0, "passed": 0, "score": 0}
            groups[result.category]["total"] += 1
            groups[result.category]["passed"] += int(result.passed)
            groups[result.category]["score"] += result.score

        for cat in groups:
            if groups[cat]["total"] > 0:
                groups[cat]["accuracy"] = round(
                    groups[cat]["passed"] / groups[cat]["total"] * 100, 1
                )
                groups[cat]["avg_score"] = round(
                    groups[cat]["score"] / groups[cat]["total"] * 100, 1
                )

        return groups


class LLMEvaluator:
    """LLM评估器 - 向LLM发送测试问题并解析回答"""

    # 系统提示模板
    SYSTEM_PROMPTS = {
        "signal_recognition": """你是一个加密货币交易分析师。
根据给定的技术指标数据，预测未来价格走势方向。

只回答: LONG(做多), SHORT(做空), 或 HOLD(观望)
同时给出简短理由(一句话)。

回复格式:
DIRECTION: [LONG/SHORT/HOLD]
REASON: [你的理由]
CONFIDENCE: [0.0-1.0]""",

        "risk_awareness": """你是一个风险管理专家。
根据给定的市场状况，判断应该采取什么行动。

可选行动: REDUCE(减仓), CLOSE(平仓), HOLD(持有), INCREASE(加仓), LEVERAGE_UP(加杠杆), LEVERAGE_DOWN(降杠杆)

回复格式:
ACTION: [你建议的行动]
REASON: [为什么]
RISK_LEVEL: [LOW/MEDIUM/HIGH/EXTREME]""",

        "consistency": """你是一个交易决策系统。
根据市场数据给出交易方向建议。

只回答: LONG(做多), SHORT(做空), 或 HOLD(观望)

回复格式:
DIRECTION: [LONG/SHORT/HOLD]
CONFIDENCE: [0.0-1.0]""",
    }

    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4-turbo",
        base_url: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model

        if OpenAI is None:
            raise ImportError("openai package required")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        elif provider == "deepseek":
            client_kwargs["base_url"] = "https://api.deepseek.com"

        self.client = OpenAI(**client_kwargs)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # 低温度保证一致性
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

    def evaluate_signal_recognition(self, case: Dict) -> EvaluationResult:
        """评估信号识别能力"""

        context = case["market_context"]
        user_prompt = f"""
当前{case['symbol']}市场状态:
- 价格: {context['price']}
- RSI(14): {context['rsi_14']}
- MACD: {context['macd']}, Signal: {context['macd_signal']}, Histogram: {context['macd_histogram']}
- ATR%: {context['atr_pct']}%
- 布林带位置: {context['bb_position']} (0=下轨, 1=上轨)
- 趋势: {context['trend']}
- 波动率排名: {context['volatility_rank']}

检测到的信号: {', '.join(case.get('detected_signals', ['无明显信号']))}

请预测未来{case['ground_truth']['lookahead_bars']}根K线的方向。
"""

        response = self._call_llm(
            self.SYSTEM_PROMPTS["signal_recognition"],
            user_prompt
        )

        # 解析回答
        parsed = self._parse_direction_response(response)
        ground_truth = case["ground_truth"]["direction"]

        # 评分
        passed = parsed.get("direction") == ground_truth
        score = 1.0 if passed else 0.0

        # 部分分数：方向错误但有合理理由
        if not passed and parsed.get("confidence", 0) < 0.5:
            score = 0.3  # 不确定时给部分分

        return EvaluationResult(
            case_id=case["id"],
            category="signal_recognition",
            passed=passed,
            score=score,
            llm_response=response,
            parsed_answer=parsed,
            ground_truth=case["ground_truth"],
            details={
                "difficulty": case.get("difficulty"),
                "actual_return": case["ground_truth"]["actual_return_pct"],
            }
        )

    def evaluate_risk_awareness(self, case: Dict) -> EvaluationResult:
        """评估风险意识"""

        context = case["market_context"]
        user_prompt = f"""
市场状况:
- 资产: {case['symbol']}
- 当前价格: {context['price']}
- RSI: {context['rsi_14']}
- 回撤: {context['drawdown_pct']}%
- 波动率: {context['atr_pct']}%
- 波动率排名: {context['volatility_rank']}
- 趋势: {context['trend']}
- MACD Histogram: {context['macd_histogram']}

风险场景: {case['scenario']}

{case['question']}
"""

        response = self._call_llm(
            self.SYSTEM_PROMPTS["risk_awareness"],
            user_prompt
        )

        # 解析回答
        parsed = self._parse_action_response(response)
        action = parsed.get("action", "").upper()

        # 评分
        correct_actions = [a.upper() for a in case["ground_truth"]["correct_actions"]]
        wrong_actions = [a.upper() for a in case["ground_truth"]["wrong_actions"]]

        if action in correct_actions:
            passed = True
            score = 1.0
        elif action in wrong_actions:
            passed = False
            score = 0.0
        else:
            # 不在预设答案中，给部分分
            passed = False
            score = 0.5

        # 风险识别加分
        risk_level = parsed.get("risk_level", "").upper()
        if risk_level in ["HIGH", "EXTREME"]:
            score = min(1.0, score + 0.2)

        return EvaluationResult(
            case_id=case["id"],
            category="risk_awareness",
            passed=passed,
            score=score,
            llm_response=response,
            parsed_answer=parsed,
            ground_truth=case["ground_truth"],
            details={"scenario": case["scenario"]}
        )

    def evaluate_consistency(
        self,
        cases: List[Dict],
    ) -> List[EvaluationResult]:
        """评估一致性 - 需要批量处理同组用例"""

        # 按scenario_group分组
        groups = {}
        for case in cases:
            group = case.get("scenario_group", "default")
            if group not in groups:
                groups[group] = []
            groups[group].append(case)

        results = []

        for group_id, group_cases in groups.items():
            responses = []

            for case in group_cases:
                context = case["market_context"]
                user_prompt = f"""
{case.get('symbol', 'BTC')}市场数据:
- RSI: {context['rsi_14']}
- MACD Histogram: {context['macd_histogram']}
- ATR%: {context['atr_pct']}%
- 趋势: {context['trend']}

{case['question']}
"""
                response = self._call_llm(
                    self.SYSTEM_PROMPTS["consistency"],
                    user_prompt
                )
                parsed = self._parse_direction_response(response)
                responses.append({
                    "case": case,
                    "response": response,
                    "parsed": parsed,
                })

            # 计算一致性
            directions = [r["parsed"].get("direction") for r in responses]
            if not directions:
                continue

            # 找最常见的方向
            from collections import Counter
            direction_counts = Counter(directions)
            most_common = direction_counts.most_common(1)[0][0]
            consistency_rate = direction_counts[most_common] / len(directions)

            # 为每个用例创建结果
            for r in responses:
                is_consistent = r["parsed"].get("direction") == most_common

                results.append(EvaluationResult(
                    case_id=r["case"]["id"],
                    category="consistency",
                    passed=is_consistent,
                    score=consistency_rate,  # 用整组一致性作为分数
                    llm_response=r["response"],
                    parsed_answer=r["parsed"],
                    ground_truth={"expected": "consistent_with_group"},
                    details={
                        "group": group_id,
                        "consistency_rate": consistency_rate,
                        "group_majority": most_common,
                    }
                ))

        return results

    def _parse_direction_response(self, response: str) -> Dict:
        """解析方向预测回答"""
        result = {"direction": None, "confidence": 0.5, "reason": ""}

        for line in response.split("\n"):
            line = line.strip().upper()
            if line.startswith("DIRECTION:"):
                value = line.replace("DIRECTION:", "").strip()
                if "LONG" in value:
                    result["direction"] = "LONG"
                elif "SHORT" in value:
                    result["direction"] = "SHORT"
                elif "HOLD" in value:
                    result["direction"] = "HOLD"
            elif line.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REASON:"):
                result["reason"] = line.replace("REASON:", "").strip()

        # 如果格式解析失败，尝试直接匹配
        if result["direction"] is None:
            upper_response = response.upper()
            if "LONG" in upper_response and "SHORT" not in upper_response:
                result["direction"] = "LONG"
            elif "SHORT" in upper_response and "LONG" not in upper_response:
                result["direction"] = "SHORT"
            elif "HOLD" in upper_response:
                result["direction"] = "HOLD"

        return result

    def _parse_action_response(self, response: str) -> Dict:
        """解析行动建议回答"""
        result = {"action": None, "risk_level": None, "reason": ""}

        for line in response.split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("ACTION:"):
                result["action"] = line_upper.replace("ACTION:", "").strip()
            elif line_upper.startswith("RISK_LEVEL:"):
                result["risk_level"] = line_upper.replace("RISK_LEVEL:", "").strip()
            elif line_upper.startswith("REASON:"):
                result["reason"] = line.replace("REASON:", "").strip()

        return result


class BenchmarkRunner:
    """Benchmark运行器 - 协调整个评估流程"""

    def __init__(
        self,
        test_suite_dir: str = "benchmark/test_suites",
        output_dir: str = "benchmark/results",
    ):
        self.test_suite_dir = Path(test_suite_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_test_suite(self, category: str) -> List[Dict]:
        """加载测试集"""
        filepath = self.test_suite_dir / f"{category}.json"
        if not filepath.exists():
            return []

        with open(filepath, "r") as f:
            data = json.load(f)
            return data.get("cases", [])

    def run_benchmark(
        self,
        evaluator: LLMEvaluator,
        categories: Optional[List[str]] = None,
        max_cases_per_category: int = 50,
    ) -> BenchmarkScore:
        """运行完整benchmark"""

        categories = categories or ["signal_recognition", "risk_awareness", "consistency"]
        all_results = []

        for category in categories:
            print(f"\n=== Evaluating {category} ===")
            cases = self.load_test_suite(category)

            if not cases:
                print(f"  No test cases found for {category}")
                continue

            cases = cases[:max_cases_per_category]
            print(f"  Running {len(cases)} cases...")

            if category == "consistency":
                results = evaluator.evaluate_consistency(cases)
            elif category == "signal_recognition":
                results = []
                for i, case in enumerate(cases):
                    result = evaluator.evaluate_signal_recognition(case)
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        print(f"    Progress: {i+1}/{len(cases)}")
                    time.sleep(0.5)  # Rate limit
            elif category == "risk_awareness":
                results = []
                for i, case in enumerate(cases):
                    result = evaluator.evaluate_risk_awareness(case)
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        print(f"    Progress: {i+1}/{len(cases)}")
                    time.sleep(0.5)

            all_results.extend(results)

            # 打印分类结果
            passed = sum(1 for r in results if r.passed)
            avg_score = sum(r.score for r in results) / len(results) if results else 0
            print(f"  Results: {passed}/{len(results)} passed, avg score: {avg_score:.2f}")

        # 计算因子分数
        factor_scores = self._calculate_factor_scores(all_results)
        total_score = sum(factor_scores.values()) / len(factor_scores) if factor_scores else 0

        return BenchmarkScore(
            model=evaluator.model,
            provider=evaluator.provider,
            timestamp=datetime.now().isoformat(),
            total_score=total_score * 100,
            factor_scores={k: v * 100 for k, v in factor_scores.items()},
            detailed_results=all_results,
            metadata={
                "total_cases": len(all_results),
                "categories": categories,
            }
        )

    def _calculate_factor_scores(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """计算各因子分数"""
        scores = {}

        for category in ["signal_recognition", "risk_awareness", "consistency"]:
            cat_results = [r for r in results if r.category == category]
            if cat_results:
                scores[category] = sum(r.score for r in cat_results) / len(cat_results)

        return scores

    def save_results(self, score: BenchmarkScore) -> Path:
        """保存评估结果"""
        filename = f"benchmark_{score.provider}_{score.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        # 转换为可序列化格式
        output = score.to_dict()
        output["detailed_results"] = [
            {
                "case_id": r.case_id,
                "category": r.category,
                "passed": r.passed,
                "score": r.score,
                "llm_response": r.llm_response,
                "parsed_answer": r.parsed_answer,
                "ground_truth": r.ground_truth,
                "details": r.details,
            }
            for r in score.detailed_results
        ]

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to {filepath}")
        return filepath


def run_full_benchmark(
    api_key: str,
    provider: str = "openai",
    model: str = "gpt-4-turbo",
    base_url: Optional[str] = None,
) -> BenchmarkScore:
    """便捷函数 - 运行完整benchmark"""

    evaluator = LLMEvaluator(
        api_key=api_key,
        provider=provider,
        model=model,
        base_url=base_url,
    )

    runner = BenchmarkRunner()
    score = runner.run_benchmark(evaluator)
    runner.save_results(score)

    return score


if __name__ == "__main__":
    import sys

    # 需要API key
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    provider = sys.argv[2] if len(sys.argv) > 2 else "openai"
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4-turbo"

    if not api_key:
        print("Usage: python benchmark_runner.py <api_key> [provider] [model]")
        print("Example: python benchmark_runner.py sk-xxx openai gpt-4-turbo")
        print("Example: python benchmark_runner.py sk-xxx deepseek deepseek-chat")
        sys.exit(1)

    score = run_full_benchmark(api_key, provider, model)

    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Model: {score.model}")
    print(f"Total Score: {score.total_score:.1f}/100")
    print("\nFactor Scores:")
    for factor, fscore in score.factor_scores.items():
        print(f"  {factor}: {fscore:.1f}")
