"""
因子化评估模块 - MAPEval Factorized Benchmark Framework

将LLM交易能力拆解为多个独立可度量的因子维度：
1. 信号因子 (Signal Factor) - 预测方向的准确性
2. 风险因子 (Risk Factor) - 风险控制能力
3. 效率因子 (Efficiency Factor) - 成本效率
4. 一致性因子 (Consistency Factor) - 决策稳定性
5. 适应性因子 (Adaptability Factor) - 市场环境适应能力
6. 推理因子 (Reasoning Factor) - 推理质量评估
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class FactorScore:
    """单个因子的评分结果"""
    name: str
    score: float  # 0-100 标准化评分
    weight: float  # 因子权重
    details: Dict[str, Any] = field(default_factory=dict)
    sub_factors: List["FactorScore"] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class BenchmarkResult:
    """完整的Benchmark评估结果"""
    session_id: str
    provider: str
    total_score: float
    factors: List[FactorScore]
    market_regime: str  # bull/bear/sideways/volatile
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "total_score": round(self.total_score, 2),
            "market_regime": self.market_regime,
            "factors": [
                {
                    "name": f.name,
                    "score": round(f.score, 2),
                    "weight": f.weight,
                    "weighted_score": round(f.weighted_score, 2),
                    "details": f.details,
                    "sub_factors": [
                        {"name": sf.name, "score": round(sf.score, 2)}
                        for sf in f.sub_factors
                    ]
                }
                for f in self.factors
            ],
            "metadata": self.metadata
        }

    def to_leaderboard_row(self) -> Dict[str, Any]:
        """生成排行榜单行数据"""
        row = {
            "provider": self.provider,
            "total_score": round(self.total_score, 2),
            "market_regime": self.market_regime,
        }
        for f in self.factors:
            row[f.name] = round(f.score, 2)
        return row


class FactorEvaluator:
    """因子化评估器 - 核心评估引擎"""

    # 因子权重配置 (总和=1.0)
    DEFAULT_WEIGHTS = {
        "signal": 0.25,      # 信号准确性
        "risk": 0.20,        # 风险控制
        "efficiency": 0.15,  # 成本效率
        "consistency": 0.15, # 决策一致性
        "adaptability": 0.15,# 市场适应性
        "reasoning": 0.10,   # 推理质量
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        # 标准化权重
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def evaluate_session(
        self,
        session_data: Dict[str, Any],
        price_data: Optional[pd.DataFrame] = None,
    ) -> BenchmarkResult:
        """评估单个交易session"""

        metadata = session_data.get("metadata", {})
        parameters = session_data.get("parameters", {})
        summary = session_data.get("summary", {})

        equity_history = summary.get("equity_history", [])
        trade_log = summary.get("trade_log", [])
        decision_log = summary.get("decision_log", [])

        # 检测市场状态
        market_regime = self._detect_market_regime(equity_history, price_data)

        # 计算各因子得分
        factors = []

        # 1. 信号因子
        signal_score = self._evaluate_signal_factor(
            decision_log, trade_log, equity_history, price_data
        )
        factors.append(signal_score)

        # 2. 风险因子
        risk_score = self._evaluate_risk_factor(
            equity_history, trade_log, parameters
        )
        factors.append(risk_score)

        # 3. 效率因子
        efficiency_score = self._evaluate_efficiency_factor(
            trade_log, equity_history, parameters
        )
        factors.append(efficiency_score)

        # 4. 一致性因子
        consistency_score = self._evaluate_consistency_factor(
            decision_log, trade_log
        )
        factors.append(consistency_score)

        # 5. 适应性因子
        adaptability_score = self._evaluate_adaptability_factor(
            decision_log, equity_history, market_regime
        )
        factors.append(adaptability_score)

        # 6. 推理因子
        reasoning_score = self._evaluate_reasoning_factor(decision_log)
        factors.append(reasoning_score)

        # 计算总分
        total_score = sum(f.weighted_score for f in factors)

        return BenchmarkResult(
            session_id=metadata.get("session_id", "unknown"),
            provider=metadata.get("llm_provider", "unknown"),
            total_score=total_score,
            factors=factors,
            market_regime=market_regime,
            metadata={
                "duration_seconds": parameters.get("duration_seconds"),
                "symbols_count": len(parameters.get("symbols", [])),
                "initial_capital": parameters.get("initial_capital"),
                "max_leverage": parameters.get("max_leverage"),
            }
        )

    def _detect_market_regime(
        self,
        equity_history: List[Dict],
        price_data: Optional[pd.DataFrame] = None
    ) -> str:
        """检测市场状态: bull/bear/sideways/volatile"""
        if not equity_history or len(equity_history) < 10:
            return "unknown"

        df = pd.DataFrame(equity_history)
        if "equity" not in df.columns:
            return "unknown"

        equity = df["equity"]
        returns = equity.pct_change().dropna()

        if len(returns) < 5:
            return "unknown"

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        volatility = returns.std() * math.sqrt(len(returns))

        # 基于收益和波动率判断市场状态
        if volatility > 0.1:  # 高波动
            return "volatile"
        elif total_return > 0.02:  # 上涨>2%
            return "bull"
        elif total_return < -0.02:  # 下跌>2%
            return "bear"
        else:
            return "sideways"

    def _evaluate_signal_factor(
        self,
        decision_log: List[Dict],
        trade_log: List[Dict],
        equity_history: List[Dict],
        price_data: Optional[pd.DataFrame] = None
    ) -> FactorScore:
        """
        评估信号因子 - 预测方向的准确性

        子因子:
        - direction_accuracy: 方向判断正确率
        - timing_quality: 入场时机质量
        - exit_quality: 出场时机质量
        """
        sub_factors = []

        # 1. 方向准确率 - 基于交易盈亏
        winning_trades = [t for t in trade_log if t.get("realized_pnl", 0) > 0]
        total_trades = len([t for t in trade_log if t.get("action") in ["close", "reduce", "reverse_close"]])

        direction_accuracy = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 50
        sub_factors.append(FactorScore(
            name="direction_accuracy",
            score=direction_accuracy,
            weight=0.5,
            details={"winning": len(winning_trades), "total": total_trades}
        ))

        # 2. 信号质量 - 基于决策置信度与结果的相关性
        confident_correct = 0
        confident_total = 0
        for decision in decision_log:
            if decision.get("action") == "REBALANCE":
                confident_total += 1
                # 检查该决策后的收益
                # 简化处理：如果applied_exposure方向与后续equity变化一致则算正确

        signal_quality = 50.0  # 默认中等
        if confident_total > 0:
            # 基于equity变化趋势评估
            if equity_history and len(equity_history) > 1:
                start_eq = equity_history[0].get("equity", 100000)
                end_eq = equity_history[-1].get("equity", 100000)
                if end_eq > start_eq:
                    signal_quality = min(80, 50 + (end_eq / start_eq - 1) * 500)
                else:
                    signal_quality = max(20, 50 + (end_eq / start_eq - 1) * 500)

        sub_factors.append(FactorScore(
            name="signal_quality",
            score=signal_quality,
            weight=0.5,
            details={"decisions_count": confident_total}
        ))

        # 综合信号因子得分
        total_score = sum(sf.score * sf.weight for sf in sub_factors)

        return FactorScore(
            name="signal",
            score=total_score,
            weight=self.weights["signal"],
            sub_factors=sub_factors,
            details={
                "win_rate": direction_accuracy,
                "total_decisions": len(decision_log),
            }
        )

    def _evaluate_risk_factor(
        self,
        equity_history: List[Dict],
        trade_log: List[Dict],
        parameters: Dict[str, Any]
    ) -> FactorScore:
        """
        评估风险因子 - 风险控制能力

        子因子:
        - max_drawdown_score: 最大回撤控制
        - leverage_discipline: 杠杆纪律
        - position_sizing: 仓位管理
        """
        sub_factors = []

        # 1. 最大回撤控制 (回撤越小分数越高)
        max_drawdown = 0.0
        if equity_history:
            df = pd.DataFrame(equity_history)
            if "equity" in df.columns:
                equity = df["equity"]
                running_max = equity.cummax()
                drawdown = (equity - running_max) / running_max
                max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0

        # 回撤<5%得100分，>50%得0分，线性插值
        drawdown_score = max(0, 100 - max_drawdown * 200)
        sub_factors.append(FactorScore(
            name="max_drawdown_score",
            score=drawdown_score,
            weight=0.4,
            details={"max_drawdown_pct": round(max_drawdown * 100, 2)}
        ))

        # 2. 杠杆纪律 - 是否超出限制
        max_leverage = parameters.get("max_leverage", 30)
        leverage_violations = 0
        for trade in trade_log:
            # 检查是否有engine_notes说明超出限制
            pass  # 需要从decision_log检查

        leverage_score = 100 - leverage_violations * 10
        sub_factors.append(FactorScore(
            name="leverage_discipline",
            score=max(0, leverage_score),
            weight=0.3,
            details={"violations": leverage_violations}
        ))

        # 3. 仓位管理 - 基于波动率调整仓位
        # 评估是否在高波动时降低仓位
        position_score = 70  # 默认中等偏上
        sub_factors.append(FactorScore(
            name="position_sizing",
            score=position_score,
            weight=0.3,
            details={}
        ))

        total_score = sum(sf.score * sf.weight for sf in sub_factors)

        return FactorScore(
            name="risk",
            score=total_score,
            weight=self.weights["risk"],
            sub_factors=sub_factors,
            details={
                "max_drawdown_pct": round(max_drawdown * 100, 2),
            }
        )

    def _evaluate_efficiency_factor(
        self,
        trade_log: List[Dict],
        equity_history: List[Dict],
        parameters: Dict[str, Any]
    ) -> FactorScore:
        """
        评估效率因子 - 成本效率

        子因子:
        - cost_efficiency: 每单位收益的成本
        - turnover_efficiency: 换手率效率
        - holding_period: 持仓周期合理性
        """
        sub_factors = []

        # 1. 成本效率
        total_commission = sum(t.get("commission", 0) for t in trade_log)
        total_slippage = sum(t.get("slippage_cost", 0) for t in trade_log)
        total_costs = total_commission + total_slippage

        total_pnl = sum(t.get("realized_pnl", 0) for t in trade_log)

        # 成本/毛收益比，越低越好
        if total_pnl > 0:
            cost_ratio = total_costs / (total_pnl + total_costs)
            cost_score = max(0, 100 - cost_ratio * 200)
        elif total_pnl < 0:
            cost_score = max(0, 30 - abs(total_pnl) / 1000)
        else:
            cost_score = 50

        sub_factors.append(FactorScore(
            name="cost_efficiency",
            score=cost_score,
            weight=0.4,
            details={
                "total_costs": round(total_costs, 2),
                "total_pnl": round(total_pnl, 2),
            }
        ))

        # 2. 换手率效率
        num_trades = len(trade_log)
        duration_hours = parameters.get("duration_seconds", 3600) / 3600
        trades_per_hour = num_trades / max(duration_hours, 0.1)

        # 理想换手率: 1-5次/小时，过多或过少都扣分
        if 1 <= trades_per_hour <= 5:
            turnover_score = 100
        elif trades_per_hour < 1:
            turnover_score = 70 + trades_per_hour * 30
        else:
            turnover_score = max(30, 100 - (trades_per_hour - 5) * 10)

        sub_factors.append(FactorScore(
            name="turnover_efficiency",
            score=turnover_score,
            weight=0.3,
            details={"trades_per_hour": round(trades_per_hour, 2)}
        ))

        # 3. 持仓周期
        hold_score = 70  # 默认
        sub_factors.append(FactorScore(
            name="holding_period",
            score=hold_score,
            weight=0.3,
        ))

        total_score = sum(sf.score * sf.weight for sf in sub_factors)

        return FactorScore(
            name="efficiency",
            score=total_score,
            weight=self.weights["efficiency"],
            sub_factors=sub_factors,
            details={
                "total_trades": num_trades,
                "total_costs": round(total_costs, 2),
            }
        )

    def _evaluate_consistency_factor(
        self,
        decision_log: List[Dict],
        trade_log: List[Dict]
    ) -> FactorScore:
        """
        评估一致性因子 - 决策稳定性

        子因子:
        - action_consistency: 动作一致性 (HOLD vs REBALANCE比例)
        - exposure_stability: 仓位稳定性
        - reasoning_consistency: 推理一致性
        """
        sub_factors = []

        # 1. 动作一致性 - HOLD比例 (HOLD多说明不频繁交易)
        hold_count = sum(1 for d in decision_log if d.get("action") == "HOLD")
        total_decisions = len(decision_log)

        hold_ratio = hold_count / total_decisions if total_decisions > 0 else 0.5
        # HOLD比例在30%-80%之间最佳
        if 0.3 <= hold_ratio <= 0.8:
            action_score = 100
        elif hold_ratio < 0.3:
            action_score = 50 + hold_ratio * 100
        else:
            action_score = 100 - (hold_ratio - 0.8) * 200

        sub_factors.append(FactorScore(
            name="action_consistency",
            score=max(0, action_score),
            weight=0.4,
            details={"hold_ratio": round(hold_ratio, 2)}
        ))

        # 2. 仓位稳定性 - 连续决策间仓位变化的标准差
        exposure_changes = []
        for i in range(1, len(decision_log)):
            prev = decision_log[i-1].get("applied_exposure", {})
            curr = decision_log[i].get("applied_exposure", {})
            if prev and curr:
                total_change = sum(
                    abs(curr.get(k, 0) - prev.get(k, 0))
                    for k in set(prev.keys()) | set(curr.keys())
                )
                exposure_changes.append(total_change)

        if exposure_changes:
            avg_change = sum(exposure_changes) / len(exposure_changes)
            # 平均变化<5最佳
            stability_score = max(0, 100 - avg_change * 10)
        else:
            stability_score = 70

        sub_factors.append(FactorScore(
            name="exposure_stability",
            score=stability_score,
            weight=0.3,
            details={"avg_exposure_change": round(avg_change if exposure_changes else 0, 2)}
        ))

        # 3. 推理一致性 - 检查推理是否前后矛盾
        reasoning_score = 70  # 默认，需要NLP分析
        sub_factors.append(FactorScore(
            name="reasoning_consistency",
            score=reasoning_score,
            weight=0.3,
        ))

        total_score = sum(sf.score * sf.weight for sf in sub_factors)

        return FactorScore(
            name="consistency",
            score=total_score,
            weight=self.weights["consistency"],
            sub_factors=sub_factors,
            details={
                "total_decisions": total_decisions,
                "hold_count": hold_count,
            }
        )

    def _evaluate_adaptability_factor(
        self,
        decision_log: List[Dict],
        equity_history: List[Dict],
        market_regime: str
    ) -> FactorScore:
        """
        评估适应性因子 - 市场环境适应能力

        子因子:
        - regime_adaptation: 市场状态适应
        - volatility_response: 波动率响应
        - trend_following: 趋势跟随能力
        """
        sub_factors = []

        # 1. 市场状态适应 - 根据市场状态调整策略
        # 牛市做多、熊市做空/减仓的能力
        adaptation_score = 60  # 默认

        if equity_history and len(equity_history) > 1:
            start_eq = equity_history[0].get("equity", 100000)
            end_eq = equity_history[-1].get("equity", 100000)
            pnl_pct = (end_eq / start_eq - 1) * 100

            if market_regime == "bull" and pnl_pct > 0:
                adaptation_score = min(100, 60 + pnl_pct * 5)
            elif market_regime == "bear" and pnl_pct > -5:
                adaptation_score = min(100, 70 + (5 + pnl_pct) * 3)
            elif market_regime == "sideways" and abs(pnl_pct) < 2:
                adaptation_score = 80
            elif market_regime == "volatile":
                # 高波动时控制损失
                if pnl_pct > -10:
                    adaptation_score = 70 + (10 + pnl_pct)

        sub_factors.append(FactorScore(
            name="regime_adaptation",
            score=adaptation_score,
            weight=0.4,
            details={"market_regime": market_regime}
        ))

        # 2. 波动率响应
        vol_response_score = 65
        sub_factors.append(FactorScore(
            name="volatility_response",
            score=vol_response_score,
            weight=0.3,
        ))

        # 3. 趋势跟随
        trend_score = 65
        sub_factors.append(FactorScore(
            name="trend_following",
            score=trend_score,
            weight=0.3,
        ))

        total_score = sum(sf.score * sf.weight for sf in sub_factors)

        return FactorScore(
            name="adaptability",
            score=total_score,
            weight=self.weights["adaptability"],
            sub_factors=sub_factors,
            details={"market_regime": market_regime}
        )

    def _evaluate_reasoning_factor(
        self,
        decision_log: List[Dict]
    ) -> FactorScore:
        """
        评估推理因子 - 推理质量

        子因子:
        - reasoning_completeness: 推理完整性
        - indicator_usage: 指标使用合理性
        - logic_coherence: 逻辑连贯性
        """
        sub_factors = []

        # 1. 推理完整性 - 检查reasoning字段长度和内容
        reasoning_lengths = []
        has_indicator_mention = 0

        for decision in decision_log:
            reasoning = decision.get("reasoning", "")
            if reasoning:
                reasoning_lengths.append(len(reasoning))
                # 检查是否提及技术指标
                indicators = ["rsi", "macd", "atr", "bollinger", "trend", "momentum"]
                if any(ind in reasoning.lower() for ind in indicators):
                    has_indicator_mention += 1

        avg_length = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
        # 理想长度: 50-200字符
        if 50 <= avg_length <= 200:
            completeness_score = 100
        elif avg_length < 50:
            completeness_score = avg_length * 2
        else:
            completeness_score = max(50, 100 - (avg_length - 200) * 0.1)

        sub_factors.append(FactorScore(
            name="reasoning_completeness",
            score=completeness_score,
            weight=0.4,
            details={"avg_reasoning_length": round(avg_length, 1)}
        ))

        # 2. 指标使用合理性
        indicator_ratio = has_indicator_mention / len(decision_log) if decision_log else 0
        indicator_score = indicator_ratio * 100

        sub_factors.append(FactorScore(
            name="indicator_usage",
            score=indicator_score,
            weight=0.3,
            details={"indicator_mention_ratio": round(indicator_ratio, 2)}
        ))

        # 3. 逻辑连贯性 (需要更复杂的NLP，暂用默认值)
        coherence_score = 65
        sub_factors.append(FactorScore(
            name="logic_coherence",
            score=coherence_score,
            weight=0.3,
        ))

        total_score = sum(sf.score * sf.weight for sf in sub_factors)

        return FactorScore(
            name="reasoning",
            score=total_score,
            weight=self.weights["reasoning"],
            sub_factors=sub_factors,
            details={
                "total_reasonings": len(reasoning_lengths),
            }
        )


class ScenarioTestSuite:
    """
    场景化测试集 - 包含预定义的市场场景

    用于在不同市场条件下系统性测试LLM
    """

    SCENARIOS = {
        "bull_strong": {
            "name": "强势牛市",
            "description": "价格持续上涨，波动率低",
            "expected_behavior": "积极做多，保持仓位",
            "weight": 0.2,
        },
        "bear_crash": {
            "name": "急跌熊市",
            "description": "价格快速下跌，波动率高",
            "expected_behavior": "快速止损或做空",
            "weight": 0.2,
        },
        "sideways": {
            "name": "横盘震荡",
            "description": "价格区间波动，无明显趋势",
            "expected_behavior": "减少交易，小仓位波段",
            "weight": 0.2,
        },
        "high_volatility": {
            "name": "高波动市场",
            "description": "剧烈波动，方向不明",
            "expected_behavior": "降低杠杆，严格止损",
            "weight": 0.2,
        },
        "trend_reversal": {
            "name": "趋势反转",
            "description": "从牛转熊或从熊转牛",
            "expected_behavior": "识别反转信号，及时调仓",
            "weight": 0.2,
        },
    }

    @classmethod
    def generate_scenario_data(cls, scenario_name: str, duration_hours: float = 1.0) -> pd.DataFrame:
        """生成指定场景的模拟价格数据"""
        import numpy as np

        if scenario_name not in cls.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        np.random.seed(42)  # 确保可复现

        n_points = int(duration_hours * 60)  # 1分钟一个点
        timestamps = pd.date_range(start="2025-01-01", periods=n_points, freq="1min")

        base_price = 100000  # 基准价格 (如BTC)

        if scenario_name == "bull_strong":
            # 稳步上涨 +5%
            trend = np.linspace(0, 0.05, n_points)
            noise = np.random.normal(0, 0.002, n_points)
            prices = base_price * (1 + trend + noise)

        elif scenario_name == "bear_crash":
            # 急跌 -15%
            trend = np.linspace(0, -0.15, n_points)
            noise = np.random.normal(0, 0.005, n_points)
            prices = base_price * (1 + trend + noise)

        elif scenario_name == "sideways":
            # 横盘 ±2%
            noise = np.random.normal(0, 0.003, n_points)
            mean_revert = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.01
            prices = base_price * (1 + mean_revert + noise)

        elif scenario_name == "high_volatility":
            # 高波动
            noise = np.random.normal(0, 0.015, n_points)
            prices = base_price * (1 + noise.cumsum() * 0.1)

        elif scenario_name == "trend_reversal":
            # 先涨后跌
            half = n_points // 2
            up_trend = np.linspace(0, 0.05, half)
            down_trend = np.linspace(0.05, -0.03, n_points - half)
            trend = np.concatenate([up_trend, down_trend])
            noise = np.random.normal(0, 0.003, n_points)
            prices = base_price * (1 + trend + noise)
        else:
            prices = np.full(n_points, base_price)

        return pd.DataFrame({
            "timestamp": timestamps,
            "BTCUSDT": prices,
        })


class BenchmarkReport:
    """生成标准化的Benchmark报告"""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results

    def generate_leaderboard(self) -> pd.DataFrame:
        """生成排行榜"""
        rows = [r.to_leaderboard_row() for r in self.results]
        df = pd.DataFrame(rows)
        return df.sort_values("total_score", ascending=False).reset_index(drop=True)

    def generate_radar_chart_data(self, result: BenchmarkResult) -> Dict[str, float]:
        """生成雷达图数据"""
        return {f.name: f.score for f in result.factors}

    def generate_comparison_report(self) -> Dict[str, Any]:
        """生成对比报告"""
        if not self.results:
            return {}

        # 按provider分组
        by_provider = {}
        for r in self.results:
            if r.provider not in by_provider:
                by_provider[r.provider] = []
            by_provider[r.provider].append(r)

        comparison = {}
        for provider, results in by_provider.items():
            avg_total = sum(r.total_score for r in results) / len(results)
            factor_avgs = {}
            for factor_name in self.results[0].factors[0].name if self.results else []:
                factor_avgs[factor_name] = sum(
                    next((f.score for f in r.factors if f.name == factor_name), 0)
                    for r in results
                ) / len(results)

            comparison[provider] = {
                "avg_total_score": round(avg_total, 2),
                "sessions_count": len(results),
                "factor_averages": factor_avgs,
            }

        return comparison

    def export_json(self, filepath: str) -> None:
        """导出JSON格式报告"""
        report = {
            "leaderboard": self.generate_leaderboard().to_dict(orient="records"),
            "comparison": self.generate_comparison_report(),
            "detailed_results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def export_markdown(self, filepath: str) -> None:
        """导出Markdown格式报告"""
        lines = ["# MAPEval Benchmark Report\n"]

        # 排行榜
        lines.append("## Leaderboard\n")
        df = self.generate_leaderboard()
        # 手动生成markdown表格，避免依赖tabulate
        if not df.empty:
            headers = "| " + " | ".join(df.columns) + " |"
            separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            lines.append(headers)
            lines.append(separator)
            for _, row in df.iterrows():
                row_str = "| " + " | ".join(str(v) for v in row.values) + " |"
                lines.append(row_str)
        lines.append("\n")

        # 因子详情
        lines.append("## Factor Scores by Provider\n")
        for result in self.results:
            lines.append(f"### {result.provider} ({result.session_id})\n")
            lines.append(f"- **Total Score**: {result.total_score:.2f}")
            lines.append(f"- **Market Regime**: {result.market_regime}\n")
            lines.append("| Factor | Score | Weight | Weighted |")
            lines.append("|--------|-------|--------|----------|")
            for f in result.factors:
                lines.append(f"| {f.name} | {f.score:.1f} | {f.weight:.2f} | {f.weighted_score:.2f} |")
            lines.append("\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# 便捷函数
def evaluate_session_file(filepath: str) -> BenchmarkResult:
    """从JSON文件评估session"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluator = FactorEvaluator()
    return evaluator.evaluate_session(data)


def evaluate_all_sessions(log_dir: str = "logs") -> List[BenchmarkResult]:
    """评估目录下所有session"""
    log_path = Path(log_dir)
    results = []

    for json_file in log_path.glob("session_*.json"):
        if "_llm_decisions" in json_file.name:
            continue  # 跳过决策日志文件
        try:
            result = evaluate_session_file(str(json_file))
            results.append(result)
        except Exception as e:
            print(f"Failed to evaluate {json_file}: {e}")

    return results


def generate_benchmark_report(log_dir: str = "logs", output_dir: str = "reports") -> str:
    """生成完整的benchmark报告"""
    results = evaluate_all_sessions(log_dir)

    if not results:
        return "No sessions found to evaluate."

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = BenchmarkReport(results)

    # 导出报告
    json_path = output_path / "benchmark_report.json"
    md_path = output_path / "benchmark_report.md"

    report.export_json(str(json_path))
    report.export_markdown(str(md_path))

    return f"Reports generated:\n- {json_path}\n- {md_path}"


if __name__ == "__main__":
    # CLI入口
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "evaluate":
            log_dir = sys.argv[2] if len(sys.argv) > 2 else "logs"
            output = generate_benchmark_report(log_dir)
            print(output)
        elif sys.argv[1] == "single":
            filepath = sys.argv[2]
            result = evaluate_session_file(filepath)
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Usage:")
        print("  python factor_evaluator.py evaluate [log_dir]  - Evaluate all sessions")
        print("  python factor_evaluator.py single <file.json> - Evaluate single session")
