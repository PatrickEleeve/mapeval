"""
历史数据收集器 - 从币安获取多时间周期数据构建测试集

支持的时间周期: 1d, 4h, 1h, 30m, 15m, 5m, 1m
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


class BinanceDataCollector:
    """从币安API收集历史K线数据"""

    BASE_URL = "https://fapi.binance.com"
    BACKUP_URLS = [
        "https://fapi1.binance.com",
        "https://fapi2.binance.com",
    ]

    # 默认交易对
    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    ]

    # 时间周期配置
    INTERVALS = {
        "1m": {"minutes": 1, "limit": 1500},
        "5m": {"minutes": 5, "limit": 1500},
        "15m": {"minutes": 15, "limit": 1500},
        "30m": {"minutes": 30, "limit": 1500},
        "1h": {"minutes": 60, "limit": 1500},
        "4h": {"minutes": 240, "limit": 1500},
        "1d": {"minutes": 1440, "limit": 1000},
    }

    def __init__(self, data_dir: str = "benchmark/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """获取K线数据"""

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        urls = [self.BASE_URL] + self.BACKUP_URLS
        for base_url in urls:
            try:
                resp = requests.get(
                    f"{base_url}/fapi/v1/klines",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                print(f"Failed with {base_url}: {e}")
                continue
        else:
            raise RuntimeError(f"All endpoints failed for {symbol}")

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df["symbol"] = symbol
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str,
        intervals: List[str] = ["1h", "4h", "1d"],
        days_back: int = 365,
    ) -> Dict[str, pd.DataFrame]:
        """获取多个时间周期的数据"""

        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

        result = {}
        for interval in intervals:
            print(f"Fetching {symbol} {interval}...")
            all_data = []
            current_start = start_time

            while current_start < end_time:
                df = self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_time,
                    limit=1500,
                )
                if df.empty:
                    break

                all_data.append(df)
                current_start = int(df["close_time"].iloc[-1].timestamp() * 1000) + 1
                time.sleep(0.1)  # Rate limit

            if all_data:
                result[interval] = pd.concat(all_data, ignore_index=True)
                result[interval] = result[interval].drop_duplicates(subset=["open_time"])

        return result

    def save_dataset(
        self,
        symbols: Optional[List[str]] = None,
        intervals: List[str] = ["1h", "4h", "1d"],
        days_back: int = 365,
        filename: Optional[str] = None,
    ) -> Path:
        """保存完整数据集"""

        symbols = symbols or self.DEFAULT_SYMBOLS

        all_data = {}
        for symbol in symbols:
            print(f"\n=== Processing {symbol} ===")
            try:
                data = self.fetch_multi_timeframe(symbol, intervals, days_back)
                all_data[symbol] = {
                    interval: df.to_dict(orient="records")
                    for interval, df in data.items()
                }
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                continue

        # 保存
        if filename is None:
            filename = f"market_data_{datetime.now().strftime('%Y%m%d')}.json"

        filepath = self.data_dir / filename
        with open(filepath, "w") as f:
            json.dump(all_data, f, default=str, indent=2)

        print(f"\nDataset saved to {filepath}")
        return filepath

    def load_dataset(self, filename: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """加载数据集"""
        filepath = self.data_dir / filename

        with open(filepath, "r") as f:
            raw_data = json.load(f)

        result = {}
        for symbol, intervals in raw_data.items():
            result[symbol] = {}
            for interval, records in intervals.items():
                df = pd.DataFrame(records)
                df["open_time"] = pd.to_datetime(df["open_time"])
                df["close_time"] = pd.to_datetime(df["close_time"])
                result[symbol][interval] = df

        return result


class TestCaseGenerator:
    """从历史数据生成测试用例"""

    def __init__(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        self.data = data

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # 趋势
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        df["trend"] = "neutral"
        df.loc[df["ma_20"] > df["ma_50"], "trend"] = "bullish"
        df.loc[df["ma_20"] < df["ma_50"], "trend"] = "bearish"

        # 波动率状态
        df["volatility_rank"] = df["atr_pct"].rolling(100).rank(pct=True)

        return df

    def generate_signal_recognition_cases(
        self,
        symbol: str,
        interval: str = "1h",
        lookahead_bars: int = 4,  # 预测未来4根K线
        min_move_pct: float = 0.5,  # 至少0.5%的移动才算有效信号
        num_cases: int = 100,
    ) -> List[Dict]:
        """
        生成信号识别测试用例

        核心逻辑：
        1. 找到有明显技术信号的时刻
        2. 记录当时的市场状态作为问题
        3. 用未来N根K线的走势作为答案
        """

        if symbol not in self.data or interval not in self.data[symbol]:
            raise ValueError(f"No data for {symbol} {interval}")

        df = self.data[symbol][interval].copy()
        df = self.calculate_indicators(df)
        df = df.dropna()

        cases = []

        for i in range(50, len(df) - lookahead_bars - 1):
            row = df.iloc[i]

            # 计算未来收益
            future_close = df.iloc[i + lookahead_bars]["close"]
            current_close = row["close"]
            future_return = (future_close - current_close) / current_close * 100

            # 跳过波动太小的点
            if abs(future_return) < min_move_pct:
                continue

            # 确定答案
            if future_return > min_move_pct:
                ground_truth = "LONG"
                difficulty = "easy" if future_return > 2.0 else "medium" if future_return > 1.0 else "hard"
            elif future_return < -min_move_pct:
                ground_truth = "SHORT"
                difficulty = "easy" if future_return < -2.0 else "medium" if future_return < -1.0 else "hard"
            else:
                continue

            # 识别信号类型
            signals = []
            if row["rsi"] < 30:
                signals.append("RSI_OVERSOLD")
            elif row["rsi"] > 70:
                signals.append("RSI_OVERBOUGHT")

            if row["macd_hist"] > 0 and df.iloc[i-1]["macd_hist"] < 0:
                signals.append("MACD_GOLDEN_CROSS")
            elif row["macd_hist"] < 0 and df.iloc[i-1]["macd_hist"] > 0:
                signals.append("MACD_DEATH_CROSS")

            if row["bb_position"] < 0.1:
                signals.append("BB_LOWER_TOUCH")
            elif row["bb_position"] > 0.9:
                signals.append("BB_UPPER_TOUCH")

            # 构建测试用例
            case = {
                "id": f"{symbol}_{interval}_{row['open_time'].strftime('%Y%m%d_%H%M')}",
                "symbol": symbol,
                "interval": interval,
                "timestamp": row["open_time"].isoformat(),

                # 市场状态 (这是给LLM看的)
                "market_context": {
                    "price": round(current_close, 2),
                    "rsi_14": round(row["rsi"], 1),
                    "macd": round(row["macd"], 4),
                    "macd_signal": round(row["macd_signal"], 4),
                    "macd_histogram": round(row["macd_hist"], 4),
                    "atr_pct": round(row["atr_pct"], 2),
                    "bb_position": round(row["bb_position"], 2),
                    "trend": row["trend"],
                    "volatility_rank": round(row["volatility_rank"], 2),
                },
                "detected_signals": signals,

                # 答案 (评估时用)
                "ground_truth": {
                    "direction": ground_truth,
                    "actual_return_pct": round(future_return, 2),
                    "lookahead_bars": lookahead_bars,
                },

                "difficulty": difficulty,
                "category": "signal_recognition",
            }

            cases.append(case)

        # 采样并平衡
        import random
        random.shuffle(cases)

        # 平衡正负样本
        long_cases = [c for c in cases if c["ground_truth"]["direction"] == "LONG"]
        short_cases = [c for c in cases if c["ground_truth"]["direction"] == "SHORT"]

        min_count = min(len(long_cases), len(short_cases), num_cases // 2)
        balanced = long_cases[:min_count] + short_cases[:min_count]
        random.shuffle(balanced)

        return balanced[:num_cases]

    def generate_risk_awareness_cases(
        self,
        symbol: str,
        interval: str = "1h",
        num_cases: int = 50,
    ) -> List[Dict]:
        """
        生成风险意识测试用例

        核心逻辑：
        1. 找到高风险时刻（大跌、高波动、极端RSI等）
        2. 构建场景问LLM应该怎么做
        3. 正确答案是"减仓/止损/不加仓"
        """

        if symbol not in self.data or interval not in self.data[symbol]:
            raise ValueError(f"No data for {symbol} {interval}")

        df = self.data[symbol][interval].copy()
        df = self.calculate_indicators(df)
        df = df.dropna()

        cases = []

        for i in range(50, len(df) - 10):
            row = df.iloc[i]

            # 计算回撤
            recent_high = df.iloc[max(0, i-20):i+1]["high"].max()
            drawdown = (recent_high - row["close"]) / recent_high * 100

            # 找高风险场景
            risk_scenario = None
            correct_actions = []
            wrong_actions = []

            # 场景1: 大幅回撤
            if drawdown > 10:
                risk_scenario = "LARGE_DRAWDOWN"
                correct_actions = ["REDUCE", "CLOSE", "HOLD"]
                wrong_actions = ["INCREASE", "LEVERAGE_UP"]

            # 场景2: RSI极端超买
            elif row["rsi"] > 85:
                risk_scenario = "EXTREME_OVERBOUGHT"
                correct_actions = ["REDUCE", "CLOSE", "SHORT"]
                wrong_actions = ["INCREASE_LONG"]

            # 场景3: 高波动环境
            elif row["volatility_rank"] > 0.9:
                risk_scenario = "HIGH_VOLATILITY"
                correct_actions = ["REDUCE_LEVERAGE", "REDUCE_POSITION", "HOLD"]
                wrong_actions = ["INCREASE_LEVERAGE"]

            # 场景4: 死叉后继续下跌
            elif (row["macd_hist"] < 0 and
                  df.iloc[i-1]["macd_hist"] < 0 and
                  df.iloc[i-2]["macd_hist"] > 0):
                risk_scenario = "CONFIRMED_DOWNTREND"
                correct_actions = ["SHORT", "CLOSE_LONG", "HOLD_SHORT"]
                wrong_actions = ["INCREASE_LONG"]

            if risk_scenario is None:
                continue

            case = {
                "id": f"risk_{symbol}_{row['open_time'].strftime('%Y%m%d_%H%M')}",
                "symbol": symbol,
                "interval": interval,
                "timestamp": row["open_time"].isoformat(),

                "scenario": risk_scenario,
                "market_context": {
                    "price": round(row["close"], 2),
                    "rsi_14": round(row["rsi"], 1),
                    "drawdown_pct": round(drawdown, 2),
                    "atr_pct": round(row["atr_pct"], 2),
                    "volatility_rank": round(row["volatility_rank"], 2),
                    "trend": row["trend"],
                    "macd_histogram": round(row["macd_hist"], 4),
                },

                "question": f"当前{symbol}处于{risk_scenario}状态，你会如何操作？",

                "ground_truth": {
                    "correct_actions": correct_actions,
                    "wrong_actions": wrong_actions,
                },

                "category": "risk_awareness",
            }

            cases.append(case)

        import random
        random.shuffle(cases)
        return cases[:num_cases]

    def generate_consistency_cases(
        self,
        symbol: str,
        interval: str = "1h",
        num_scenarios: int = 20,
        variations_per_scenario: int = 5,
    ) -> List[Dict]:
        """
        生成一致性测试用例

        核心逻辑：
        1. 创建标准场景
        2. 对同一场景略微变化（加噪音）
        3. 期望LLM对相似场景给出一致的决策
        """

        if symbol not in self.data or interval not in self.data[symbol]:
            raise ValueError(f"No data for {symbol} {interval}")

        df = self.data[symbol][interval].copy()
        df = self.calculate_indicators(df)
        df = df.dropna()

        import random

        cases = []
        scenario_id = 0

        # 选取有代表性的时刻
        indices = random.sample(range(50, len(df) - 10), min(num_scenarios, len(df) - 60))

        for i in indices:
            row = df.iloc[i]
            scenario_id += 1

            base_context = {
                "price": round(row["close"], 2),
                "rsi_14": round(row["rsi"], 1),
                "macd_histogram": round(row["macd_hist"], 4),
                "atr_pct": round(row["atr_pct"], 2),
                "trend": row["trend"],
            }

            # 生成变体
            for var in range(variations_per_scenario):
                # 添加小噪音
                noisy_context = base_context.copy()
                noisy_context["rsi_14"] = round(base_context["rsi_14"] + random.uniform(-2, 2), 1)
                noisy_context["macd_histogram"] = round(
                    base_context["macd_histogram"] * random.uniform(0.9, 1.1), 4
                )
                noisy_context["atr_pct"] = round(base_context["atr_pct"] + random.uniform(-0.1, 0.1), 2)

                case = {
                    "id": f"consistency_{scenario_id}_{var}",
                    "scenario_group": f"scenario_{scenario_id}",
                    "variation": var,
                    "symbol": symbol,

                    "market_context": noisy_context,

                    "question": "基于以上市场数据，你建议做多(LONG)、做空(SHORT)还是观望(HOLD)?",

                    "category": "consistency",
                    "note": "同一scenario_group内的用例应该得到一致的方向判断",
                }

                cases.append(case)

        return cases

    def export_test_suite(
        self,
        output_dir: str = "benchmark/test_suites",
        symbols: Optional[List[str]] = None,
        interval: str = "1h",
    ) -> Dict[str, Path]:
        """导出完整测试集"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        symbols = symbols or list(self.data.keys())
        all_cases = {
            "signal_recognition": [],
            "risk_awareness": [],
            "consistency": [],
        }

        for symbol in symbols:
            if symbol not in self.data:
                continue

            print(f"Generating cases for {symbol}...")

            try:
                signal_cases = self.generate_signal_recognition_cases(symbol, interval, num_cases=50)
                all_cases["signal_recognition"].extend(signal_cases)
            except Exception as e:
                print(f"  Signal recognition failed: {e}")

            try:
                risk_cases = self.generate_risk_awareness_cases(symbol, interval, num_cases=30)
                all_cases["risk_awareness"].extend(risk_cases)
            except Exception as e:
                print(f"  Risk awareness failed: {e}")

            try:
                consistency_cases = self.generate_consistency_cases(symbol, interval)
                all_cases["consistency"].extend(consistency_cases)
            except Exception as e:
                print(f"  Consistency failed: {e}")

        # 保存
        saved_files = {}
        for category, cases in all_cases.items():
            if not cases:
                continue

            filepath = output_path / f"{category}.json"
            with open(filepath, "w") as f:
                json.dump({
                    "category": category,
                    "generated_at": datetime.now().isoformat(),
                    "total_cases": len(cases),
                    "cases": cases,
                }, f, indent=2, default=str)

            saved_files[category] = filepath
            print(f"Saved {len(cases)} {category} cases to {filepath}")

        return saved_files


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        # 收集数据
        collector = BinanceDataCollector()
        symbols = sys.argv[2:] if len(sys.argv) > 2 else None
        collector.save_dataset(
            symbols=symbols,
            intervals=["1h", "4h", "1d"],
            days_back=180,
        )

    elif len(sys.argv) > 1 and sys.argv[1] == "generate":
        # 生成测试用例
        collector = BinanceDataCollector()
        data_file = sys.argv[2] if len(sys.argv) > 2 else None

        if data_file:
            data = collector.load_dataset(data_file)
        else:
            # 找最新的数据文件
            data_files = list(Path("benchmark/data").glob("market_data_*.json"))
            if not data_files:
                print("No data files found. Run 'collect' first.")
                sys.exit(1)
            data = collector.load_dataset(data_files[-1].name)

        generator = TestCaseGenerator(data)
        generator.export_test_suite()

    else:
        print("Usage:")
        print("  python data_collector.py collect [symbols...]  - Collect market data")
        print("  python data_collector.py generate [data_file]  - Generate test cases")
