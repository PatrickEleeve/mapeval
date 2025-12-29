"""Entry point for real-time leveraged futures trading."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from config import AGENT_CONFIG, DEEPSEEK_API_KEY, OPENAI_API_KEY, QWEN_API_KEY, TRADING_CONFIG
from data_manager import RealTimeMarketData, BacktestMarketData, load_historical_data
from llm_agent import LLMAgent, BaselineAgent
from log_manager import SessionLogger
from reporter import RealTimeReporter
from tui_reporter import TUIReporter
from trading_engine import RealTimeTradingEngine


AVAILABLE_INDICATORS = (
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "coefficient_of_variation",
    "ma_slope",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-time leveraged trading session.")
    parser.add_argument(
        "--mode",
        choices=["realtime", "backtest"],
        default="realtime",
        help="Execution mode: 'realtime' connects to live API, 'backtest' replays history.",
    )
    parser.add_argument(
        "--strategy",
        choices=["llm", "buy_hold", "ma_crossover", "random"],
        default="llm",
        help="Strategy to run.",
    )
    parser.add_argument(
        "--duration",
        choices=TRADING_CONFIG["duration_seconds"].keys(),
        default="1h",
        help="Trading session length.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TRADING_CONFIG["symbols"],
        help="Symbols to trade (Binance perpetual symbols).",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=TRADING_CONFIG["poll_interval_seconds"],
        help="Seconds between market price polls.",
    )
    parser.add_argument(
        "--decision",
        type=float,
        default=TRADING_CONFIG["decision_interval_seconds"],
        help="Seconds between strategy decisions.",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=TRADING_CONFIG["max_leverage"],
        help="Maximum aggregate leverage allowed.",
    )
    parser.add_argument(
        "--per-symbol-max-exposure",
        type=float,
        default=TRADING_CONFIG.get("per_symbol_max_exposure", TRADING_CONFIG["max_leverage"]),
        help="Maximum absolute leverage allowed on any single symbol.",
    )
    parser.add_argument(
        "--max-exposure-delta",
        type=float,
        default=TRADING_CONFIG.get("max_exposure_delta", TRADING_CONFIG["max_leverage"]),
        help="Maximum per-symbol change in leverage between consecutive decisions.",
    )
    parser.add_argument(
        "--history-interval",
        default=TRADING_CONFIG["history_interval"],
        help="Kline interval used to bootstrap history (e.g., 1m, 5m).",
    )
    parser.add_argument(
        "--history-lookback",
        type=int,
        default=TRADING_CONFIG["history_lookback"],
        help="Number of historical bars to seed the data buffer with.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=60.0,
        help="Seconds between live console summaries.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where session logs should be written.",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=list(AVAILABLE_INDICATORS) + ["all", "none"],
        help="Technical indicators to expose to the LLM (use 'all' for every indicator).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=sorted(AGENT_CONFIG.keys()),
        default="openai",
        help="Which LLM provider to use for generating trading signals.",
    )
    parser.add_argument(
        "--min-long-exposure",
        type=float,
        default=TRADING_CONFIG.get("min_long_exposure", 0.0),
        help="Minimum positive leverage exposure (fraction of equity) required for long trades.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=TRADING_CONFIG["initial_capital"],
        help="Initial account equity in USDT.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=TRADING_CONFIG.get("commission_rate", 0.0),
        help="Commission rate (e.g. 0.0005 for 0.05%).",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=TRADING_CONFIG.get("slippage", 0.0),
        help="Slippage rate (e.g. 0.0005 for 0.05%).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to CSV file for loading/saving backtest data (e.g. data/benchmark.csv).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Enable TUI (Text User Interface) dashboard.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and run with command-line arguments only.",
    )
    parser.add_argument(
        "--gross-leverage-cap",
        type=float,
        default=TRADING_CONFIG.get("gross_leverage_cap", 3.0),
        help="Maximum sum of absolute exposures (gross leverage cap).",
    )
    parser.add_argument(
        "--net-exposure-cap",
        type=float,
        default=TRADING_CONFIG.get("net_exposure_cap", 1.0),
        help="Maximum absolute net exposure (long - short).",
    )
    parser.add_argument(
        "--max-open-positions",
        type=int,
        default=TRADING_CONFIG.get("max_open_positions", 5),
        help="Maximum number of simultaneous open positions.",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=TRADING_CONFIG.get("max_turnover_per_step", 2.0),
        help="Maximum sum of exposure changes per step.",
    )
    return parser.parse_args()


def _select_duration(duration_label: str) -> float:
    durations: Dict[str, int] = TRADING_CONFIG["duration_seconds"]
    return float(durations.get(duration_label, durations["1h"]))


def _normalize_indicators(raw: Sequence[str] | None) -> List[str]:
    if raw is None:
        return []
    normalized: List[str] = []
    seen = set()
    tokens = [token.lower() for token in raw]
    if "all" in tokens:
        return list(AVAILABLE_INDICATORS)
    for token in tokens:
        if token in ("none", ""):
            continue
        if token not in AVAILABLE_INDICATORS:
            continue
        if token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return normalized


def _prompt_menu(title: str, options: Sequence[str], default_value: str) -> str:
    indexed = list(options)
    if not indexed:
        raise ValueError("No options provided for selection.")
    try:
        default_index = indexed.index(default_value) + 1
    except ValueError:
        default_index = 1

    while True:
        print(title)
        for idx, option in enumerate(indexed, start=1):
            print(f"  {idx}) {option}")
        raw = input(f"Choose option [{default_index}]: ").strip()
        if not raw:
            return indexed[default_index - 1]
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(indexed):
                return indexed[selected - 1]
        print("Invalid choice. Try again.")


def _prompt_float(prompt: str, default_value: float) -> float:
    while True:
        raw = input(f"{prompt} [{default_value}]: ").strip()
        if not raw:
            return default_value
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive number.")


def _prompt_indicators(default: Sequence[str] | None) -> List[str]:
    default_list = _normalize_indicators(default)
    default_display = ", ".join(default_list) if default_list else "none"
    print("Available technical indicators:")
    for name in AVAILABLE_INDICATORS:
        print(f"  - {name}")
    raw = input(f"Select indicators (comma or space separated) [{default_display}]: ").strip()
    if not raw:
        return default_list
    tokens = [token for token in raw.replace(",", " ").split(" ") if token]
    return _normalize_indicators(tokens)


def _api_key_for_provider(provider: str) -> str:
    if provider == "deepseek":
        return DEEPSEEK_API_KEY
    if provider == "qwen":
        return QWEN_API_KEY
    return OPENAI_API_KEY


def _run_trading_session(
    *,
    provider: str,
    symbols: Sequence[str],
    duration_label: str,
    poll: float,
    decision: float,
    max_leverage: float,
    per_symbol_max_exposure: float,
    max_exposure_delta: float,
    history_interval: str,
    history_lookback: int,
    print_interval: float,
    log_dir: str,
    min_long_exposure: float,
    initial_capital: float,
    indicators: Sequence[str] | None = None,
    mode: str = "realtime",
    strategy: str = "llm",
    commission: float = 0.0,
    slippage: float = 0.0,
    data_path: str | None = None,
    use_ui: bool = False,
    gross_leverage_cap: float = 3.0,
    net_exposure_cap: float = 1.0,
    max_open_positions: int = 5,
    max_turnover: float = 2.0,
) -> None:
    provider_key = provider.lower()
    provider_config: Dict[str, float] = AGENT_CONFIG.get(provider_key, {})
    duration_seconds = _select_duration(duration_label)
    uppercase_symbols: List[str] = [symbol.upper() for symbol in symbols]
    selected_indicators = _normalize_indicators(indicators)

    print("=== Real-Time Leveraged Trading ===")
    print(f"Mode: {mode} | Strategy: {strategy}")
    print(f"Provider: {provider_key}")
    print(f"Symbols: {', '.join(uppercase_symbols)}")
    print(f"Duration: {duration_label} ({duration_seconds:.0f} seconds)")
    print(f"Initial capital: {initial_capital:.2f} USDT")
    print(f"Polling interval: {poll} s | Decision interval: {decision} s")
    print(f"Maximum leverage: {max_leverage}x")
    print(f"Per-symbol cap: {per_symbol_max_exposure}x | Max delta/step: {max_exposure_delta}x")
    print(f"Commission: {commission*100:.4f}% | Slippage: {slippage*100:.4f}%")
    print(f"[RISK] Gross leverage cap: {gross_leverage_cap}x | Net exposure cap: ±{net_exposure_cap}x")
    print(f"[RISK] Max positions: {max_open_positions} | Max turnover/step: {max_turnover}x")
    if min_long_exposure > 0:
        print(f"Minimum long exposure: {min_long_exposure:.4f}x of equity")
    if selected_indicators:
        print(f"Indicators: {', '.join(selected_indicators)}")
    else:
        print("Indicators: none")

    if mode == "backtest":
        print("Loading historical data for backtest...")
        # Load enough data for lookback + simulation
        # Assuming 1m interval, 2000 rows covers > 1 day
        historical_df = load_historical_data(
            uppercase_symbols, 
            history_interval, 
            2000,
            cache_path=data_path
        )
        market_data = BacktestMarketData(
            historical_df,
            uppercase_symbols,
            interval=history_interval,
            lookback=history_lookback
        )
    else:
        market_data = RealTimeMarketData(
            symbols=uppercase_symbols,
            interval=history_interval,
            lookback=history_lookback,
        )
        try:
            market_data.start_websocket()
        except Exception:
            pass

    if strategy == "llm":
        agent = LLMAgent(
            api_key=_api_key_for_provider(provider_key),
            config=provider_config,
            provider=provider_key,
            base_url=provider_config.get("base_url"),
            symbols=uppercase_symbols,
            max_leverage=max_leverage,
            per_symbol_max_exposure=per_symbol_max_exposure,
            max_exposure_delta=max_exposure_delta,
            indicators=selected_indicators,
            gross_leverage_cap=gross_leverage_cap,
            net_exposure_cap=net_exposure_cap,
            max_open_positions=max_open_positions,
            max_turnover_per_step=max_turnover,
        )
    else:
        agent = BaselineAgent(
            strategy=strategy,
            symbols=uppercase_symbols,
            max_leverage=max_leverage,
        )

    if use_ui:
        reporter = TUIReporter()
    else:
        reporter = RealTimeReporter(print_interval_seconds=print_interval)
        
    session_logger = SessionLogger(log_dir)
    engine = RealTimeTradingEngine(
        market_data=market_data,
        agent=agent,
        initial_capital=initial_capital,
        max_leverage=max_leverage,
        poll_interval_seconds=poll,
        decision_interval_seconds=decision,
        min_long_exposure=min_long_exposure,
        per_symbol_max_exposure=per_symbol_max_exposure,
        max_exposure_delta=max_exposure_delta,
        commission_rate=commission,
        slippage=slippage,
        liquidation_threshold=TRADING_CONFIG.get("liquidation_threshold", 0.05),
        gross_leverage_cap=gross_leverage_cap,
        net_exposure_cap=net_exposure_cap,
        max_open_positions=max_open_positions,
        max_turnover_per_step=max_turnover,
    )

    run_args = {
        "mode": mode,
        "strategy": strategy,
        "duration_label": duration_label,
        "duration_seconds": duration_seconds,
        "symbols": uppercase_symbols,
        "poll_interval_seconds": poll,
        "decision_interval_seconds": decision,
        "max_leverage": max_leverage,
        "per_symbol_max_exposure": per_symbol_max_exposure,
        "max_exposure_delta": max_exposure_delta,
        "history_interval": history_interval,
        "history_lookback": history_lookback,
        "log_dir": log_dir,
        "min_long_exposure": min_long_exposure,
        "llm_provider": provider_key,
        "initial_capital": initial_capital,
        "indicators": selected_indicators,
        "commission": commission,
        "slippage": slippage,
        "use_ui": use_ui,
        "gross_leverage_cap": gross_leverage_cap,
        "net_exposure_cap": net_exposure_cap,
        "max_open_positions": max_open_positions,
        "max_turnover": max_turnover,
    }
    session_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        summary = engine.run(duration_seconds=duration_seconds, reporter=reporter)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Finalizing current session...")
        summary = {
            "equity_history": engine.equity_history,
            "trade_log": engine.trade_log,
            "decision_log": engine.decision_log,
            "final_account": {
                "balance": engine.account.balance,
                "equity": engine.account.equity,
                "realized_pnl": engine.account.realized_pnl,
                "unrealized_pnl": engine.account.unrealized_pnl,
                "margin_used": engine.account.margin_used,
                "available_margin": engine.account.available_margin,
            },
            "reports": reporter.finalize(engine.equity_history, engine.trade_log),
        }

    session_end = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    log_path = session_logger.save_session(
        run_args=run_args,
        summary=summary,
        start_time=session_start,
        end_time=session_end,
    )
    try:
        market_data.stop_websocket()
    except Exception:
        pass

    final_account = summary.get("final_account", {})
    print("\n=== Session Metadata ===")
    print(f"Final equity:   {final_account.get('equity', 0.0):.2f} USDT")
    print(f"Realized PnL:   {final_account.get('realized_pnl', 0.0):.2f} USDT")
    print(f"Unrealized PnL: {final_account.get('unrealized_pnl', 0.0):.2f} USDT")
    print(f"Trades logged:  {len(summary.get('trade_log', []))}")
    print(f"Session log written to: {log_path}\n")


def _interactive_cli(args: argparse.Namespace) -> None:
    print("=== Interactive Session Setup ===")
    symbols = [symbol.upper() for symbol in args.symbols]
    durations = list(TRADING_CONFIG["duration_seconds"].keys())
    duration_label = _prompt_menu("Select duration", durations, args.duration)
    initial_capital = _prompt_float("Initial capital (USDT)", args.initial_capital)
    providers = sorted(AGENT_CONFIG.keys())
    provider_choice = _prompt_menu("Select provider or all", providers + ["all"], "all")
    if provider_choice == "all":
        providers_to_run = providers
    else:
        providers_to_run = [provider_choice]
    indicators = _prompt_indicators(args.indicators)

    for provider in providers_to_run:
        _run_trading_session(
            provider=provider,
            symbols=symbols,
            duration_label=duration_label,
            poll=args.poll,
            decision=args.decision,
            max_leverage=args.max_leverage,
            per_symbol_max_exposure=args.per_symbol_max_exposure,
            max_exposure_delta=args.max_exposure_delta,
            history_interval=args.history_interval,
            history_lookback=args.history_lookback,
            print_interval=args.print_interval,
            log_dir=args.log_dir,
            min_long_exposure=args.min_long_exposure,
            initial_capital=initial_capital,
            indicators=indicators,
            mode=args.mode,
            strategy=args.strategy,
            commission=args.commission,
            slippage=args.slippage,
            data_path=args.data_path,
            use_ui=args.ui,
            gross_leverage_cap=args.gross_leverage_cap,
            net_exposure_cap=args.net_exposure_cap,
            max_open_positions=args.max_open_positions,
            max_turnover=args.max_turnover,
        )


def main() -> None:
    args = _parse_args()
    if not getattr(args, "non_interactive", False):
        _interactive_cli(args)
    else:
        _run_trading_session(
            provider=args.llm_provider,
            symbols=args.symbols,
            duration_label=args.duration,
            poll=args.poll,
            decision=args.decision,
            max_leverage=args.max_leverage,
            per_symbol_max_exposure=args.per_symbol_max_exposure,
            max_exposure_delta=args.max_exposure_delta,
            history_interval=args.history_interval,
            history_lookback=args.history_lookback,
            print_interval=args.print_interval,
            log_dir=args.log_dir,
            min_long_exposure=args.min_long_exposure,
            initial_capital=args.initial_capital,
            indicators=args.indicators,
            mode=args.mode,
            strategy=args.strategy,
            commission=args.commission,
            slippage=args.slippage,
            data_path=args.data_path,
            use_ui=args.ui,
            gross_leverage_cap=args.gross_leverage_cap,
            net_exposure_cap=args.net_exposure_cap,
            max_open_positions=args.max_open_positions,
            max_turnover=args.max_turnover,
        )


if __name__ == "__main__":
    main()
