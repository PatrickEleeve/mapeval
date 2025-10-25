"""Entry point for real-time leveraged futures trading."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, List

from config import AGENT_CONFIG, DEEPSEEK_API_KEY, OPENAI_API_KEY, TRADING_CONFIG
from data_manager import RealTimeMarketData
from llm_agent import LLMAgent
from log_manager import SessionLogger
from reporter import RealTimeReporter
from trading_engine import RealTimeTradingEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-time leveraged trading session.")
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
    return parser.parse_args()


def _select_duration(duration_label: str) -> float:
    durations: Dict[str, int] = TRADING_CONFIG["duration_seconds"]
    return float(durations.get(duration_label, durations["1h"]))


def main() -> None:
    args = _parse_args()
    duration_seconds = _select_duration(args.duration)
    symbols: List[str] = [symbol.upper() for symbol in args.symbols]
    provider = args.llm_provider.lower()
    provider_config: Dict[str, float] = AGENT_CONFIG.get(provider, {})

    print("=== Real-Time Leveraged Trading ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Duration: {args.duration} ({duration_seconds:.0f} seconds)")
    print(f"Polling interval: {args.poll} s | Decision interval: {args.decision} s")
    print(f"Maximum leverage: {args.max_leverage}x")
    print(f"LLM provider: {provider}")
    if args.min_long_exposure > 0:
        print(f"Minimum long exposure: {args.min_long_exposure:.4f}x of equity")

    market_data = RealTimeMarketData(
        symbols=symbols,
        interval=args.history_interval,
        lookback=args.history_lookback,
    )
    try:
        market_data.start_websocket()
    except Exception:
        pass
    if provider == "deepseek":
        api_key = DEEPSEEK_API_KEY
    else:
        api_key = OPENAI_API_KEY

    base_url = provider_config.get("base_url")

    agent = LLMAgent(
        api_key=api_key,
        config=provider_config,
        provider=provider,
        base_url=base_url,
        symbols=symbols,
        max_leverage=args.max_leverage,
    )
    reporter = RealTimeReporter(print_interval_seconds=args.print_interval)
    session_logger = SessionLogger(args.log_dir)
    engine = RealTimeTradingEngine(
        market_data=market_data,
        agent=agent,
        initial_capital=TRADING_CONFIG["initial_capital"],
        max_leverage=args.max_leverage,
        poll_interval_seconds=args.poll,
        decision_interval_seconds=args.decision,
        min_long_exposure=args.min_long_exposure,
    )

    run_args = {
        "duration_label": args.duration,
        "duration_seconds": duration_seconds,
        "symbols": symbols,
        "poll_interval_seconds": args.poll,
        "decision_interval_seconds": args.decision,
        "max_leverage": args.max_leverage,
        "history_interval": args.history_interval,
        "history_lookback": args.history_lookback,
        "log_dir": args.log_dir,
        "min_long_exposure": args.min_long_exposure,
        "llm_provider": provider,
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
    print(f"Session log written to: {log_path}")


if __name__ == "__main__":
    main()
