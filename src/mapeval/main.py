"""Entry point for real-time leveraged futures trading."""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from mapeval.config import AGENT_CONFIG, DEEPSEEK_API_KEY, OPENAI_API_KEY, QWEN_API_KEY, TRADING_CONFIG
from mapeval.data_manager import BacktestMarketData, RealTimeMarketData, load_historical_data
from mapeval.llm_agent import BaselineAgent, LLMAgent
from mapeval.log_manager import SessionLogger
from mapeval.rate_limiter import RateLimiter
from mapeval.reporter import RealTimeReporter
from mapeval.risk_manager import RiskLimits, RiskManager
from mapeval.tui_reporter import TUIReporter
from mapeval.trading_engine import RealTimeTradingEngine
from mapeval.event_bus import EventBus
from mapeval.events import Event, EventType, risk_alert
from mapeval.notifier import LogNotifier, create_notifier
from mapeval.security import AuditLogger, ReadOnlyGuard, generate_api_token, mask_key, validate_api_keys
from mapeval.order_executor import create_executor
from mapeval import binance_data_source
from mapeval.binance_futures_client import BinanceFuturesClient

logger = logging.getLogger(__name__)

# Optional imports for database and API server
try:
    from mapeval.database import DatabaseManager
    from mapeval.db_models import SQLALCHEMY_AVAILABLE
    DB_AVAILABLE = SQLALCHEMY_AVAILABLE
except ImportError:
    DB_AVAILABLE = False

try:
    from mapeval.api_server import FASTAPI_AVAILABLE, start_api_server
    API_AVAILABLE = FASTAPI_AVAILABLE
except ImportError:
    API_AVAILABLE = False

# Global engine reference for signal handlers
_active_engine: Optional[RealTimeTradingEngine] = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    sig_name = signal.Signals(signum).name
    print(f"\n[{sig_name}] Graceful shutdown initiated...")
    if _active_engine is not None:
        _active_engine.shutdown()
    else:
        sys.exit(0)


def _atexit_handler():
    """Safety net: ensure positions are closed on exit."""
    if _active_engine is not None and _active_engine.account.positions:
        logger.warning("atexit: closing %d remaining positions", len(_active_engine.account.positions))
        _active_engine.shutdown()


AVAILABLE_INDICATORS = (
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "coefficient_of_variation",
    "ma_slope",
)


def _compact_symbols(symbols: Sequence[str], limit: int = 4) -> str:
    """Render a short symbol summary for startup output."""
    visible = [symbol.upper() for symbol in symbols[:limit]]
    remainder = max(0, len(symbols) - len(visible))
    if remainder:
        return f"{', '.join(visible)} +{remainder} more"
    return ", ".join(visible)


def _print_startup_summary(
    *,
    mode: str,
    execution_mode: str,
    strategy: str,
    provider_key: str,
    symbols: Sequence[str],
    duration_label: str,
    duration_seconds: float,
    initial_capital: float,
    use_ui: bool,
    reconcile_interval: float,
) -> None:
    """Print a concise startup summary tailored to the selected mode."""
    print("=== MAPEval ===")
    summary = f"{mode}/{execution_mode} | strategy={strategy}"
    if strategy == "llm":
        summary += f" | provider={provider_key}"
    print(summary)
    print(f"Symbols: {_compact_symbols(symbols)}")
    print(f"Duration: {duration_label} ({duration_seconds:.0f}s) | Capital: {initial_capital:.2f} USDT")
    if use_ui:
        print("UI: enabled")
    if execution_mode in ("paper", "live"):
        print(f"Safety: reconcile every {reconcile_interval:.0f}s")
    elif execution_mode == "simulation":
        print("Safety: simulation only, no external orders")


def _resolve_live_environment(binance_testnet: bool, binance_mainnet: bool) -> str:
    """Return the target Binance environment for live execution."""
    if binance_testnet and binance_mainnet:
        raise ValueError("Choose only one of --binance-testnet or --binance-mainnet")
    if binance_mainnet:
        return "mainnet"
    return "testnet"


def _required_live_confirmation(environment: str) -> str:
    """Return the exact confirmation phrase required for live execution."""
    return f"ENABLE BINANCE {environment.upper()} LIVE"


def _confirm_live_execution(
    environment: str,
    provided_confirmation: str | None,
    non_interactive: bool,
) -> None:
    """Require an explicit confirmation phrase before live trading starts."""
    expected = _required_live_confirmation(environment)
    if non_interactive:
        if provided_confirmation != expected:
            raise ValueError(
                f"Live trading requires --live-confirmation '{expected}' for {environment}"
            )
        return

    if provided_confirmation == expected:
        return

    print(f"[SAFE] Live mode target: Binance {environment}")
    print(f"[SAFE] Type the exact confirmation phrase to continue: {expected}")
    confirmed = input("Confirmation: ").strip()
    if confirmed != expected:
        raise ValueError("Live trading confirmation failed")


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
        help="Commission rate (e.g. 0.0005 for 0.05%%).",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=TRADING_CONFIG.get("slippage", 0.0),
        help="Slippage rate (e.g. 0.0005 for 0.05%%).",
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
        "--execution-mode",
        choices=["simulation", "paper", "live"],
        default="simulation",
        help="Execution mode: simulation (no real orders), paper (real data, simulated fills), live (real orders).",
    )
    parser.add_argument(
        "--allow-live-trading",
        action="store_true",
        help="Required safety switch for live execution mode.",
    )
    parser.add_argument(
        "--binance-testnet",
        action="store_true",
        help="Use Binance Futures testnet for live execution.",
    )
    parser.add_argument(
        "--binance-mainnet",
        action="store_true",
        help="Use Binance Futures mainnet for live execution. Overrides the safe default testnet target.",
    )
    parser.add_argument(
        "--live-confirmation",
        default=None,
        help="Exact confirmation phrase required for live execution.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Block order placement and cancellation while keeping monitoring active.",
    )
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Start the REST API server for monitoring (requires fastapi).",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the REST API server.",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Bearer/X-API-Key token for protecting REST API access. Auto-generated for paper/live if omitted.",
    )
    parser.add_argument(
        "--reconcile-interval",
        type=float,
        default=300.0,
        help="Seconds between exchange reconciliation checks in paper/live modes (0 disables).",
    )
    parser.add_argument(
        "--enable-db",
        action="store_true",
        help="Enable database persistence (requires sqlalchemy).",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL (default: sqlite:///data/trading.db).",
    )
    parser.add_argument(
        "--telegram-bot-token",
        default=None,
        help="Telegram bot token for notifications.",
    )
    parser.add_argument(
        "--telegram-chat-id",
        default=None,
        help="Telegram chat ID for notifications.",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Webhook URL for notifications.",
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
    execution_mode: str = "simulation",
    allow_live_trading: bool = False,
    binance_testnet: bool = False,
    binance_mainnet: bool = False,
    live_confirmation: str | None = None,
    read_only: bool = False,
    non_interactive: bool = False,
    enable_api: bool = False,
    api_port: int = 8000,
    api_token: str | None = None,
    reconcile_interval: float = 300.0,
    enable_db: bool = False,
    db_url: str | None = None,
    telegram_bot_token: str | None = None,
    telegram_chat_id: str | None = None,
    webhook_url: str | None = None,
) -> None:
    provider_key = provider.lower()
    provider_config: Dict[str, float] = AGENT_CONFIG.get(provider_key, {})
    duration_seconds = _select_duration(duration_label)
    uppercase_symbols: List[str] = [symbol.upper() for symbol in symbols]
    selected_indicators = _normalize_indicators(indicators)
    _print_startup_summary(
        mode=mode,
        execution_mode=execution_mode,
        strategy=strategy,
        provider_key=provider_key,
        symbols=uppercase_symbols,
        duration_label=duration_label,
        duration_seconds=duration_seconds,
        initial_capital=initial_capital,
        use_ui=use_ui,
        reconcile_interval=reconcile_interval,
    )

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
        try:
            market_data = RealTimeMarketData(
                symbols=uppercase_symbols,
                interval=history_interval,
                lookback=history_lookback,
            )
        except Exception as exc:
            raise RuntimeError(
                "Market data bootstrap failed. Check Binance connectivity, TLS/network access, "
                f"or reduce symbol count. Root cause: {exc}"
            ) from exc
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

    # Initialize rate limiter for Binance API
    rate_limiter = RateLimiter(max_weight_per_minute=1200)
    binance_data_source.set_rate_limiter(rate_limiter)

    # Initialize risk manager
    risk_limits = RiskLimits(
        max_drawdown=0.20,
        max_daily_loss=0.05,
        max_consecutive_losses=5,
        cooldown_after_loss_seconds=300,
        min_equity_floor=0.10,
    )
    risk_mgr = RiskManager(limits=risk_limits)

    # ── Event Bus ──────────────────────────────────────────────────
    event_bus = EventBus()

    # ── Notification System ──────────────────────────────────────
    notifier_config = {}
    if telegram_bot_token and telegram_chat_id:
        notifier_config["telegram"] = {
            "bot_token": telegram_bot_token,
            "chat_id": telegram_chat_id,
        }
    if webhook_url:
        notifier_config["webhook"] = {"url": webhook_url}
    notifier = create_notifier(notifier_config)

    # Wire notifier to event bus for key events
    def _on_risk_alert(evt: Event) -> None:
        payload = evt.payload
        notifier.send_alert(
            level=payload.get("severity", "warning"),
            title=f"Risk Alert: {payload.get('alert_type', 'unknown')}",
            message=payload.get("message", ""),
            data=payload.get("details"),
        )

    def _on_order_filled(evt: Event) -> None:
        payload = evt.payload
        notifier.send_alert(
            level="info",
            title=f"Order Filled: {payload.get('symbol', '')}",
            message=f"{payload.get('side', '')} {payload.get('quantity', 0):.4f} @ {payload.get('price', 0):.2f}",
            data=payload,
        )

    def _on_force_close(evt: Event) -> None:
        notifier.send_alert(
            level="critical",
            title="Force Close Triggered",
            message=evt.payload.get("message", "Positions force-closed by risk manager"),
            data=evt.payload,
        )

    event_bus.subscribe(EventType.RISK_ALERT, _on_risk_alert)
    event_bus.subscribe(EventType.ORDER_FILLED, _on_order_filled)
    event_bus.subscribe(EventType.FORCE_CLOSE, _on_force_close)

    # ── Audit Logger ─────────────────────────────────────────────
    audit_logger = AuditLogger(log_dir=os.path.join(log_dir, "audit"))

    # ── Database ─────────────────────────────────────────────────
    db_manager = None
    if enable_db and DB_AVAILABLE:
        try:
            db_manager = DatabaseManager(url=db_url)
            db_manager.create_tables()
            print(f"[DB] Database initialized: {db_manager.url}")
        except Exception as exc:
            logger.warning("Failed to initialize database: %s", exc)
            db_manager = None
    elif enable_db and not DB_AVAILABLE:
        print("[DB] sqlalchemy not installed. Install with: pip install sqlalchemy>=2.0")

    # ── Order Executor ───────────────────────────────────────────
    order_executor = None
    if execution_mode in ("paper", "live"):
        binance_client = None
        if execution_mode == "live":
            if not allow_live_trading:
                raise ValueError("Live trading requires --allow-live-trading")
            live_environment = _resolve_live_environment(binance_testnet, binance_mainnet)
            _confirm_live_execution(
                environment=live_environment,
                provided_confirmation=live_confirmation,
                non_interactive=non_interactive,
            )
            binance_api_key = os.getenv("BINANCE_API_KEY")
            binance_api_secret = os.getenv("BINANCE_API_SECRET")
            key_warnings = validate_api_keys(binance_api_key, binance_api_secret)
            if key_warnings:
                raise ValueError(f"Live trading blocked due to credential issues: {'; '.join(key_warnings)}")
            binance_client = BinanceFuturesClient(
                api_key=binance_api_key or "",
                api_secret=binance_api_secret or "",
                testnet=(live_environment == "testnet"),
                rate_limiter=rate_limiter,
            )
            print(
                f"[LIVE] Binance credentials loaded: key={mask_key(binance_api_key or '')} "
                f"| environment={live_environment}"
            )
        order_executor = create_executor(
            mode=execution_mode,
            commission_rate=commission,
            slippage=slippage,
            initial_balance=initial_capital,
            binance_client=binance_client,
            market_data_provider=market_data,
        )
        print(f"[OMS] Order executor: {type(order_executor).__name__}")

    # ── Read-Only Guard (safety for live mode) ───────────────────
    read_only_guard = ReadOnlyGuard(enabled=read_only)
    if read_only and order_executor is not None:
        from mapeval.order_executor import GuardedOrderExecutor
        order_executor = GuardedOrderExecutor(order_executor, read_only_guard)
        print("[SAFE] Read-only guard enabled; order placement is blocked")

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
        risk_manager=risk_mgr,
        execution_mode=execution_mode,
        order_executor=order_executor,
        reconcile_interval_seconds=reconcile_interval,
    )

    # Store event bus and notifier on engine for access by other components
    engine.event_bus = event_bus
    engine.notifier = notifier
    engine.audit_logger = audit_logger
    engine.read_only_guard = read_only_guard

    # Register signal handlers for graceful shutdown
    global _active_engine
    _active_engine = engine
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_atexit_handler)

    # ── API Server ───────────────────────────────────────────────
    api_thread = None
    resolved_api_token = api_token or os.getenv("MAPEVAL_API_TOKEN")
    if enable_api and API_AVAILABLE:
        if execution_mode in ("paper", "live") and not resolved_api_token:
            resolved_api_token = generate_api_token()
            print(f"[API] Generated token for this session: {resolved_api_token}")
        api_thread = start_api_server(
            host="0.0.0.0",
            port=api_port,
            engine=engine,
            event_bus=event_bus,
            db_manager=db_manager,
            api_token=resolved_api_token,
        )
        if api_thread is not None:
            print(f"[API] Monitoring server started on http://localhost:{api_port}")
            if resolved_api_token:
                print("[API] Auth enabled for REST endpoints via Authorization: Bearer <token> or X-API-Key")
    elif enable_api and not API_AVAILABLE:
        print("[API] fastapi not installed. Install with: pip install fastapi uvicorn")

    # Publish session start event
    event_bus.publish(Event(
        event_type=EventType.SESSION_START,
        payload={
            "execution_mode": execution_mode,
            "symbols": uppercase_symbols,
            "initial_capital": initial_capital,
        },
        source="main",
    ))

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
        "execution_mode": execution_mode,
        "allow_live_trading": allow_live_trading,
        "binance_testnet": binance_testnet,
        "binance_mainnet": binance_mainnet,
        "live_confirmation": bool(live_confirmation),
        "read_only": read_only,
        "enable_api": enable_api,
        "api_token": bool(resolved_api_token),
        "reconcile_interval": reconcile_interval,
        "enable_db": enable_db,
    }
    session_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        summary = engine.run(duration_seconds=duration_seconds, reporter=reporter)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Closing positions and finalizing...")
        engine.shutdown()
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
            "risk_report": risk_mgr.get_risk_report(engine.account.equity),
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

    # ── Persist to Database ──────────────────────────────────────
    if db_manager is not None:
        try:
            from mapeval.repositories import (
                SessionRepository, TradeRepository, DecisionRepository, EquityRepository,
            )
            with db_manager.session() as db_session:
                session_repo = SessionRepository(db_session)
                session_id = session_repo.create(
                    initial_capital=initial_capital,
                    execution_mode=execution_mode,
                    config=run_args,
                )
                trade_repo = TradeRepository(db_session)
                trade_repo.save_batch(session_id, summary.get("trade_log", []))

                decision_repo = DecisionRepository(db_session)
                for decision in summary.get("decision_log", []):
                    decision_repo.save(session_id, decision)

                equity_repo = EquityRepository(db_session)
                for snap in summary.get("equity_history", []):
                    equity_repo.save(session_id, snap)

                session_repo.complete(
                    session_id=session_id,
                    final_equity=final_account.get("equity", 0.0),
                    total_pnl=final_account.get("realized_pnl", 0.0),
                    total_trades=len(summary.get("trade_log", [])),
                )
            print(f"[DB] Session {session_id} persisted to database")
        except Exception as exc:
            logger.warning("Failed to persist session to database: %s", exc)

    # ── Publish Session End Event ────────────────────────────────
    event_bus.publish(Event(
        event_type=EventType.SESSION_END,
        payload={
            "final_equity": final_account.get("equity", 0.0),
            "realized_pnl": final_account.get("realized_pnl", 0.0),
            "total_trades": len(summary.get("trade_log", [])),
        },
        source="main",
    ))

    print("\n=== Session Metadata ===")
    print(f"Execution mode: {execution_mode}")
    print(f"Final equity:   {final_account.get('equity', 0.0):.2f} USDT")
    print(f"Realized PnL:   {final_account.get('realized_pnl', 0.0):.2f} USDT")
    print(f"Unrealized PnL: {final_account.get('unrealized_pnl', 0.0):.2f} USDT")
    print(f"Trades logged:  {len(summary.get('trade_log', []))}")
    print(f"Session log written to: {log_path}")
    print(f"Event bus stats: {event_bus.get_stats()}\n")


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
            execution_mode=args.execution_mode,
            allow_live_trading=args.allow_live_trading,
            binance_testnet=args.binance_testnet,
            binance_mainnet=args.binance_mainnet,
            live_confirmation=args.live_confirmation,
            read_only=args.read_only,
            non_interactive=False,
            enable_api=args.enable_api,
            api_port=args.api_port,
            api_token=args.api_token,
            reconcile_interval=args.reconcile_interval,
            enable_db=args.enable_db,
            db_url=args.db_url,
            telegram_bot_token=args.telegram_bot_token,
            telegram_chat_id=args.telegram_chat_id,
            webhook_url=args.webhook_url,
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
            execution_mode=args.execution_mode,
            allow_live_trading=args.allow_live_trading,
            binance_testnet=args.binance_testnet,
            binance_mainnet=args.binance_mainnet,
            live_confirmation=args.live_confirmation,
            read_only=args.read_only,
            non_interactive=args.non_interactive,
            enable_api=args.enable_api,
            api_port=args.api_port,
            api_token=args.api_token,
            reconcile_interval=args.reconcile_interval,
            enable_db=args.enable_db,
            db_url=args.db_url,
            telegram_bot_token=args.telegram_bot_token,
            telegram_chat_id=args.telegram_chat_id,
            webhook_url=args.webhook_url,
        )


if __name__ == "__main__":
    main()
