"""
config.py
Centralized configuration for the Forex Trading Agent.

Loads settings from a .env file and defines the symbols to watch.
Separates configuration from application logic (SOLID's SRP).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
# Try to import mt5 to get timeframe constants
try:
    import MetaTrader5 as mt5
except ImportError:
    logger.warning("MetaTrader5 not found, using integer fallbacks for timeframes.")


    # Define fallbacks if mt5 is not installed
    class mt5:
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_H1 = 16385
        TIMEFRAME_H4 = 16388

# --- Load .env file ---
# Create a file named .env in the same directory and add your keys:
# DEEPSEEK_API_KEY=sk-your-key-here
# GEMINI_API_KEY=your-key-here
# MT5_LOGIN=your-account-number
# MT5_PASSWORD=your-password
# MT5_SERVER=your-broker-server
load_dotenv()

from agent_context_builder import SymbolWatchConfig, TimeframeConfig




@dataclass(frozen=True)
class Config:
    """
    Holds all configuration for the application, loaded from environment variables.
    """
    # LLM Configuration
    llm_provider: str = field(default=os.getenv("LLM_PROVIDER", "gemini"))
    deepseek_api_key: Optional[str] = field(default=os.getenv("DEEPSEEK_API_KEY"))
    gemini_api_key: Optional[str] = field(default=os.getenv("GEMINI_API_KEY", ""))

    # MetaTrader 5 Configuration
    mt5_login: int = field(default=int(os.getenv("MT5_LOGIN", 0)))
    mt5_password: str = field(default=os.getenv("MT5_PASSWORD", ""))
    mt5_server: str = field(default=os.getenv("MT5_SERVER", ""))
    # Path to MT5 terminal.exe (adjust as needed)
    mt5_path: str = field(default=os.getenv("MT5_PATH", "C:\\Program Files\\MetaTrader 5\\terminal64.exe"))

    # Scheduler Configuration
    run_interval_seconds: int = field(default=int(os.getenv("RUN_INTERVAL_SECONDS", 300)))  # 5 minutes

    # Trading Configuration
    system_prompt_path: str = field(default="trend.txt")
    dry_run: bool = field(default=os.getenv("DRY_RUN", "true").lower() == "true")

    # This is the list of symbols and timeframes the agent will analyze
    watch_list: List[SymbolWatchConfig] = field(default_factory=lambda: [
        SymbolWatchConfig(
            symbol="EURUSD",
            timeframes=[
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M3, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M15, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H1, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H4, candle_count=200),
            ]
        ),
        SymbolWatchConfig(
            symbol="GBPUSD",
            timeframes=[
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M3, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M15, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H1, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H4, candle_count=200),
            ]
        ),
        SymbolWatchConfig(
            symbol="USDJPY",
            timeframes=[
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M3, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M15, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H1, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H4, candle_count=200),
            ]
        ),
        SymbolWatchConfig(
            symbol="XAUUSD",
            timeframes=[
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M3, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M15, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H1, candle_count=200),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H4, candle_count=200),
            ]
        ),
    ])


def load_config() -> Config:
    """Loads and validates the application configuration."""
    cfg = Config()

    # Validate MT5 config
    if not cfg.mt5_login or not cfg.mt5_password or not cfg.mt5_server:
        raise ValueError("MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER must be set in .env file")

    # Validate LLM config
    if cfg.llm_provider == "deepseek" and not cfg.deepseek_api_key:
        raise ValueError("LLM_PROVIDER is 'deepseek' but DEEPSEEK_API_KEY is not set.")
    if cfg.llm_provider == "gemini" and not cfg.gemini_api_key:
        logger.warning("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set. Using default.")

    logger.info(f"Configuration loaded. LLM: {cfg.llm_provider}, Dry Run: {cfg.dry_run}")
    return cfg