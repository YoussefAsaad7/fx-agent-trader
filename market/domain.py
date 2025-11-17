"""
MT5 Domain Models
-----------------

This file defines the pure data classes (dataclasses) that represent
the core concepts of the trading domain.

These models are completely independent of the MetaTrader5 library
or any other infrastructure concern. They are the "nouns" of the system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class SymbolInfo:
    """
    A domain model representing all critical properties of a trading symbol.
    """
    name: str
    point: float
    digits: int
    spread: int                 # Spread in integer points
    trade_contract_size: float

    # Lot Sizing & Risk (CRITICAL)
    volume_min: float           # Minimum trade volume (e.g., 0.01)
    volume_step: float          # Lot step (e.g., 0.01 or 1.0)
    volume_max: float
    trade_tick_value: float     # The value of 1.0 'point' for 1 lot in account currency

    # Informational
    swap_long: float
    swap_short: float

@dataclass
class TickQuote:
    """Represents a live bid/ask price tick."""
    symbol: str
    bid: float
    ask: float
    time: float

@dataclass
class MarketCandle:
    """Represents a single OHLCV candle."""
    time: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: Optional[int] = None

@dataclass
class AccountState:
    """Represents the current state of the trading account."""
    login: int
    balance: float
    equity: float
    free_margin: float
    margin_level: Optional[float]
    currency: str

@dataclass
class OrderResult:
    """Represents the outcome of a trade execution request."""
    success: bool
    ticket: Optional[int]
    comment: Optional[str]
    raw: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class PositionInfo:
    """Consolidated info about an open position."""
    ticket: int
    symbol: str
    type: int # 0 for BUY, 1 for SELL
    volume: float
    open_price: float
    open_time: datetime
    profit: float
    swap: float
    comment: str
    current_price: float
    stop_loss: float
    take_profit: float

@dataclass(frozen=True)
class PositionHistoryInfo:
    """Consolidated info about a *closed* trade deal."""
    ticket: int
    symbol: str
    type: int # 0 for BUY, 1 for SELL
    volume: float
    open_price: float
    close_price: float
    open_time: datetime
    close_time: datetime
    profit: float
    swap: float
    comment: str

@dataclass(frozen=True)
class TechnicalIndicators:
    """
    Domain model for calculated indicators. All lists are oldest -> newest.
    """
    # Using field to allow default empty lists
    ema_slow: List[float] = field(default_factory=list)
    ema_fast: List[float] = field(default_factory=list)
    macd: List[float] = field(default_factory=list)
    macd_signal: List[float] = field(default_factory=list)
    rsi: List[float] = field(default_factory=list)
    atr: List[float] = field(default_factory=list)