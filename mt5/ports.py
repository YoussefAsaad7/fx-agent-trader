"""
MT5 Application Ports (Interfaces)
----------------------------------

This file defines the abstract interfaces (Ports) that the core
application logic will interact with. This adheres to the
Dependency Inversion Principle (D of SOLID).

The core application should only depend on these protocols,
not on the concrete MT5 implementations.
"""

from typing import Protocol, List, Optional
from .domain import (
    SymbolInfo,
    MarketCandle,
    AccountState,
    PositionInfo,
    PositionHistoryInfo,
    TickQuote,
    OrderResult,
    TechnicalIndicators
)

# --- Market Data Port ---

class IMarketDataRepository(Protocol):
    """Interface for fetching market and account data."""

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetches and normalizes symbol information."""
        ...

    async def get_last_candles(self, symbol: str, timeframe: int, count: int) -> List[MarketCandle]:
        """Fetches the last 'count' candles for a symbol."""
        ...

    async def get_account_state(self) -> AccountState:
        """Fetches the current account state."""
        ...

    async def get_open_positions(self) -> List[PositionInfo]:
        """Fetches all open positions from the account."""
        ...

    async def get_historical_trade(self, ticket: int) -> Optional[PositionHistoryInfo]:
        """
        Fetches the deal history for a specific position ticket
        to find its closing details.
        """
        ...

    async def get_tick(self, symbol: str) -> Optional[TickQuote]:
        """Returns the latest live tick (bid/ask) for the given symbol."""
        ...


# --- Trade Execution Port ---

class ITradeExecutionService(Protocol):
    """Interface for executing and managing trade orders."""

    async def place_market_order(self,
                                 symbol: str,
                                 action: str,  # 'buy' or 'sell'
                                 lot: float,
                                 stop_loss: Optional[float],
                                 take_profit: Optional[float],
                                 comment: Optional[str]) -> OrderResult:
        """Places a new market order."""
        ...

    async def close_position(self,
                             position: PositionInfo,
                             comment: Optional[str]) -> OrderResult:
        """Closes an entire existing open position."""
        ...

    # --- NEW FEATURES ---

    async def partial_close_position(self,
                                     ticket: int,
                                     lot_to_close: float,
                                     comment: Optional[str]) -> OrderResult:
        """
        Partially closes an existing open position by a specified lot amount.
        """
        ...

    async def update_position_sl(self,
                                 ticket: int,
                                 stop_loss: float) -> OrderResult:
        """
        Updates the stop loss for an existing open position.
        """
        ...

    async def update_position_tp(self,
                                 ticket: int,
                                 take_profit: float) -> OrderResult:
        """
        Updates the take profit for an existing open position.
        """
        ...


# --- Technical Analysis Port ---

class ITechnicalIndicatorService(Protocol):
    """
    Interface for a service that calculates TA indicators from candle data.
    """
    async def calculate_indicators(self,
                                     candles: List[MarketCandle],
                                     fast_ema_period: int = 12,
                                     slow_ema_period: int = 26,
                                     macd_signal_period: int = 9,
                                     rsi_period: int = 14,
                                     atr_period: int = 14) -> TechnicalIndicators:
        """Calculates a standard set of indicators from the given candles."""
        ...