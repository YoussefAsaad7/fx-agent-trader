"""
MT5 Infrastructure Package
==========================

This package provides a clean, asynchronous, and SOLID-compliant adapter
for interacting with the MetaTrader 5 trading platform.

It is structured using Domain-Driven Design (DDD) and Ports & Adapters
principles to separate domain logic from infrastructure concerns.

Package Structure:
------------------
- domain.py:      Contains pure, MT5-agnostic data classes (domain models).
- ports.py:       Defines the abstract interfaces (Ports) for services.
- connector.py:   Manages the MT5 connection lifecycle.
- adapters.py:    Provides concrete implementations (Adapters) of the ports
                  using the MetaTrader5 library.
- indicators.py:  Contains services for technical indicator calculations.
- utils.py:       Provides helper functions for calculations and string formatting.

Public API:
-----------
This __init__.py file acts as a Facade, re-exporting the key
public components so that consumers can import them directly from the `mt5`
package, e.g., `from mt5 import MT5Connector, IMarketDataRepository`.
"""

import logging

# Set up a default null handler to avoid "No handler found" warnings
# if the consuming application doesn't configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Export Domain Models
from .domain import (
    SymbolInfo,
    TickQuote,
    MarketCandle,
    AccountState,
    OrderResult,
    PositionInfo,
    PositionHistoryInfo,
    TechnicalIndicators
)

# Export Ports (Interfaces)
from .ports import (
    IMarketDataRepository,
    ITradeExecutionService,
    ITechnicalIndicatorService
)

# Export Connection Manager
from .connector import MT5Connector

# Export Adapters (Concrete Implementations)
from .adapters import (
    MT5MarketDataRepository,
    MT5TradeExecutionService
)

# Export Indicator Services
from .indicators import PandasTAIndicatorService

# Export Utilities
from .utils import (
    round_to_step,
    calculate_lot_size,
    safe_comment
)


__all__ = [
    # Domain
    "SymbolInfo",
    "TickQuote",
    "MarketCandle",
    "AccountState",
    "OrderResult",
    "PositionInfo",
    "PositionHistoryInfo",
    "TechnicalIndicators",

    # Ports
    "IMarketDataRepository",
    "ITradeExecutionService",
    "ITechnicalIndicatorService",

    # Infrastructure & Services
    "MT5Connector",
    "MT5MarketDataRepository",
    "MT5TradeExecutionService",
    "PandasTAIndicatorService",

    # Utilities
    "calculate_lot_size",
    "round_to_step",
    "safe_comment",
]