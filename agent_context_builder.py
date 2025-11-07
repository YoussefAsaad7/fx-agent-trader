"""
agent_context_builder.py

This module provides a professional, DDD-style service for building the
complete data context required by the Forex trading agent.

It uses the services from `mt5_module.py` to fetch, calculate, and
structure all data (account, positions, market, and technical)
for multiple symbols and timeframes.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Import all necessary components from the upgraded mt5_module
from mt5 import (
    IMarketDataRepository,
    ITechnicalIndicatorService,
    AccountState,
    PositionInfo,
    SymbolInfo,
    MarketCandle,
    TechnicalIndicators,
    MT5Connector,
    MT5MarketDataRepository,
    PandasTAIndicatorService
)

logger = logging.getLogger(__name__)


# ----------------------------- Configuration Models -----------------------------

@dataclass(frozen=True)
class TimeframeConfig:
    """Defines what data to get for a single timeframe."""
    timeframe: int  # e.g., mt5.TIMEFRAME_M5
    candle_count: int = 50  # Default 50 candles
    # --- Indicator parameters ---
    fast_ema: int = 12
    slow_ema: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    atr_period: int = 14


@dataclass(frozen=True)
class SymbolWatchConfig:
    """Defines which symbol to watch and which timeframes to get."""
    symbol: str
    timeframes: List[TimeframeConfig]


# ----------------------------- Agent Data Models (DDD) ----------------------------
# These models represent the final, structured data for the agent's prompt

@dataclass(frozen=True)
class AgentMarketData:
    """
    All market data for a *single timeframe* of a symbol,
    including candles and indicators.
    """
    timeframe: int
    candles: List[MarketCandle]
    indicators: TechnicalIndicators


@dataclass(frozen=True)
class AgentSymbolContext:
    """
    The complete context for a *single symbol*, including its
    info and all requested timeframe data.
    """
    symbol_info: SymbolInfo
    market_data: List[AgentMarketData]  # List of data for each timeframe


@dataclass(frozen=True)
class AgentFullContext:
    """
    The complete, final data object to be passed to the agent.
    Contains all information needed to make a decision.
    """
    account_state: AccountState
    open_positions: List[PositionInfo]
    market_context: Dict[str, AgentSymbolContext]  # Keyed by symbol name
    economic_events: List[Dict[str, str]] = field(default_factory=list)  # Mocked


# ----------------------------- Builder Service ----------------------------------

class AgentContextBuilder:
    """
    A service that encapsulates the logic of fetching, calculating,
    and structuring all data for the agent.
    """

    def __init__(self,
                 repo: IMarketDataRepository,
                 indicator_svc: ITechnicalIndicatorService):
        self._repo = repo
        self._indicator_svc = indicator_svc

    async def build_context(self, watch_list: List[SymbolWatchConfig]) -> AgentFullContext:
        """
        Builds the complete data context for the agent based on the watch list.
        """
        logger.info("Building agent context...")

        # 1. Fetch account-wide data in parallel
        logger.debug("Fetching account state and positions...")
        account_task = self._repo.get_account_state()
        positions_task = self._repo.get_open_positions()
        # TODO: Add economic event fetching task here
        # events_task = self._fetch_economic_events()

        account_state, open_positions = await asyncio.gather(
            account_task,
            positions_task
        )
        logger.info(f"Account: {account_state.login}, Positions: {len(open_positions)}")

        # 2. Fetch data for each symbol
        symbol_tasks = []
        for config in watch_list:
            symbol_tasks.append(self._build_symbol_context(config))

        market_context_list = await asyncio.gather(*symbol_tasks)

        # 3. Assemble final context
        market_context_dict: Dict[str, AgentSymbolContext] = {}
        for context in market_context_list:
            if context:
                market_context_dict[context.symbol_info.name] = context

        logger.info("Agent context build complete.")
        return AgentFullContext(
            account_state=account_state,
            open_positions=open_positions,
            market_context=market_context_dict,
            economic_events=[]  # Placeholder
        )

    async def _build_symbol_context(self, config: SymbolWatchConfig) -> Optional[AgentSymbolContext]:
        """
        Fetches SymbolInfo and all timeframe data for a single symbol.
        """
        logger.debug(f"Building context for {config.symbol}...")
        try:
            symbol_info = await self._repo.get_symbol_info(config.symbol)
        except Exception as e:
            logger.error(f"Failed to get symbol info for {config.symbol}: {e}. Skipping symbol.")
            return None

        # Fetch all timeframes for this symbol in parallel
        timeframe_tasks = []
        for tf_config in config.timeframes:
            timeframe_tasks.append(self._build_market_data(
                symbol_info.name,
                tf_config
            ))

        market_data_list = await asyncio.gather(*timeframe_tasks)

        return AgentSymbolContext(
            symbol_info=symbol_info,
            market_data=[data for data in market_data_list if data]
        )

    async def _build_market_data(self,
                                 symbol: str,
                                 tf_config: TimeframeConfig) -> Optional[AgentMarketData]:
        """
        Fetches candles and calculates indicators for a single timeframe.
        """
        logger.debug(f"Fetching {tf_config.candle_count} candles for {symbol} @ {tf_config.timeframe}...")
        try:
            # 1. Get candles
            candles = await self._repo.get_last_candles(
                symbol,
                tf_config.timeframe,
                tf_config.candle_count
            )

            if not candles:
                logger.warning(f"No candles returned for {symbol} @ {tf_config.timeframe}")
                return None

            # 2. Calculate indicators
            logger.debug(f"Calculating indicators for {symbol} @ {tf_config.timeframe}...")
            indicators = await self._indicator_svc.calculate_indicators(
                candles=candles,
                fast_ema_period=tf_config.fast_ema,
                slow_ema_period=tf_config.slow_ema,
                macd_signal_period=tf_config.macd_signal,
                rsi_period=tf_config.rsi_period,
                atr_period=tf_config.atr_period
            )

            return AgentMarketData(
                timeframe=tf_config.timeframe,
                candles=candles,
                indicators=indicators
            )

        except Exception as e:
            logger.error(f"Failed to build market data for {symbol} @ {tf_config.timeframe}: {e}")
            return None


# -------------------------- Example Usage ----------------------------------

async def main():
    """
    Demonstrates how to use the AgentContextBuilder.
    """

    # This example requires pandas and pandas-ta
    # pip install pandas pandas-ta
    try:
        import MetaTrader5 as mt5
        import pandas
        import pandas_ta
    except ImportError:
        logger.error("This example requires MetaTrader5, pandas, and pandas-ta.")
        logger.error("Please run: pip install MetaTrader5 pandas pandas-ta")
        return

    # 1. Define the watch list
    watch_list = [
        SymbolWatchConfig(
            symbol="EURUSD",
            timeframes=[
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M5, candle_count=100),
                TimeframeConfig(timeframe=mt5.TIMEFRAME_H1, candle_count=50)
            ]
        ),
        SymbolWatchConfig(
            symbol="USDJPY",
            timeframes=[
                TimeframeConfig(timeframe=mt5.TIMEFRAME_M15, candle_count=60)
            ]
        )
    ]

    # 2. Setup dependencies
    connector = MT5Connector()
    try:
        await connector.connect()
        if not connector.connected:
            return

        repo = MT5MarketDataRepository(connector)
        indicator_svc = PandasTAIndicatorService()

        # 3. Initialize the builder
        builder = AgentContextBuilder(repo, indicator_svc)

        # 4. Build the context
        full_agent_context = await builder.build_context(watch_list)

        # 5. Inspect the result (this object is now ready for the agent)
        logger.info("\n--- AGENT CONTEXT BUILD COMPLETE ---")
        logger.info(f"Account Currency: {full_agent_context.account_state.account_currency}")
        logger.info(f"Open Positions: {len(full_agent_context.open_positions)}")

        for symbol, data in full_agent_context.market_context.items():
            logger.info(f"\nSymbol: {symbol}")
            logger.info(f"  Volume Step: {data.symbol_info.volume_step}")
            logger.info(f"  Tick Value: {data.symbol_info.trade_tick_value}")
            for tf_data in data.market_data:
                logger.info(f"  Timeframe: {tf_data.timeframe}")
                logger.info(f"    Candles Fetched: {len(tf_data.candles)}")
                if tf_data.indicators.rsi:
                    logger.info(f"    Latest RSI: {tf_data.indicators.rsi[-1]}")
                if tf_data.indicators.atr:
                    logger.info(f"    Latest ATR: {tf_data.indicators.atr[-1]}")

    except Exception as e:
        logger.exception(f"Example failed: {e}")
    finally:
        if connector.connected:
            await connector.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
