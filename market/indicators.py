"""
Technical Indicator Services
----------------------------

This file contains implementations of the `ITechnicalIndicatorService`
port.

These services are responsible for performing technical analysis
calculations on `MarketCandle` data.
"""

import logging
from typing import List

# Try to import pandas and pandas-ta
try:
    import pandas as pd
    import pandas_ta as ta
except ImportError:
    pd = None
    ta = None
    logging.warning("pandas and pandas-ta not found. Indicator service will not work.")
    logging.warning("Please run: pip install pandas pandas-ta")

import asyncio
from .domain import MarketCandle, TechnicalIndicators
from .ports import ITechnicalIndicatorService

logger = logging.getLogger(__name__)


class PandasTAIndicatorService(ITechnicalIndicatorService):
    """
    Calculates TA indicators using the pandas-ta library.
    This service is async to allow for non-blocking I/O if pandas-ta
    were ever to become async, but here it runs calculations in the
    default executor (via asyncio.to_thread) to avoid blocking the event loop.
    """
    def __init__(self):
        if pd is None or ta is None:
            raise ImportError("`pandas` and `pandas-ta` must be installed to use this service.")

    async def calculate_indicators(self,
                                     candles: List[MarketCandle],
                                     fast_ema_period: int = 12,
                                     slow_ema_period: int = 26,
                                     macd_signal_period: int = 9,
                                     rsi_period: int = 14,
                                     atr_period: int = 14) -> TechnicalIndicators:

        if len(candles) < max(slow_ema_period, rsi_period, atr_period):
            logger.warning(f"Not enough candles ({len(candles)}) to calculate indicators.")
            return TechnicalIndicators()

        # Run the blocking pandas logic in a separate thread
        return await asyncio.to_thread(
            self._calculate_sync,
            candles,
            fast_ema_period,
            slow_ema_period,
            macd_signal_period,
            rsi_period,
            atr_period
        )

    def _calculate_sync(self,
                        candles: List[MarketCandle],
                        fast_ema_period: int,
                        slow_ema_period: int,
                        macd_signal_period: int,
                        rsi_period: int,
                        atr_period: int) -> TechnicalIndicators:
        """Synchronous calculation logic to be run in a thread."""
        try:
            # 1. Convert list of dataclasses to pandas DataFrame
            df = pd.DataFrame([vars(c) for c in candles])
            if df.empty:
                return TechnicalIndicators()

            df.set_index('time', inplace=True)
            # Ensure correct dtypes
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])

            # 2. Calculate Indicators using pandas-ta
            df.ta.ema(length=fast_ema_period, append=True, col_names='ema_fast')
            df.ta.ema(length=slow_ema_period, append=True, col_names='ema_slow')
            df.ta.macd(fast=fast_ema_period, slow=slow_ema_period, signal=macd_signal_period, append=True)
            df.ta.rsi(length=rsi_period, append=True, col_names='rsi')
            df.ta.atr(length=atr_period, append=True, col_names='atr')

            # 3. Clean and convert back to lists
            df.fillna(0.0, inplace=True) # Replace NaNs with 0.0 for the agent

            # Dynamically get MACD column names
            macd_col = f'MACD_{fast_ema_period}_{slow_ema_period}_{macd_signal_period}'
            signal_col = f'MACDs_{fast_ema_period}_{slow_ema_period}_{macd_signal_period}'

            return TechnicalIndicators(
                ema_fast=df['ema_fast'].tolist(),
                ema_slow=df['ema_slow'].tolist(),
                macd=df[macd_col].tolist() if macd_col in df.columns else [],
                macd_signal=df[signal_col].tolist() if signal_col in df.columns else [],
                rsi=df['rsi'].tolist(),
                atr=df['atr'].tolist()
            )
        except Exception as e:
            logger.exception(f"Error during indicator calculation: {e}")
            return TechnicalIndicators()