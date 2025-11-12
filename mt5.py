"""
mt5_module.py
A professional MT5 integration module designed for use with an autonomous trading agent.
Principles: SOLID, Domain-Driven Design (DDD), asyncio, OOP.
Dependencies: MetaTrader5 (pip install MetaTrader5), pandas-ta (pip install pandas-ta)

Revision 3:
- Added `PositionInfo` dataclass for open positions.
- Added `TechnicalIndicators` dataclass for TA data.
- Added `get_open_positions` to the data repository.
- Added `ITechnicalIndicatorService` and `PandasTAIndicatorService` to
  calculate EMA, MACD, RSI, and ATR from candle data.
"""

from __future__ import annotations
import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple, Dict, Any
import math


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Try to import MetaTrader5 but handle the case where it's unavailable
try:
    import MetaTrader5 as mt5  # type: ignore
except Exception as e:
    mt5 = None  # type: ignore
    logger.warning(f"MetaTrader5 package not found or failed to import: {e}")

# Try to import pandas-ta
try:
    import pandas as pd
    import pandas_ta as ta
except ImportError:
    pd = None
    ta = None
    logger.warning("pandas and pandas-ta not found. Indicator service will not work.")
    logger.warning("Please run: pip install pandas pandas-ta")



# ----------------------------- Domain Layer (DDD) -----------------------------

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
    symbol: str
    bid: float
    ask: float
    time: float

@dataclass
class MarketCandle:
    time: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: Optional[int] = None

@dataclass
class AccountState:
    login: int
    balance: float
    equity: float
    free_margin: float
    margin_level: Optional[float]
    currency: str

@dataclass
class OrderResult:
    success: bool
    ticket: Optional[int]
    comment: Optional[str]
    raw: Optional[Dict[str, Any]] = None

# --- NEW DATACLASSES FOR REVISION 3 ---

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


# --------------------------- Ports / Interfaces (SOLID) -------------------------

class IMarketDataRepository(Protocol):
    """Interface for fetching market and account data."""

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        ...

    async def get_last_candles(self, symbol: str, timeframe: int, count: int) -> List[MarketCandle]:
        ...

    async def get_account_state(self) -> AccountState:
        ...

    async def get_open_positions(self) -> List[PositionInfo]:
        """
        NEW: Fetches all open positions from the account.
        """
        ...

    async def get_historical_trade(self, ticket: int) -> Optional[PositionHistoryInfo]:
        """
        Fetches the deal history for a specific position ticket
        to find its closing details.
        """
        ...

    async def get_tick(self, symbol: str) -> Optional[TickQuote]:
        ...

class ITradeExecutionService(Protocol):
    async def place_market_order(self,
                                 symbol: str,
                                 action: str,  # 'buy' or 'sell'
                                 lot: float,
                                 stop_loss: Optional[float],
                                 take_profit: Optional[float],
                                 comment: Optional[str]) -> OrderResult:
        ...

    async def close_position(self,
                             position: PositionInfo,
                             comment: Optional[str]) -> OrderResult: # <-- MODIFIED
        ...

# --- NEW INTERFACE FOR REVISION 3 ---

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
        ...

# -------------------------- Application / Infrastructure ------------------------

class MT5Connector:
    """
    Handles the lifecycle of the MetaTrader5 connection.
    Uses ThreadPoolExecutor to run blocking C-lib calls asynchronously.
    """
    def __init__(self):

        self._connected = False
        self._lock = asyncio.Lock()
        # ThreadPoolExecutor for converting blocking MT5 calls to async-friendly calls
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="MT5Worker")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

    async def connect(self) -> None:
        """Initializes the MT5 connection."""
        async with self._lock:
            if self._connected:
                return
            if mt5 is None:
                logger.error("MetaTrader5 package not available. Cannot connect.")
                raise ImportError("MetaTrader5 package not installed.")

            loop = asyncio.get_running_loop()

            # mt5.initialize() is blocking
            ok = await loop.run_in_executor(
                self._executor,
                mt5.initialize
            )

            if not ok:
                await self._log_mt5_error("MT5 initialize failed")
                raise ConnectionError("Failed to initialize MT5. Check path.")



            self._connected = True
            logger.info("MT5 initialized and login successful.")

    async def disconnect(self) -> None:
        """Shuts down the MT5 connection."""
        async with self._lock:
            if not self._connected or mt5 is None:
                self._connected = False
                return

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, mt5.shutdown)
            self._connected = False
            logger.info("MT5 shutdown completed.")

    @property
    def connected(self) -> bool:
        return self._connected

    def executor(self):
        """Provides access to the executor for services to use."""
        return self._executor

    async def _log_mt5_error(self, message: str):
        """Helper to log the last MT5 error."""
        if mt5 is None:
            logger.error(message)
            return

        loop = asyncio.get_running_loop()
        last_error = await loop.run_in_executor(self._executor, mt5.last_error)
        logger.error(f"{message}: {last_error}")

# -------------------------- Concrete Repository / Service -----------------------

class MT5MarketDataRepository(IMarketDataRepository):
    """
    Concrete implementation that fetches market and account data from MT5.
    """
    def __init__(self, connector: MT5Connector):
        if mt5 is None:
            raise ImportError("MetaTrader5 package not installed.")
        self._connector = connector
        self._executor = self._connector.executor()

    async def _run_in_executor(self, blocking_func, *args, **kwargs):
        """Helper to run a blocking function in the connector's thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: blocking_func(*args, **kwargs)
        )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetches and normalizes symbol information."""
        raw = await self._run_in_executor(mt5.symbol_info, symbol)
        if raw is None:
            raise ValueError(f"Symbol {symbol} not found or not available.")

        # Helper to safely get attributes
        def _get(attr, default):
            return getattr(raw, attr, default)

        # MT5's 'point' is the *actual* tick size (e.g., 1e-05)
        # We use trade_tick_value which is the value of one 'point'
        # move for 1.0 lot in account currency.
        point = float(_get("point", 10 ** (-_get("digits", 5))))

        return SymbolInfo(
            name=symbol,
            point=point,
            digits=int(_get("digits", 5)),
            trade_tick_value=float(_get("trade_tick_value", point * _get("trade_contract_size", 100000))),
            trade_contract_size=float(_get("trade_contract_size", 100000)),
            swap_long=float(_get("swap_long", 0.0)),
            swap_short=float(_get("swap_short", 0.0)),
            spread=int(_get("spread", 0)),
            volume_min=float(_get("volume_min", 0.01)),
            volume_max=float(_get("volume_max", 100.0)),
            volume_step=float(_get("volume_step", 0.01))
        )

    async def get_last_candles(self, symbol: str, timeframe: int, count: int) -> List[MarketCandle]:
        """Fetches the last 'count' candles for a symbol."""
        raw_tuples = await self._run_in_executor(
            mt5.copy_rates_from_pos,
            symbol, timeframe, 0, count
        )

        if raw_tuples is None or len(raw_tuples) == 0:
            logger.warning(f"No candle data returned for {symbol} T={timeframe}")
            return []

        candles = []
        for r in raw_tuples:
            candles.append(MarketCandle(
                time=datetime.fromtimestamp(int(r[0]), tz=timezone.utc),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                tick_volume=int(r[5]),
                spread=int(r[6])
            ))
        return candles

    async def get_account_state(self) -> AccountState:
        """Fetches the current account state."""
        raw = await self._run_in_executor(mt5.account_info)
        if raw is None:
            raise RuntimeError("Failed to fetch account information from MT5.")

        return AccountState(
            login=int(raw.login),
            balance=float(raw.balance),
            equity=float(raw.equity),
            free_margin=float(getattr(raw, "margin_free", 0.0)),
            margin_level=float(getattr(raw, "margin_level", 0.0)) or None,
            currency=str(getattr(raw, "currency", "USD"))
        )

    # --- NEW METHOD FOR REVISION 3 ---
    async def get_open_positions(self) -> List[PositionInfo]:
        """Fetches all currently open positions."""
        raw_positions = await self._run_in_executor(mt5.positions_get)
        if raw_positions is None:
            logger.warning("mt5.positions_get() returned None.")
            return []

        positions = []
        for pos in raw_positions:
            try:
                positions.append(PositionInfo(
                    ticket=int(pos.ticket),
                    symbol=str(pos.symbol),
                    type=int(pos.type),
                    volume=float(pos.volume),
                    open_price=float(pos.price_open),
                    open_time=datetime.fromtimestamp(int(pos.time), tz=timezone.utc),
                    profit=float(getattr(pos, "profit", 0.0)),
                    swap=float(getattr(pos, "swap", 0.0)),
                    comment=str(getattr(pos, "comment", "")),
                    current_price=float(getattr(pos, "price_current", 0.0)),
                    stop_loss=float(getattr(pos, "sl", 0.0)),
                    take_profit=float(getattr(pos, "tp", 0.0))
                ))
            except Exception as e:
                logger.exception(f"Error parsing position {getattr(pos, 'ticket', 'N/A')}: {e}")

        return positions

    async def get_historical_trade(self, ticket: int) -> Optional[PositionHistoryInfo]:
        """
        Fetches the deal history for a specific position ticket
        to find its closing details.

        Note: This assumes the 'ticket' is the position ID. MT5 deals
        are linked by 'position_id'.
        """
        try:
            # We fetch the last 10 days of history. This should be enough.
            from_date = datetime.now(timezone.utc) - pd.Timedelta(days=10)
            to_date = datetime.now(timezone.utc) + pd.Timedelta(days=1)

            # Fetch deals based on the position_id (which is the ticket)
            deals = await self._run_in_executor(
                mt5.history_deals_get,
                from_date,
                to_date
            )

            if deals is None:
                logger.warning(f"Could not fetch deal history for ticket {ticket}")
                return None

            # Find deals related to this position
            # A position is formed by one or more "in" deals and "out" deals.
            # We need to find the "out" deal that closed this position.

            related_deals = [d for d in deals if d.position_id == ticket]

            if not related_deals:
                logger.warning(f"No deals found for position_id {ticket} in history.")
                return None

            # Find the "in" deal (open) and "out" deal (close)
            # entry_type 0 = DEAL_ENTRY_IN (open)
            # entry_type 1 = DEAL_ENTRY_OUT (close)

            open_deal = next((d for d in related_deals if d.entry == 0), None)
            close_deal = next((d for d in related_deals if d.entry == 1), None)

            if not open_deal or not close_deal:
                # This could happen if the position is partially closed.
                # For simplicity, we only handle full 1-in-1-out trades.
                logger.warning(f"Could not find full open/close deals for {ticket}.")
                return None

            return PositionHistoryInfo(
                ticket=ticket,
                symbol=str(close_deal.symbol),
                type=int(open_deal.type),  # 0=BUY, 1=SELL
                volume=float(close_deal.volume),
                open_price=float(open_deal.price),
                close_price=float(close_deal.price),
                open_time=datetime.fromtimestamp(int(open_deal.time), tz=timezone.utc),
                close_time=datetime.fromtimestamp(int(close_deal.time), tz=timezone.utc),
                profit=float(close_deal.profit),
                swap=float(close_deal.swap),
                comment=str(close_deal.comment)
            )

        except Exception as e:
            logger.exception(f"Error fetching history for ticket {ticket}: {e}")
            return None

    async def get_tick(self, symbol: str) -> Optional[TickQuote]:
        """
        Returns the latest live tick (bid/ask) for the given symbol.
        """
        def _fetch_tick():
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None
            return TickQuote(symbol, tick.bid, tick.ask, tick.time)

        # Run MT5 call in a thread executor
        return await asyncio.to_thread(_fetch_tick)


class MT5TradeExecutionService(ITradeExecutionService):
    """
    Concrete implementation for trade execution.
    """

    def __init__(self,
                 connector: MT5Connector,
                 repo: IMarketDataRepository,
                 default_deviation: int = 100):
        if mt5 is None:
            raise ImportError("MetaTrader5 package not installed.")
        self._connector = connector
        self._repo = repo
        self._deviation = default_deviation  # max slippage in points
        self._executor = self._connector.executor()

    async def _run_in_executor(self, blocking_func, *args, **kwargs):
        """Helper to run a blocking function in the connector's thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: blocking_func(*args, **kwargs)
        )

    async def _prepare_order_request(self,
                                     symbol: str,
                                     action: str,
                                     lot: float,
                                     stop_loss: Optional[float],
                                     take_profit: Optional[float],
                                     comment: Optional[str]) -> Dict[str, Any]:
        """Builds the MT5 order request dictionary."""

        if action not in ("buy", "sell"):
            raise ValueError("action must be 'buy' or 'sell'")

        # Fetch live price
        tick = await self._run_in_executor(mt5.symbol_info_tick, symbol)
        if tick is None:
            raise ValueError(f"Could not fetch tick data for {symbol}")

        price = tick.ask if action == "buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL

        # Fetch symbol info for validation
        info = await self._repo.get_symbol_info(symbol)

        # 1. Round Lot Size
        lot = round_to_step(lot, info.volume_step)

        # 2. Validate Lot Size
        if lot < info.volume_min:
            logger.warning(f"Lot size {lot} < min {info.volume_min}. Adjusting to min.")
            lot = info.volume_min
        if lot > info.volume_max:
            logger.warning(f"Lot size {lot} > max {info.volume_max}. Adjusting to max.")
            lot = info.volume_max

        # 3. Normalize SL/TP
        sl = round(stop_loss, info.digits) if stop_loss is not None else 0.0
        tp = round(take_profit, info.digits) if take_profit is not None else 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": sl,
            "tp": tp,
            "deviation": int(self._deviation),
            "magic": 123456,  # Agent's magic number
            "comment": self.safe_comment(comment) or "auto-agent",
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        return request

    async def place_market_order(self,
                                 symbol: str,
                                 action: str,
                                 lot: float,
                                 stop_loss: Optional[float],
                                 take_profit: Optional[float],
                                 comment: Optional[str] = None) -> OrderResult:
        """Places a new market order."""
        try:
            request = await self._prepare_order_request(
                symbol, action, lot, stop_loss, take_profit, comment
            )

            logger.info(f"Sending order request: {request}")
            result = await self._run_in_executor(lambda: mt5.order_send(**request))

            return self._parse_result(result)

        except Exception as exc:
            logger.exception(f"Error in place_market_order: {exc}")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=None)

    async def close_position(self,
                             position: PositionInfo,  # <-- MODIFIED
                             comment: Optional[str] = None) -> OrderResult:  # <-- MODIFIED
        """Closes an existing open position."""
        try:
            symbol = position.symbol
            volume = position.volume
            ticket = position.ticket

            # Determine opposite order type
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            # Get live price
            tick = await self._run_in_executor(mt5.symbol_info_tick, symbol)
            if tick is None:
                raise ValueError(f"Could not fetch tick data for {symbol}")

            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            logger.info(comment)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "position": int(ticket),  # Specify the position ticket to close
                "price": float(price),
                "deviation": int(self._deviation),
                "magic": 123456,  # Agent's magic number
                "comment":  "auto-agent-close",
                "type_filling": mt5.ORDER_FILLING_FOK, # FOK is standard
            }

            logger.info(f"Sending close request: {request}")
            result = await self._run_in_executor(lambda: mt5.order_send(**request))

            return self._parse_result(result)

        except Exception as exc:
            logger.exception(f"Error in close_position: {exc}")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=None)

    def _parse_result(self, result: Any) -> OrderResult:
        """Helper to parse the complex result from mt5.order_send."""
        if result is None:
            logger.info(mt5.last_error())
            return OrderResult(success=False, ticket=None, comment="mt5.order_send returned None", raw=None)

        # Convert namedtuple to dict if necessary
        raw_dict = result._asdict() if hasattr(result, "_asdict") else dict(result)

        retcode = getattr(result, "retcode", -1)
        comment = getattr(result, "comment", "Unknown error")

        # 10009 is "Request executed"
        # 10008 is "Request processing" (also good)
        if retcode in (10009, 10008):
            ticket = int(getattr(result, "order", 0))
            if ticket > 0:
                logger.info(f"Order success, retcode: {retcode}, ticket: {ticket}")
                return OrderResult(success=True, ticket=ticket, comment=comment, raw=raw_dict)

        logger.error(f"Order failed, retcode: {retcode}, comment: {comment}, raw: {raw_dict}")
        return OrderResult(success=False, ticket=None, comment=f"retcode: {retcode} - {comment}", raw=raw_dict)

    def safe_comment(self, comment: str) -> str:
        import re
        """Ensure MT5 comment field is ASCII-only and <=31 bytes."""
        if not comment:
            return "auto-agent"
        # Convert to str in case it’s not
        comment = str(comment)
        # Remove non-printable/non-ASCII characters
        comment = re.sub(r"[^\x20-\x7E]", "", comment)
        # Truncate to max 31 bytes (safe for UTF-8)
        while len(comment.encode("ascii", "ignore")) > 31:
            comment = comment[:-1]
        return comment
# --- NEW SERVICE FOR REVISION 3 ---

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
        # EMA
        df.ta.ema(length=fast_ema_period, append=True, col_names='ema_fast')
        df.ta.ema(length=slow_ema_period, append=True, col_names='ema_slow')

        # MACD
        # This appends 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
        df.ta.macd(fast=fast_ema_period, slow=slow_ema_period, signal=macd_signal_period, append=True)

        # RSI
        df.ta.rsi(length=rsi_period, append=True, col_names='rsi')

        # ATR
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

# -------------------------- Utilities & Helpers --------------------------------
def round_to_step(value: float, step: float) -> float:
    """
    Rounds a value to the nearest valid step size.
    e.g., round_to_step(0.123, 0.01) -> 0.12
    e.g., round_to_step(0.128, 0.01) -> 0.13
    e.g., round_to_step(0.12, 0.05) -> 0.10
    e.g., round_to_step(0.14, 0.05) -> 0.15
    """
    if step <= 0:
        return value

    # Use math.fsum for precision
    # (value / step)
    # round(...)
    # ... * step

    # We must account for floating point inaccuracies
    # e.g., 0.01 -> "0.01" -> 2 decimal places

    try:
        # Get number of decimal places from the step
        step_str = str(step)
        if 'e-' in step_str:
            decimals = int(step_str.split('e-')[-1])
        elif '.' in step_str:
            decimals = len(step_str.split('.')[-1])
        else:
            decimals = 0

        return round(round(value / step) * step, decimals)

    except Exception:
        # Fallback for complex steps
        return round(value / step) * step


def calculate_lot_size(account_balance: float,
                         risk_percent: float,
                         stop_loss_pips: float,
                         pip_value_per_lot: float,
                         volume_step: float) -> float:
    """
    Deterministic lot size calculation based on the agent's prompt.

    Args:
        account_balance (float): Current account balance.
        risk_percent (float): Desired risk (e.g., 0.01 for 1%).
        stop_loss_pips (float): Stop loss distance in pips (e.g., 50.0).
        pip_value_per_lot (float): The value of 1 pip for 1.0 lot.
                                   FOR MT5, THIS IS `symbol_info.trade_tick_value`
                                   AND `stop_loss_pips` MUST BE IN `point` units.
        volume_step (float): The minimum lot step (e.g., 0.01 or 1.0).

    Returns:
        float: The correctly rounded lot size.
    """
    if stop_loss_pips <= 0 or pip_value_per_lot <= 0:
        logger.warning(f"Invalid inputs to calculate_lot_size: SL_pips={stop_loss_pips}, PipVal={pip_value_per_lot}")
        return 0.0
    if volume_step <= 0:
        raise ValueError("volume_step must be positive")

    risk_usd = Decimal(str(account_balance)) * Decimal(str(risk_percent))
    denom = Decimal(str(stop_loss_pips)) * Decimal(str(pip_value_per_lot))

    if denom == 0:
        logger.error("Lot size calculation failed: denominator is zero.")
        return 0.0

    lot = float(risk_usd / denom)

    # --- FIX 1: Correct rounding based on volume_step ---
    # Example: lot=0.25, volume_step=0.1 -> 0.2
    # Example: lot=0.25, volume_step=1.0 -> 0.0
    # Use Decimal for precision in rounding
    lot_quantized = math.floor(lot / volume_step) * volume_step

    # Return rounded to a safe number of decimals
    return round(lot_quantized, 8)

# -------------------------- Example Agent Integration ---------------------------

async def example_decision_cycle(symbol: str, timeframe_int: int):
    """
    Example of how an external decision engine can use the repository + execution service.
    This function is illustrative — integrate with your decision logic.
    """
    if mt5 is None:
        logger.error("Cannot run example: MetaTrader5 package is not installed.")
        return

    connector = MT5Connector()
    try:
        await connector.connect()
        if not connector.connected:
            logger.error("Failed to connect to MT5, example cannot run.")
            return

        repo = MT5MarketDataRepository(connector)
        exec_svc = MT5TradeExecutionService(connector)

        tick = await repo.get_tick("EURUSD")
        logger.info(f"Live EURUSD Tick {tick}")
        # --- NEW: Setup indicator service ---
        indicator_svc = None
        if pd and ta:
            indicator_svc = PandasTAIndicatorService()
        else:
            logger.error("Pandas/TA-Lib not found. Skipping indicator calculation.")
            return

        # 1. Get Account and Symbol Info (This is what the agent's "context" would be)
        account = await repo.get_account_state()
        logger.info(f"Account: {account}")

        # --- NEW: Get Open Positions ---
        positions = await repo.get_open_positions()
        logger.info(f"Open Positions: {len(positions)}")
        for pos in positions:
            logger.info(f"  -> {pos.symbol} {pos.type} {pos.volume} @ {pos.open_price} (Ticket: {pos.ticket})")

        info = await repo.get_symbol_info(symbol)
        logger.info(f"SymbolInfo: {info}")

        candles = await repo.get_last_candles(symbol, timeframe_int, 200) # count must be above max of specified periods for indicators
        if not candles or len(candles) < 2:
            logger.warning("Not enough candle data to make a decision.")
            return

        latest = candles[-1]
        previous = candles[-2]

        # --- NEW: Calculate Indicators ---
        indicators = await indicator_svc.calculate_indicators(candles, rsi_period=14)
        if indicators.rsi:
            logger.info(f"Indicators: Latest RSI: {indicators.rsi[-1]} {indicators.atr} {indicators.macd} {indicators.macd_signal} {indicators.ema_fast} {indicators.ema_slow}")

        # 2. Agent Decision Logic (Example)
        # Simple signal: if price broke previous high AND RSI not overbought
        is_overbought = indicators.rsi[-1] > 70 if indicators.rsi else False

        if latest.close > previous.high and not is_overbought:
            logger.info("Signal: Price broke previous high & not overbought. Preparing BUY order.")

            # --- AGENT'S RISK MANAGEMENT ---

            # 1. Define SL price
            # Naive SL: 1x ATR
            stop_loss_points = 0
            if indicators.atr:
                stop_loss_points = int(indicators.atr[-1] / info.point)
            else:
                stop_loss_points = 500 # Fallback

            logger.info(f"Using SL of {stop_loss_points} points based on ATR: {indicators.atr[-1] if indicators.atr else 'N/A'}")

            stop_loss_price = latest.close - (stop_loss_points * info.point)

            # 2. Define TP price
            take_profit_points = stop_loss_points * 2 # 2:1 R:R
            take_profit_price = latest.close + (take_profit_points * info.point)

            # 3. Define Risk %
            risk_percent = 0.01 # 1% risk

            # 4. Calculate Lot Size (USING THE NEW, ROBUST METHOD)

            # The agent's "pips" are MT5's "points"
            # The prompt's "Stop Loss (Pips)" is stop_loss_points
            stop_loss_in_pips = float(stop_loss_points)

            # The prompt's "Pip Value per 1.0 Lot" is info.trade_tick_value
            pip_value_per_lot = info.trade_tick_value

            # The broker's lot step is info.volume_step
            volume_step = info.volume_step

            lot = calculate_lot_size(
                account_balance=account.balance,
                risk_percent=risk_percent,
                stop_loss_pips=stop_loss_in_pips,
                pip_value_per_lot=pip_value_per_lot,
                volume_step=volume_step
            )

            logger.info(f"Calculated Lot: {lot} (Risk: {risk_percent*100}%, SL Pips: {stop_loss_in_pips}, PipVal: {pip_value_per_lot}, VolStep: {volume_step})")

            # 5. Place Order
            if lot > 0:
                result = await exec_svc.place_market_order(
                    symbol=symbol,
                    action="buy",
                    lot=lot,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    comment="example-agent-buy"
                )
                logger.info(f"Place order result: {result}")
            else:
                logger.warning("Lot size calculated to 0.0. Order not placed. (SL too wide or risk too small?)")

        else:
            logger.info("No entry signal - holding")

    except Exception as ex:
        logger.exception(f"Example run failed: {ex}")
    finally:
        if connector.connected:
            await connector.disconnect()

# ------------------------------- Module Export ---------------------------------

__all__ = [
    # Classes
    "MT5Connector",
    "MT5MarketDataRepository",
    "MT5TradeExecutionService",
    "PandasTAIndicatorService",
    # Dataclasses
    "SymbolInfo",
    "MarketCandle",
    "AccountState",
    "OrderResult",
    "PositionInfo",
    "TechnicalIndicators",
    "TickQuote",
    # Protocols
    "IMarketDataRepository",
    "ITradeExecutionService",
    "ITechnicalIndicatorService",
    # Utilities
    "calculate_lot_size",
    "round_to_step"
]

# If run as script, demonstrate module usage
if __name__ == "__main__":
    import asyncio

    # Determine MT5 timeframe constant
    # mt5.TIMEFRAME_M15 = 15
    EXAMPLE_TIMEFRAME = 15
    EXAMPLE_SYMBOL = "EURUSD"

    async def main():
        logger.info(f"--- Running MT5 Module Example for {EXAMPLE_SYMBOL} ({EXAMPLE_TIMEFRAME}) ---")
        await example_decision_cycle(EXAMPLE_SYMBOL, EXAMPLE_TIMEFRAME)
        logger.info("--- Example run complete ---")

    asyncio.run(main())

