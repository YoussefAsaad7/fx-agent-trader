"""
mt5_module.py
A professional MT5 integration module designed for use with an autonomous trading agent.
Principles: SOLID, Domain-Driven Design (DDD), asyncio, OOP.
Dependencies: MetaTrader5 (pip install MetaTrader5)

Revision 2:
- Implements `volume_step` for correct lot size calculation (Fixes crit-bug-1).
- Implements `trade_tick_value` for robust pip value (Fixes crit-bug-2).
- Removes the need for a separate PipValueCalculator class.
"""

from __future__ import annotations
import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
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


# ----------------------------- Domain Layer (DDD) -----------------------------

@dataclass(frozen=True)
class SymbolInfo:
    """
    A domain model representing all critical properties of a trading symbol.
    """
    name: str
    point: float
    digits: int
    spread: int  # Spread in integer points
    trade_contract_size: float

    # Lot Sizing & Risk (CRITICAL)
    volume_min: float  # Minimum trade volume (e.g., 0.01)
    volume_step: float  # Lot step (e.g., 0.01 or 1.0)
    trade_tick_value: float  # The value of 1.0 'point' for 1 lot in account currency

    # Informational
    swap_long: float
    swap_short: float
    currency_base: str
    currency_profit: str
    currency_margin: str


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
    account_currency: str


@dataclass
class OrderResult:
    success: bool
    ticket: Optional[int]
    comment: Optional[str]
    raw: Optional[Dict[str, Any]] = None


# --------------------------- Ports / Interfaces (SOLID) -------------------------

class IMarketDataRepository(Protocol):
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        ...

    async def get_last_candles(self, symbol: str, timeframe: int, count: int) -> List[MarketCandle]:
        ...

    async def get_account_state(self) -> AccountState:
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

    async def close_position(self, ticket: int) -> OrderResult:
        ...


# -------------------------- Application / Infrastructure ------------------------

class MT5Connector:
    """
    Responsible for initializing and shutting down the MetaTrader5 native connection.
    Keeps connection logic centralized (Single Responsibility).
    """

    def __init__(self):
        self._connected = False
        self._lock = asyncio.Lock()
        # ThreadPoolExecutor for converting blocking MT5 calls to async-friendly calls
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

    async def connect(self) -> None:
        async with self._lock:
            if self._connected:
                return
            if mt5 is None:
                logger.warning("MetaTrader5 package not available in this environment.")
                self._connected = False
                return
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(self._executor, mt5.initialize)
            if not ok:
                raise ConnectionError(
                    f"MT5 initialize failed: {mt5.last_error() if hasattr(mt5, 'last_error') else 'unknown'}")
            self._connected = True
            logger.info("MT5 initialized successfully.")

    async def disconnect(self) -> None:
        async with self._lock:
            if not self._connected:
                return
            if mt5 is None:
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
        if not self._connected:
            raise ConnectionError("MT5 is not connected. Cannot get executor.")
        return self._executor


# -------------------------- Concrete Repository / Service -----------------------

class MT5MarketDataRepository(IMarketDataRepository):
    """
    Concrete implementation that fetches market data from MT5.
    Uses run_in_executor to keep library calls non-blocking for asyncio usage.
    """

    def __init__(self, connector: MT5Connector):
        if not isinstance(connector, MT5Connector):
            raise TypeError("Expected MT5Connector instance")
        self._connector = connector

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed or not available.")

        loop = asyncio.get_running_loop()

        # 1. Get full SymbolInfo object
        raw = await loop.run_in_executor(self._connector.executor(), lambda: mt5.symbol_info(symbol))
        if raw is None:
            raise ValueError(f"Symbol {symbol} not found in MT5 terminal.")

        # 2. Get tick info for live spread
        tick = await loop.run_in_executor(self._connector.executor(), lambda: mt5.symbol_info_tick(symbol))
        if tick is None:
            logger.warning(f"Could not fetch tick for {symbol}, spread may be stale.")
            # Use spread from symbol_info if tick fails
            spread = getattr(raw, "spread", 0)
        else:
            # Live spread from tick is more accurate
            spread = getattr(tick, "spread", getattr(raw, "spread", 0))

        # 3. Extract critical fields
        point = getattr(raw, "point", 0.0)
        if point == 0.0:
            point = 10 ** (-getattr(raw, "digits", 5))

        digits = getattr(raw, "digits", 5)
        contract_size = getattr(raw, "trade_contract_size", 100000.0)

        # --- FIX 1: Volume Step ---
        volume_step = getattr(raw, "volume_step", 0.01)
        if volume_step == 0.0:
            volume_step = 0.01  # Safety fallback

        volume_min = getattr(raw, "volume_min", 0.01)
        if volume_min == 0.0:
            volume_min = 0.01  # Safety fallback

        # --- FIX 2: Tick Value ---
        # This is the value of 1.0 'point' for 1 lot in account currency
        # Use trade_tick_value_profit for longs, _loss for shorts.
        # We'll use the 'profit' one as the standard. Agent can adapt if needed.
        trade_tick_value = getattr(raw, "trade_tick_value_profit", getattr(raw, "trade_tick_value", 0.0))
        if trade_tick_value == 0.0:
            # Fallback calculation if broker field is empty (rare)
            logger.warning(f"trade_tick_value is 0 for {symbol}. Calculating fallback.")
            trade_tick_value = point * contract_size
            # This fallback is only correct if profit currency == account currency

        return SymbolInfo(
            name=symbol,
            point=float(point),
            digits=int(digits),
            spread=int(spread),
            trade_contract_size=float(contract_size),

            # Critical fields
            volume_min=float(volume_min),
            volume_step=float(volume_step),
            trade_tick_value=float(trade_tick_value),

            # Informational
            swap_long=float(getattr(raw, "swap_long", 0.0)),
            swap_short=float(getattr(raw, "swap_short", 0.0)),
            currency_base=getattr(raw, "currency_base", ""),
            currency_profit=getattr(raw, "currency_profit", ""),
            currency_margin=getattr(raw, "currency_margin", ""),
        )

    async def get_last_candles(self, symbol: str, timeframe: int, count: int) -> List[MarketCandle]:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed or not available.")

        loop = asyncio.get_running_loop()

        def _copy():
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        raw = await loop.run_in_executor(self._connector.executor(), _copy)
        if raw is None:
            raise RuntimeError(f"Failed to fetch candles for {symbol} timeframe {timeframe}: {mt5.last_error()}")

        candles: List[MarketCandle] = []
        for r in raw:
            t = datetime.fromtimestamp(int(r[0]), tz=timezone.utc)
            candles.append(MarketCandle(
                time=t,
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                tick_volume=int(r[5]),
                spread=int(r[6]) if len(r) > 6 else None
            ))
        return candles

    async def get_account_state(self) -> AccountState:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed or not available.")

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(self._connector.executor(), mt5.account_info)

        if raw is None:
            raise RuntimeError(f"Failed to fetch account information from MT5: {mt5.last_error()}")

        return AccountState(
            login=int(raw.login),
            balance=float(raw.balance),
            equity=float(raw.equity),
            free_margin=float(getattr(raw, "margin_free", 0.0)),
            margin_level=float(getattr(raw, "margin_level", 0.0)) if getattr(raw, "margin_level",
                                                                             None) is not None else None,
            account_currency=str(getattr(raw, "currency", "USD"))
        )


class MT5TradeExecutionService(ITradeExecutionService):
    """
    Trade execution service that encapsulates order placing and closing logic.
    """

    def __init__(self, connector: MT5Connector, default_deviation: int = 20):
        if not isinstance(connector, MT5Connector):
            raise TypeError("Expected MT5Connector instance")
        self._connector = connector
        self._deviation = default_deviation  # max slippage in points

    async def _get_live_price_and_min_volume(self, symbol: str, action: str) -> Tuple[float, float, float]:
        """Helper to get current price, min volume, and volume step."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed or not available.")

        loop = asyncio.get_running_loop()

        # Use symbol_info to get min volume
        si = await loop.run_in_executor(self._connector.executor(), lambda: mt5.symbol_info(symbol))
        if si is None:
            raise ValueError(f"Symbol not found: {symbol}")

        volume_min = getattr(si, "volume_min", 0.01)
        volume_step = getattr(si, "volume_step", 0.01)

        # Use symbol_info_tick to get live price
        info = await loop.run_in_executor(self._connector.executor(), lambda: mt5.symbol_info_tick(symbol))
        if info is None:
            # Fallback to symbol_info price if tick fails
            price = si.ask if action == "buy" else si.bid
        else:
            price = info.ask if action == "buy" else info.bid

        return float(price), float(volume_min), float(volume_step)

    async def place_market_order(self,
                                 symbol: str,
                                 action: str,
                                 lot: float,
                                 stop_loss: Optional[float],
                                 take_profit: Optional[float],
                                 comment: Optional[str] = None) -> OrderResult:
        if mt5 is None:
            return OrderResult(success=False, ticket=None, comment="MetaTrader5 package not installed.", raw=None)

        loop = asyncio.get_running_loop()

        try:
            price, volume_min, volume_step = await self._get_live_price_and_min_volume(symbol, action)

            # --- VALIDATION ---
            if lot < volume_min:
                return OrderResult(success=False, ticket=None, comment=f"Lot size {lot} is below minimum {volume_min}",
                                   raw=None)

            # Check if lot size adheres to volume step
            if (Decimal(str(lot)) * 10 ** 8) % (Decimal(str(volume_step)) * 10 ** 8) != 0:
                return OrderResult(success=False, ticket=None,
                                   comment=f"Lot size {lot} has invalid step. Must be multiple of {volume_step}",
                                   raw=None)

            order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL

            # Clean SL/TP
            sl_price = float(stop_loss) if stop_loss is not None and stop_loss > 0 else 0.0
            tp_price = float(take_profit) if take_profit is not None and take_profit > 0 else 0.0

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot),  # Already validated and rounded by calculate_lot_size
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": int(self._deviation),
                "magic": 123456,
                "comment": comment or "auto-trade",
                "type_filling": mt5.ORDER_FILLING_FOK,  # FOK is standard
            }
        except Exception as e:
            logger.error(f"Error preparing order request: {e}")
            return OrderResult(success=False, ticket=None, comment=f"Error preparing order: {e}", raw=None)

        def _send():
            return mt5.order_send(request)

        result = await loop.run_in_executor(self._connector.executor(), _send)

        if result is None:
            return OrderResult(success=False, ticket=None, comment=f"MT5 returned None. LastError: {mt5.last_error()}",
                               raw=None)

        try:
            retcode = getattr(result, "retcode", None)
            logger.info(f"Order send result: {result}")

            # 10009 is TRADE_RETCODE_DONE
            if retcode is not None and int(retcode) == 10009:
                ticket = getattr(result, "order", None)
                return OrderResult(success=True, ticket=int(ticket) if ticket is not None else None,
                                   comment="Order placed", raw=result._asdict())

            # If there is an error code, capture it
            comment = f"Order failed: retcode={retcode}, comment={getattr(result, 'comment', 'N/A')}"
            logger.warning(comment)
            return OrderResult(success=False, ticket=None, comment=comment, raw=result._asdict())

        except Exception as exc:
            logger.exception("Error interpreting order result")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=str(result))

    async def close_position(self, ticket: int) -> OrderResult:
        if mt5 is None:
            return OrderResult(success=False, ticket=None, comment="MetaTrader5 package not installed.", raw=None)

        loop = asyncio.get_running_loop()

        def _get_pos():
            return mt5.positions_get(ticket=ticket)

        pos = await loop.run_in_executor(self._connector.executor(), _get_pos)

        if not pos:
            return OrderResult(success=False, ticket=None, comment=f"No position found with ticket {ticket}", raw=None)

        position = pos[0]
        symbol = position.symbol
        volume = position.volume

        # close type depends on position direction
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

        # get current price
        try:
            price, _, _ = await self._get_live_price_and_min_volume(symbol,
                                                                    "sell" if order_type == mt5.ORDER_TYPE_SELL else "buy")
        except Exception as e:
            return OrderResult(success=False, ticket=None, comment=f"Failed to get closing price: {e}", raw=None)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": int(position.ticket),
            "price": price,
            "deviation": int(self._deviation),
            "magic": 123456,
            "comment": "close-position",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        def _send():
            return mt5.order_send(request)

        result = await loop.run_in_executor(self._connector.executor(), _send)

        if result is None:
            return OrderResult(success=False, ticket=None, comment=f"MT5 returned None. LastError: {mt5.last_error()}",
                               raw=None)

        retcode = getattr(result, "retcode", None)
        if retcode is not None and int(retcode) == 10009:
            return OrderResult(success=True, ticket=int(getattr(result, "order", None)), comment="Position closed",
                               raw=result._asdict())

        return OrderResult(success=False, ticket=None, comment=f"Close failed: retcode={retcode}", raw=result._asdict())


# -------------------------- Utilities & Helpers --------------------------------

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

        # 1. Get Account and Symbol Info (This is what the agent's "context" would be)
        account = await repo.get_account_state()
        logger.info(f"Account: {account}")

        info = await repo.get_symbol_info(symbol)
        logger.info(f"SymbolInfo: {info}")

        candles = await repo.get_last_candles(symbol, timeframe_int, 10)
        if not candles or len(candles) < 2:
            logger.warning("Not enough candle data to make a decision.")
            return

        latest = candles[-1]
        previous = candles[-2]

        # 2. Agent Decision Logic (Example)
        # Simple signal: if price broke previous high, buy
        if latest.close > previous.high or True:
            logger.info("Signal: Price broke previous high. Preparing BUY order.")

            # --- AGENT'S RISK MANAGEMENT ---

            # 1. Define SL price
            # Naive SL: 1x ATR (just an example, ATR not calculated here)
            # Let's use 500 points
            stop_loss_points = 500
            stop_loss_price = latest.close - (stop_loss_points * info.point)

            # 2. Define TP price
            take_profit_points = 1000  # 2:1 R:R
            take_profit_price = latest.close + (take_profit_points * info.point)

            # 3. Define Risk %
            risk_percent = 0.01  # 1% risk

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

            logger.info(
                f"Calculated Lot: {lot} (Risk: {risk_percent * 100}%, SL Pips: {stop_loss_in_pips}, PipVal: {pip_value_per_lot}, VolStep: {volume_step})")

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
                t = result.ticket
                logger.info(f"closing {t}")
                await asyncio.sleep(10)
                r = await exec_svc.close_position(ticket=t)
                logger.info(f"closing result: {r}")
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
    # Dataclasses
    "SymbolInfo",
    "MarketCandle",
    "AccountState",
    "OrderResult",
    # Protocols
    "IMarketDataRepository",
    "ITradeExecutionService",
    # Utilities
    "calculate_lot_size",
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
