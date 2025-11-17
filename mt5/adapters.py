"""
MT5 Infrastructure Adapters
---------------------------

This file contains the concrete implementations (Adapters) of the
ports defined in `ports.py`.

These classes depend directly on the `MetaTrader5` library and the
`MT5Connector`. They are responsible for translating the application's
requests (e.g., `get_symbol_info`) into specific MT5 library calls
and mapping the MT5 results back to the application's domain models.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

# Try to import pandas for get_historical_trade
try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas not found. get_historical_trade may not work.")

from .connector import MT5Connector
from .ports import IMarketDataRepository, ITradeExecutionService
from .domain import (
    SymbolInfo,
    MarketCandle,
    AccountState,
    PositionInfo,
    PositionHistoryInfo,
    TickQuote,
    OrderResult
)
from .utils import round_to_step, safe_comment

logger = logging.getLogger(__name__)


# --- Market Data Adapter ---

class MT5MarketDataRepository(IMarketDataRepository):
    """
    Concrete implementation that fetches market and account data from MT5.
    """

    def __init__(self, connector: MT5Connector):
        self.mt5 = connector.mt5
        self._connector = connector

    async def _run(self, blocking_func, *args, **kwargs):
        """Alias for the connector's executor runner."""
        return await self._connector.run_in_executor(blocking_func, *args, **kwargs)

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Fetches and normalizes symbol information."""
        raw = await self._run(self.mt5.symbol_info, symbol)
        if raw is None:
            raise ValueError(f"Symbol {symbol} not found or not available.")

        # Helper to safely get attributes
        def _get(attr, default):
            return getattr(raw, attr, default)

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
        raw_tuples = await self._run(
            self.mt5.copy_rates_from_pos,
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
        raw = await self._run(self.mt5.account_info)
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

    async def get_open_positions(self) -> List[PositionInfo]:
        """Fetches all currently open positions."""
        raw_positions = await self._run(self.mt5.positions_get)
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
        """
        if pd is None:
            logger.error("pandas is not installed. Cannot fetch historical trades.")
            return None

        try:
            # We fetch the last 10 days of history.
            from_date = datetime.now(timezone.utc) - pd.Timedelta(days=10)
            to_date = datetime.now(timezone.utc) + pd.Timedelta(days=1)

            # Fetch deals based on the position_id (which is the ticket)
            deals = await self._run(
                self.mt5.history_deals_get,
                from_date,
                to_date
            )

            if deals is None:
                logger.warning(f"Could not fetch deal history for ticket {ticket}")
                return None

            related_deals = [d for d in deals if d.position_id == ticket]
            if not related_deals:
                logger.warning(f"No deals found for position_id {ticket} in history.")
                return None

            # entry_type 0 = DEAL_ENTRY_IN (open)
            # entry_type 1 = DEAL_ENTRY_OUT (close)
            open_deal = next((d for d in related_deals if d.entry == 0), None)
            close_deal = next((d for d in related_deals if d.entry == 1), None)

            if not open_deal or not close_deal:
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
        tick = await self._run(self.mt5.symbol_info_tick, symbol)
        if not tick:
            return None
        return TickQuote(symbol, tick.bid, tick.ask, tick.time)


# --- Trade Execution Adapter ---

class MT5TradeExecutionService(ITradeExecutionService):
    """
    Concrete implementation for trade execution.
    """

    def __init__(self,
                 connector: MT5Connector,
                 repo: IMarketDataRepository,
                 default_deviation: int = 100,
                 magic_number: int = 123456):
        self.mt5 = connector.mt5
        self._connector = connector
        self._repo = repo
        self._deviation = default_deviation  # max slippage in points
        self._magic = magic_number

    async def _run(self, blocking_func, *args, **kwargs):
        """Alias for the connector's executor runner."""
        return await self._connector.run_in_executor(blocking_func, *args, **kwargs)

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

        tick = await self._repo.get_tick(symbol)
        if tick is None:
            raise ValueError(f"Could not fetch tick data for {symbol}")

        price = tick.ask if action == "buy" else tick.bid
        order_type = self.mt5.ORDER_TYPE_BUY if action == "buy" else self.mt5.ORDER_TYPE_SELL

        info = await self._repo.get_symbol_info(symbol)

        lot = round_to_step(lot, info.volume_step)
        if lot < info.volume_min:
            logger.warning(f"Lot size {lot} < min {info.volume_min}. Adjusting to min.")
            lot = info.volume_min
        if lot > info.volume_max:
            logger.warning(f"Lot size {lot} > max {info.volume_max}. Adjusting to max.")
            lot = info.volume_max

        sl = round(stop_loss, info.digits) if stop_loss is not None else 0.0
        tp = round(take_profit, info.digits) if take_profit is not None else 0.0

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": order_type,
            "price": float(price),
            "sl": sl,
            "tp": tp,
            "deviation": int(self._deviation),
            "magic": self._magic,
            "comment": safe_comment(comment) or "auto-agent",
            "type_filling": self.mt5.ORDER_FILLING_FOK
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
            result = await self._run(lambda: self.mt5.order_send(**request))

            return self._parse_result(result)

        except Exception as exc:
            logger.exception(f"Error in place_market_order: {exc}")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=None)

    async def close_position(self,
                             position: PositionInfo,
                             comment: Optional[str] = None) -> OrderResult:
        """Closes an existing open position."""
        try:
            symbol = position.symbol
            volume = position.volume
            ticket = position.ticket

            # Determine opposite order type
            order_type = self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY

            tick = await self._repo.get_tick(symbol)
            if tick is None:
                raise ValueError(f"Could not fetch tick data for {symbol}")

            price = tick.bid if order_type == self.mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "position": int(ticket),  # Specify the position ticket to close
                "price": float(price),
                "deviation": int(self._deviation),
                "magic": self._magic,
                "comment": safe_comment(comment) or "auto-agent-close",
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }

            logger.info(f"Sending close request: {request}")
            result = await self._run(lambda: self.mt5.order_send(**request))

            return self._parse_result(result)

        except Exception as exc:
            logger.exception(f"Error in close_position: {exc}")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=None)

    # --- NEW FEATURE IMPLEMENTATIONS ---

    async def partial_close_position(self,
                                     ticket: int,
                                     lot_to_close: float,
                                     comment: Optional[str] = None) -> OrderResult:
        """Partially closes an existing open position by a specified lot amount."""
        try:
            # 1. Get position details
            raw_pos = await self._run(self.mt5.positions_get, ticket=ticket)
            if not raw_pos:
                return OrderResult(success=False, ticket=None, comment=f"Position {ticket} not found.")

            position = raw_pos[0]
            symbol = position.symbol

            # 2. Get symbol info for rounding
            info = await self._repo.get_symbol_info(symbol)

            # 3. Validate and round lot size
            lot = round_to_step(lot_to_close, info.volume_step)
            if lot <= 0:
                return OrderResult(success=False, ticket=None,
                                   comment=f"Lot to close ({lot_to_close}) rounded to {lot}, which is invalid.")
            if lot > position.volume:
                logger.warning(
                    f"Lot to close ({lot}) > position volume ({position.volume}). Clamping to position volume.")
                lot = round_to_step(position.volume, info.volume_step)

            # 4. Determine opposite order type
            order_type = self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY

            # 5. Get current price
            tick = await self._repo.get_tick(symbol)
            if tick is None:
                raise ValueError(f"Could not fetch tick data for {symbol}")

            price = tick.bid if order_type == self.mt5.ORDER_TYPE_SELL else tick.ask

            # 6. Build request
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot),
                "type": order_type,
                "position": int(ticket),
                "price": float(price),
                "deviation": int(self._deviation),
                "magic": self._magic,
                "comment": safe_comment(comment) or "auto-agent-partial",
                "type_filling": self.mt5.ORDER_FILLING_FOK,
            }

            logger.info(f"Sending partial close request: {request}")
            result = await self._run(lambda: self.mt5.order_send(**request))

            return self._parse_result(result)

        except Exception as exc:
            logger.exception(f"Error in partial_close_position: {exc}")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=None)

    async def update_position_sl(self, ticket: int, stop_loss: float) -> OrderResult:
        """Updates the stop loss for an existing open position."""
        return await self._modify_position_sltp(ticket, sl=stop_loss)

    async def update_position_tp(self, ticket: int, take_profit: float) -> OrderResult:
        """Updates the take profit for an existing open position."""
        return await self._modify_position_sltp(ticket, tp=take_profit)

    async def _modify_position_sltp(self,
                                    ticket: int,
                                    sl: Optional[float] = None,
                                    tp: Optional[float] = None) -> OrderResult:
        """
        Internal helper to modify SL/TP of an open position.
        Note: MT5's TRADE_ACTION_SLTP requires *both* values.
        We must fetch the existing position to keep the other value unchanged.
        """
        try:
            # 1. Get current position data
            raw_pos = await self._run(self.mt5.positions_get, ticket=ticket)
            if not raw_pos:
                return OrderResult(success=False, ticket=None, comment=f"Position {ticket} not found.")

            position = raw_pos[0]

            # 2. Get symbol info for rounding
            info = await self._repo.get_symbol_info(position.symbol)

            # 3. Determine new SL/TP values
            new_sl = sl if sl is not None else position.sl
            new_tp = tp if tp is not None else position.tp

            # 4. Round to correct digits
            new_sl = round(new_sl, info.digits) if new_sl != 0.0 else 0.0
            new_tp = round(new_tp, info.digits) if new_tp != 0.0 else 0.0

            # 5. Build request
            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "position": int(ticket),
                "sl": float(new_sl),
                "tp": float(new_tp),
                "magic": self._magic,
            }

            logger.info(f"Sending SL/TP modify request: {request}")
            result = await self._run(lambda: self.mt5.order_send(**request))

            return self._parse_result(result)

        except Exception as exc:
            logger.exception(f"Error in _modify_position_sltp: {exc}")
            return OrderResult(success=False, ticket=None, comment=str(exc), raw=None)

    # --- Private Helpers ---

    def _parse_result(self, result: Any) -> OrderResult:
        """Helper to parse the complex result from mt5.order_send."""
        if result is None:
            last_err = self.mt5.last_error()
            logger.error(f"Order send failed, result was None. Last Error: {last_err}")
            return OrderResult(success=False, ticket=None, comment=f"mt5.order_send returned None. Error: {last_err}",
                               raw=None)

        raw_dict = result._asdict() if hasattr(result, "_asdict") else dict(result)

        retcode = getattr(result, "retcode", -1)
        comment = getattr(result, "comment", "Unknown error")

        # 10009 is "Request executed"
        # 10008 is "Request processing" (also good)
        if retcode in (10009, 10008):
            # For SL/TP modify, ticket is in `order`
            # For new orders, ticket is in `order`
            # For close orders, ticket is in `order`
            ticket = int(getattr(result, "order", 0))
            if ticket == 0:
                # Sometimes the ticket is in the request object
                # FIX: result.request is an object (TradeRequest), not a dict.
                # We must use getattr() to access its 'position' attribute.
                request_obj = getattr(result, "request", None)
                if request_obj:
                    ticket = int(getattr(request_obj, "position", 0))

            logger.info(f"Order success, retcode: {retcode}, ticket: {ticket}, comment: {comment}")
            return OrderResult(success=True, ticket=ticket, comment=comment, raw=raw_dict)

        logger.error(f"Order failed, retcode: {retcode}, comment: {comment}, raw: {raw_dict}")
        return OrderResult(success=False, ticket=None, comment=f"retcode: {retcode} - {comment}", raw=raw_dict)