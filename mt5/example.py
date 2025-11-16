"""
Example Usage Script for the MT5 Package

This script demonstrates how to import and use the refactored `mt5` package
to connect, fetch data, calculate indicators, and manage trades.

It also includes examples of the new features:
- Updating Stop Loss
- Updating Take Profit
- Partial Position Closing
"""

import asyncio
import logging
import random
import time

# --- Setup Project-Level Logging ---
# This setup is important. The mt5 package itself doesn't configure
# logging, so the application using it should.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Import from the new mt5 package ---
# Notice all imports are from the top-level 'mt5' package
from mt5 import (
    MT5Connector,
    MT5MarketDataRepository,
    MT5TradeExecutionService,
    PandasTAIndicatorService,
    IMarketDataRepository,
    ITradeExecutionService,
    ITechnicalIndicatorService,
    calculate_lot_size,
    OrderResult
)

# --- Example Constants ---
EXAMPLE_TIMEFRAME = 15  # M15
EXAMPLE_SYMBOL = "EURUSD"


async def example_trade_logic(repo: IMarketDataRepository,
                              exec_svc: ITradeExecutionService,
                              indicator_svc: ITechnicalIndicatorService):
    """
    Example of a single decision-making cycle.
    """
    logger.info("--- Starting New Decision Cycle ---")

    # 1. Get Account and Symbol Info (Agent's "Context")
    account = await repo.get_account_state()
    logger.info(f"Account State: Balance={account.balance} {account.currency}, Equity={account.equity}")

    info = await repo.get_symbol_info(EXAMPLE_SYMBOL)
    logger.info(f"Symbol Info: {info.name}, MinVol={info.volume_min}, Step={info.volume_step}")

    # 2. Get Market Data & Indicators
    candles = await repo.get_last_candles(EXAMPLE_SYMBOL, EXAMPLE_TIMEFRAME, 200)
    if not candles:
        logger.warning("No candle data, skipping cycle.")
        return

    indicators = await indicator_svc.calculate_indicators(candles)
    if indicators.rsi:
        logger.info(f"Indicators: Latest RSI={indicators.rsi[-1]:.2f}, Latest ATR={indicators.atr[-1]:.5f}")

    # 3. Check Open Positions
    positions = await repo.get_open_positions()
    logger.info(f"Found {len(positions)} open positions.")

    eurusd_positions = [p for p in positions if p.symbol == EXAMPLE_SYMBOL]

    # 4. --- Example: Manage Existing Positions ---
    if eurusd_positions:
        pos = eurusd_positions[0]  # Manage the first found position
        logger.info(f"Managing existing position: {pos.ticket}, Vol={pos.volume}, Profit={pos.profit}")

        # --- NEW FEATURE 1: Update SL ---
        # Example: Move SL to breakeven if in profit
        if pos.profit > 1.0 and pos.stop_loss != pos.open_price:
            logger.info(f"Moving SL to breakeven for ticket {pos.ticket}")
            res_sl = await exec_svc.update_position_sl(pos.ticket, pos.open_price)
            logger.info(f"SL Update Result: {res_sl}")

        # --- NEW FEATURE 2: Partial Close ---
        # Example: Close 25% of position if profit > $5
        if pos.profit > 5.0 and pos.volume >= info.volume_min * 2:
            lot_to_close = round(pos.volume * 0.25, 2)
            # Ensure partial close is valid
            if lot_to_close >= info.volume_min:
                logger.info(f"Attempting partial close of {lot_to_close} lots for ticket {pos.ticket}")
                res_partial = await exec_svc.partial_close_position(pos.ticket, lot_to_close, "auto-partial-profit")
                logger.info(f"Partial Close Result: {res_partial}")
            else:
                logger.info(f"Skipping partial close, calculated lot {lot_to_close} is below min {info.volume_min}")

        # --- NEW FEATURE 3: Update TP ---
        # Example: Set a new random TP
        current_tick = await repo.get_tick(EXAMPLE_SYMBOL)
        if current_tick:
            new_tp = current_tick.ask + (random.randint(50, 100) * info.point)  # New 50-100 point TP
            new_tp = round(new_tp, info.digits)
            logger.info(f"Attempting to update TP for {pos.ticket} to {new_tp}")
            res_tp = await exec_svc.update_position_tp(pos.ticket, new_tp)
            logger.info(f"TP Update Result: {res_tp}")

        return  # Don't open a new trade if we are managing one

    # 5. --- Example: Open New Position ---
    # Simple signal: if RSI < 30, buy.
    if indicators.rsi and indicators.rsi[-1] < 30:
        logger.info("Signal: RSI is oversold. Preparing BUY order.")

        # --- Risk Management ---
        stop_loss_points = int(indicators.atr[-1] / info.point) * 2  # 2x ATR
        if stop_loss_points == 0: stop_loss_points = 500  # Fallback

        current_tick = await repo.get_tick(EXAMPLE_SYMBOL)
        if not current_tick:
            logger.error("Could not get tick for placing order.")
            return

        stop_loss_price = current_tick.ask - (stop_loss_points * info.point)
        take_profit_price = current_tick.ask + (stop_loss_points * 2 * info.point)  # 2:1 R:R

        lot = calculate_lot_size(
            account_balance=account.balance,
            risk_percent=0.01,  # 1% risk
            stop_loss_pips=float(stop_loss_points),
            pip_value_per_lot=info.trade_tick_value,
            volume_step=info.volume_step
        )
        logger.info(f"Calculated Lot: {lot} (Risk: 1%, SL Pips: {stop_loss_points})")

        if lot >= info.volume_min:
            result: OrderResult = await exec_svc.place_market_order(
                symbol=EXAMPLE_SYMBOL,
                action="buy",
                lot=lot,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                comment="example-agent-buy"
            )
            logger.info(f"Place order result: {result}")
        else:
            logger.warning("Lot size calculated to 0.0 or below min. Order not placed.")
    else:
        logger.info("No entry signal - holding")


async def main():
    """
    Main application entry point.
    """
    logger.info(f"--- Running MT5 Module Example for {EXAMPLE_SYMBOL} (M{EXAMPLE_TIMEFRAME}) ---")

    connector = None
    try:
        # 1. Setup Dependencies
        connector = MT5Connector()
        await connector.connect()

        # --- Dependency Injection ---
        # The services are instantiated and 'injected' with their dependencies.
        repo = MT5MarketDataRepository(connector)
        exec_svc = MT5TradeExecutionService(connector, repo)
        indicator_svc = PandasTAIndicatorService()

        # 2. Run the main logic loop (e.g., once every 60 seconds)
        # For this example, we'll just run it once.
        await example_trade_logic(repo, exec_svc, indicator_svc)

        # In a real agent, you might do this:
        # while True:
        #     await example_trade_logic(repo, exec_svc, indicator_svc)
        #     await asyncio.sleep(60)

    except ImportError:
        logger.error("---")
        logger.error("MetaTrader5 or pandas-ta not found. Please install required packages.")
        logger.error("pip install MetaTrader5 pandas pandas-ta")
        logger.error("---")
    except ConnectionError as ce:
        logger.error(f"Failed to connect to MT5: {ce}")
    except Exception as ex:
        logger.exception(f"Example run failed with an unexpected error: {ex}")
    finally:
        if connector and connector.connected:
            logger.info("Shutting down MT5 connection...")
            await connector.disconnect()
            logger.info("--- Example run complete ---")


if __name__ == "__main__":
    # This allows the async main function to be run
    asyncio.run(main())