"""
trade_reconciler.py
A service dedicated to reconciling the agent's internal trade database
with the broker's ground truth. It finds and logs trades that were
closed by Take Profit (TP) or Stop Loss (SL).
"""

import logging
from typing import Protocol
from persistence import ITradeRepository
from mt5 import IMarketDataRepository

logger = logging.getLogger(__name__)


class ITradeReconciler(Protocol):
    """Interface for the Trade Reconciliation Service."""

    async def reconcile_trades(self) -> int:
        """
        Compares locally tracked open trades with the broker's open positions.
        Fetches history for any "ghost" trades (closed by SL/TP) and
        updates the persistence repository.

        Returns:
            The number of ghost trades reconciled.
        """
        ...


class TradeReconciler(ITradeReconciler):
    """Implementation of the reconciliation service."""

    def __init__(self,
                 trade_repo: ITradeRepository,
                 market_repo: IMarketDataRepository):
        self._trade_repo = trade_repo
        self._market_repo = market_repo
        logger.info("TradeReconciler initialized.")

    async def reconcile_trades(self) -> int:
        """
        Compares locally tracked open trades with the broker's open positions.
        Fetches history for any "ghost" trades (closed by SL/TP) and
        updates the persistence repository.
        """
        logger.info("Starting trade reconciliation...")

        try:
            # 1. Get our list of "open" trades
            local_open_tickets = await self._trade_repo.get_open_trade_tickets()
            if not local_open_tickets:
                logger.info("No locally tracked open trades to reconcile.")
                return 0

            # 2. Get the broker's list of *actually* open trades
            broker_open_positions = await self._market_repo.get_open_positions()
            broker_open_tickets = {pos.ticket for pos in broker_open_positions}

            logger.info(f"Local open tickets: {local_open_tickets}")
            logger.info(f"Broker open tickets: {broker_open_tickets}")

            # 3. Find the "ghost" trades
            ghost_trade_tickets = [
                ticket for ticket in local_open_tickets
                if ticket not in broker_open_tickets
            ]

            if not ghost_trade_tickets:
                logger.info("Reconciliation complete. No ghost trades found.")
                return 0

            logger.warning(f"Found {len(ghost_trade_tickets)} ghost trade(s): {ghost_trade_tickets}")

            # 4. Fetch history for each ghost and update our DB
            reconciled_count = 0
            for ticket in ghost_trade_tickets:
                logger.info(f"Fetching history for ghost trade {ticket}...")
                closed_trade = await self._market_repo.get_historical_trade(ticket)

                if closed_trade:
                    # Mark it as "CLOSED" in our database
                    await self._trade_repo.update_trade_as_closed(
                        ticket=ticket,
                        close_time=closed_trade.close_time,
                        close_price=closed_trade.close_price,
                        profit=closed_trade.profit,
                        swap=closed_trade.swap
                    )
                    logger.info(f"Successfully reconciled and closed trade {ticket}. Profit: {closed_trade.profit}")
                    reconciled_count += 1
                else:
                    # This can happen if the trade closed but history is not yet available
                    logger.warning(f"Could not fetch history for ghost trade {ticket}. Will retry next cycle.")

            logger.info(f"Reconciliation finished. {reconciled_count} trades updated.")
            return reconciled_count

        except Exception as e:
            logger.exception(f"Error during trade reconciliation: {e}")
            return 0