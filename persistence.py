"""
persistence.py
Provides data persistence services for the trading agent.

Uses SQLite to store closed trades and calculate historical performance metrics (like Sharpe Ratio).
Follows SOLID and DDD principles.
"""

import asyncio
import sqlite3
import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------- Domain Layer (DDD) -----------------------------

@dataclass(frozen=True)
class ClosedTrade:
    """
    Represents a closed and recorded trade.
    """
    id: Optional[int]
    symbol: str
    profit: float
    open_time: datetime
    close_time: datetime
    volume: float
    llm_reasoning: str
    llm_confidence: float


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Represents calculated historical performance metrics.
    """
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    average_profit: float
    average_loss: float
    total_net_profit: float


# --------------------------- Interfaces / Ports (SOLID) ---------------------------

class ITradeRepository(Protocol):
    """
    Interface (Port) for the persistence repository.
    """

    async def initialize(self) -> None:
        """Initializes the database table."""
        ...

    async def store_trade(self, trade: ClosedTrade) -> None:
        """Stores a closed trade."""
        ...

    async def get_all_trades(self) -> List[ClosedTrade]:
        """Retrieves all historical trades."""
        ...

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculates and returns all historical performance metrics."""
        ...


# --------------------------- Adapters / Implementation ---------------------------

class SQLiteTradeRepository(ITradeRepository):
    """
    Implements the ITradeRepository interface (Adapter) using SQLite.
    """

    def __init__(self, db_path: str = "trade_history.db"):
        self._db_path = db_path
        self._lock = asyncio.Lock()
        logger.info(f"Persistence service will use database: {db_path}")

    def _get_connection(self):
        """Gets a database connection."""
        return sqlite3.connect(self._db_path)

    async def initialize(self) -> None:
        """Creates the trades table if it does not exist."""
        query = """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            profit REAL NOT NULL,
            open_time TEXT NOT NULL,
            close_time TEXT NOT NULL,
            volume REAL NOT NULL,
            llm_reasoning TEXT,
            llm_confidence REAL
        );
        """
        async with self._lock:
            await asyncio.to_thread(self._execute_query, query)
        logger.info("Database table 'trades' has been initialized.")

    def _execute_query(self, query: str, params: tuple = ()):
        """Synchronously executes a database query in a thread."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database error: {e}\nQuery: {query}")
            raise

    async def store_trade(self, trade: ClosedTrade) -> None:
        """Inserts a trade into the database."""
        query = """
        INSERT INTO trades (symbol, profit, open_time, close_time, volume, llm_reasoning, llm_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            trade.symbol,
            trade.profit,
            trade.open_time.isoformat(),
            trade.close_time.isoformat(),
            trade.volume,
            trade.llm_reasoning,
            trade.llm_confidence
        )
        async with self._lock:
            await asyncio.to_thread(self._execute_query, query, params)
        logger.info(f"Stored trade: {trade.symbol} Profit {trade.profit:.2f}")

    async def get_all_trades(self) -> List[ClosedTrade]:
        """Retrieves all trades from the database."""
        query = "SELECT id, symbol, profit, open_time, close_time, volume, llm_reasoning, llm_confidence FROM trades ORDER BY close_time;"

        async with self._lock:
            rows = await asyncio.to_thread(self._execute_query, query)

        trades: List[ClosedTrade] = []
        for row in rows:
            try:
                trades.append(ClosedTrade(
                    id=row[0],
                    symbol=row[1],
                    profit=row[2],
                    open_time=datetime.fromisoformat(row[3]),
                    close_time=datetime.fromisoformat(row[4]),
                    volume=row[5],
                    llm_reasoning=row[6],
                    llm_confidence=row[7]
                ))
            except Exception as e:
                logger.error(f"Failed to parse trade data row: {row}, Error: {e}")
        return trades

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculates performance metrics for all historical trades."""
        trades = await self.get_all_trades()

        if not trades:
            logger.warning("No trade history found, returning default metrics.")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)

        profits: List[float] = [t.profit for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        total_trades = len(trades)
        total_net_profit = sum(profits)
        total_wins = len(wins)
        total_losses = len(losses)

        win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0

        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))

        average_profit = sum_wins / total_wins if total_wins > 0 else 0
        average_loss = sum_losses / total_losses if total_losses > 0 else 0

        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0

        # Calculate Sharpe Ratio (assuming risk-free rate is 0)
        # Simple version: (Mean Return) / (Std Dev of Returns)
        returns_array = np.array(profits)
        sharpe_ratio = 0.0
        if np.std(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array)
            # Annualized (e.g., assuming 1 trade/day, T=252)
            # sharpe_ratio = sharpe_ratio * np.sqrt(252)

        return PerformanceMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            average_profit=average_profit,
            average_loss=average_loss,
            total_net_profit=total_net_profit
        )


# -------------------------- Example Usage ----------------------------------

async def main():
    """Demonstrates usage of the persistence module."""

    # 1. Initialize repository
    repo = SQLiteTradeRepository(db_path="ex.db")  # Use example DB for testing
    await repo.initialize()

    # 2. Store some mock trades
    trade1 = ClosedTrade(
        id=None, symbol="EURUSD", profit=150.50,
        open_time=datetime(2025, 1, 1, 10, 0),
        close_time=datetime(2025, 1, 1, 12, 0),
        volume=0.1, llm_reasoning="RSI overbought", llm_confidence=0.8
    )
    trade2 = ClosedTrade(
        id=None, symbol="USDJPY", profit=-75.20,
        open_time=datetime(2025, 1, 2, 10, 0),
        close_time=datetime(2025, 1, 2, 12, 0),
        volume=0.05, llm_reasoning="Trend break", llm_confidence=0.7
    )
    await repo.store_trade(trade1)
    await repo.store_trade(trade2)

    # 3. Get all trades
    all_trades = await repo.get_all_trades()
    logger.info(f"\n--- Retrieved All Trades ({len(all_trades)}) ---")
    for trade in all_trades:
        logger.info(trade)

    # 4. Get performance metrics
    metrics = await repo.get_performance_metrics()
    logger.info("\n--- Performance Metrics ---")
    logger.info(metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())

