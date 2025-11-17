"""
persistence.py
Handles the SQLite database for storing trade history and calculating
performance metrics (e.g., Sharpe Ratio).

Implements the ITradeRepository interface.
"""

import numpy as np
import sqlite3
import logging
import asyncio  # <-- NEW: Import asyncio
from dataclasses import dataclass
from typing import List, Optional, Protocol, Callable, Any
from datetime import datetime
from market import PositionInfo

logger = logging.getLogger(__name__)


# ----------------------------- Domain Layer (DDD) -----------------------------

@dataclass(frozen=True)
class ClosedTrade:
    """Represents a single completed trade for storage."""
    ticket: int
    symbol: str
    action: str  # "BUY" or "SELL"
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    lot_size: float
    profit: float
    swap: float
    # LLM context
    reasoning: Optional[str]
    confidence: Optional[float]


@dataclass(frozen=True)
class PerformanceMetrics:
    """Holds calculated performance metrics."""
    sharpe_ratio: float
    total_trades: int
    win_rate: float  # 0.0 to 1.0
    profit_factor: float  # Total Profit / Total Loss
    average_profit : float
    average_loss : float
    total_net_profit : float


# --------------------------- Interfaces / Ports (SOLID) ---------------------------

class ITradeRepository(Protocol):
    """
    Interface for the persistence layer, handling trade history
    and performance metrics.
    """

    async def store_new_open_trade(self,
                                   position: PositionInfo,
                                   reasoning: str,
                                   confidence: float) -> None:
        """
        Stores a new trade that has just been opened.
        It is stored with an 'OPEN' status.
        """
        ...

    async def update_trade_as_closed(self,
                                     ticket: int,
                                     close_time: datetime,
                                     close_price: float,
                                     profit: float,
                                     swap: float) -> None:
        """
        Updates an existing 'OPEN' trade to 'CLOSED' status,
        filling in the closing details.
        """
        ...

    async def get_open_trade_tickets(self) -> List[int]:
        """Fetches all trade tickets currently marked as 'OPEN' in the DB."""
        ...

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculates performance metrics from all 'CLOSED' trades in the DB.
        """
        ...


# --------------------------- Adapters / Implementation ---------------------------

class SQLiteTradeRepository(ITradeRepository):
    """
    SQLite implementation of the trade repository.
    This implementation runs all blocking SQLite operations in a
    separate thread pool to be fully async-compatible.
    """

    def __init__(self, db_path: str = "trade_history.db"):
        self._db_path = db_path
        # self._conn = None <-- REMOVED: We don't share connections
        logger.info(f"Persistence service will use database: {db_path}")
        self._initialize_db()  # Run init synchronously

    # <<< NEW BLOCK START >>>
    async def _run_in_executor(self, blocking_func: Callable[..., Any], *args: Any) -> Any:
        """Runs a blocking function in asyncio's default thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, blocking_func, *args)

    # <<< NEW BLOCK END >>>

    def _connect(self) -> sqlite3.Connection:
        """
        Creates a new database connection.
        Each blocking task will create its own.
        """
        # Add a 10-second timeout to wait for locks
        conn = sqlite3.connect(self._db_path, timeout=10.0)  # <-- MODIFIED
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _initialize_db(self):
        """Creates the 'trades' table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS trades (
            ticket INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            lot_size REAL NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open_price REAL NOT NULL,
            reasoning TEXT,
            confidence REAL,
            status TEXT NOT NULL DEFAULT 'OPEN', -- 'OPEN' or 'CLOSED'
            close_time TIMESTAMP,
            close_price REAL,
            profit REAL,
            swap REAL,
            CHECK (status IN ('OPEN', 'CLOSED')),
            CHECK (action IN ('BUY', 'SELL'))
        );
        """
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);
        """

        conn = None  # <-- MODIFIED
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_index_sql)
            conn.commit()
            logger.info("Database table 'trades' has been initialized.")
        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")
        finally:
            if conn:  # <-- MODIFIED
                conn.close()

    # <<< NEW BLOCK START >>>
    def _db_store_new_open_trade(self, params: tuple):
        """Blocking function to store a new trade."""
        sql = """
        INSERT INTO trades (
            ticket, symbol, action, lot_size, open_time, open_price,
            reasoning, confidence, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN');
        """
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Stored new open trade, ticket: {params[0]}")
        except sqlite3.IntegrityError:
            logger.warning(f"Trade {params[0]} is already in the database.")
        except Exception as e:
            logger.exception(f"Failed to store new open trade {params[0]}: {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    async def store_new_open_trade(self,
                                   position: PositionInfo,
                                   reasoning: str,
                                   confidence: float) -> None:
        params = (
            position.ticket,
            position.symbol,
            "BUY" if position.type == 0 else "SELL",
            position.volume,
            position.open_time.isoformat(),
            position.open_price,
            reasoning,
            confidence
        )
        await self._run_in_executor(self._db_store_new_open_trade, params)

    # <<< NEW BLOCK END >>>

    # <<< NEW BLOCK START >>>
    def _db_update_trade_as_closed(self, params: tuple):
        """Blocking function to update a trade to 'CLOSED'."""
        sql = """
        UPDATE trades
        SET 
            status = 'CLOSED',
            close_time = ?,
            close_price = ?,
            profit = ?,
            swap = ?
        WHERE 
            ticket = ? AND status = 'OPEN';
        """
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.rowcount == 0:
                logger.warning(f"No 'OPEN' trade found with ticket {params[4]} to update as closed.")
            else:
                logger.info(f"Updated trade {params[4]} to 'CLOSED'.")
            conn.commit()
        except Exception as e:
            logger.exception(f"Failed to update trade {params[4]} as closed: {e}")
            if conn: conn.rollback()
        finally:
            if conn: conn.close()

    async def update_trade_as_closed(self,
                                     ticket: int,
                                     close_time: datetime,
                                     close_price: float,
                                     profit: float,
                                     swap: float) -> None:
        params = (
            close_time.isoformat(),
            close_price,
            profit,
            swap,
            ticket
        )
        await self._run_in_executor(self._db_update_trade_as_closed, params)

    # <<< NEW BLOCK END >>>

    # <<< NEW BLOCK START >>>
    def _db_get_open_trade_tickets(self) -> List[int]:
        """Blocking function to get all 'OPEN' tickets."""
        sql = "SELECT ticket FROM trades WHERE status = 'OPEN';"
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            logger.exception(f"Failed to fetch open trade tickets: {e}")
            return []
        finally:
            if conn: conn.close()

    async def get_open_trade_tickets(self) -> List[int]:
        return await self._run_in_executor(self._db_get_open_trade_tickets)

    # <<< NEW BLOCK END >>>

    # <<< NEW BLOCK START >>>
    def _db_get_performance_metrics(self) -> PerformanceMetrics:
        """Blocking function to calculate performance metrics."""
        sql = "SELECT profit FROM trades WHERE status = 'CLOSED';"
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

            if not rows:
                logger.warning("No 'CLOSED' trade history found, returning default metrics.")
                return self._default_metrics()

            profits = [row[0] for row in rows]
            return self._calculate_metrics(profits)

        except Exception as e:
            logger.exception(f"Failed to calculate performance metrics: {e}")
            return self._default_metrics()
        finally:
            if conn: conn.close()

    async def get_performance_metrics(self) -> PerformanceMetrics:
        return await self._run_in_executor(self._db_get_performance_metrics)

    # <<< NEW BLOCK END >>>

    def _default_metrics(self) -> PerformanceMetrics:
        """Returns a default set of metrics when no history exists."""
        return PerformanceMetrics(
            sharpe_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0
        )

    def _calculate_metrics(self, profits: List[float]) -> PerformanceMetrics:
        """Helper to calculate metrics from a list of profits."""
        total_trades = len(profits)
        if total_trades == 0:
            return self._default_metrics()

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        total_net_profit = sum(profits)
        total_wins = len(wins)
        total_losses = len(losses)

        win_rate = (total_wins / total_trades) * 100
        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))

        average_profit = sum_wins / total_wins if total_wins > 0 else 0
        average_loss = sum_losses / total_losses if total_losses > 0 else 0
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0

        # --- Sharpe Ratio (Sample std dev, assume risk-free = 0)
        if total_trades > 1:
            mean_return = float(np.mean(profits))
            std_dev = float(np.std(profits, ddof=1))  # cast for type safety
            sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
        else:
            sharpe_ratio = 0

        # Optional annualization (if 252 trading days/year or adjust for your frequency)
        # sharpe_ratio *= math.sqrt(252)

        return PerformanceMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            average_profit=average_profit,
            average_loss=average_loss,
            total_net_profit=total_net_profit
        )