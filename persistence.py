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
    balance_at_entry: float  # <--- NEW: Crucial for % calculations
    # LLM context
    reasoning: Optional[str]
    confidence: Optional[float]


@dataclass(frozen=True)
class PerformanceMetrics:
    """Holds calculated performance metrics."""
    sharpe_ratio: float      # Annualized (or Per-Trade Risk adjusted)
    sortino_ratio: float     # Downside deviation only
    max_drawdown_pct: float  # <--- NEW: Critical Risk Metric
    total_trades: int
    win_rate: float          # 0.0 to 100.0
    profit_factor: float     # Gross Profit / Gross Loss
    average_return_pct: float # <--- NEW: Avg % gain per trade
    total_net_profit: float


# --------------------------- Interfaces / Ports (SOLID) ---------------------------

class ITradeRepository(Protocol):
    """
    Interface for the persistence layer, handling trade history
    and performance metrics.
    """

    async def store_new_open_trade(self,
                                   position: PositionInfo,
                                   account_balance: float,  # <--- NEW ARGUMENT
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

    async def sync_open_trade_states(self) -> dict[int, float]:
        """
        Fetches a mapping of {ticket: peak_profit} for all currently 'OPEN' trades.
        Used to initialize the in-memory state at the start of a cycle.
        """
        ...

    async def update_peak_profit(self, ticket: int, new_peak: float) -> None:
        """
        Updates the peak_profit (High-Water Mark) for a specific open trade.
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
            balance_at_entry REAL NOT NULL DEFAULT 0.0,
            reasoning TEXT,
            confidence REAL,
            status TEXT NOT NULL DEFAULT 'OPEN', -- 'OPEN' or 'CLOSED'
            close_time TIMESTAMP,
            close_price REAL,
            profit REAL,
            swap REAL,
            peak_profit REAL DEFAULT 0.0,
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
            balance_at_entry, reasoning, confidence, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN');
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
                                   account_balance: float,
                                   reasoning: str,
                                   confidence: float) -> None:
        params = (
            position.ticket,
            position.symbol,
            "BUY" if position.type == 0 else "SELL",
            position.volume,
            position.open_time.isoformat(),
            position.open_price,
            account_balance,  # <--- Saving the context of capital
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


    def _db_get_performance_metrics(self) -> PerformanceMetrics:
        # We need Profit AND Balance At Entry to calculate returns %
        sql = "SELECT profit, balance_at_entry FROM trades WHERE status = 'CLOSED';"
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

            if not rows:
                return self._default_metrics()

            return self._calculate_institutional_metrics(rows)

        except Exception as e:
            logger.exception(f"Failed to calculate metrics: {e}")
            return self._default_metrics()
        finally:
            if conn: conn.close()

    async def get_performance_metrics(self) -> PerformanceMetrics:
        return await self._run_in_executor(self._db_get_performance_metrics)


    def _db_sync_open_trade_states(self) -> dict[int, float]:
        """Blocking: Returns {ticket: peak_profit} for all OPEN trades."""
        sql = "SELECT ticket, peak_profit FROM trades WHERE status = 'OPEN';"
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            # Return dict: {ticket_id: peak_value}
            return {row[0]: (row[1] if row[1] is not None else 0.0) for row in rows}
        except Exception as e:
            logger.error(f"Failed to sync open trade states: {e}")
            return {}
        finally:
            if conn: conn.close()

    async def sync_open_trade_states(self) -> dict[int, float]:
        return await self._run_in_executor(self._db_sync_open_trade_states)

    # <<< NEW: Granular Update Logic >>>
    def _db_update_peak_profit(self, params: tuple):
        """Blocking: Updates peak_profit for a specific ticket."""
        sql = "UPDATE trades SET peak_profit = ? WHERE ticket = ?;"
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update peak profit for {params[1]}: {e}")
        finally:
            if conn: conn.close()

    async def update_peak_profit(self, ticket: int, new_peak: float) -> None:
        await self._run_in_executor(self._db_update_peak_profit, (new_peak, ticket))

    def _default_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_net_profit=0.0,
            average_return_pct=0.0
        )

    def _calculate_institutional_metrics(self, data: List[tuple]) -> PerformanceMetrics:
        """
        Calculates metrics based on Percentage Returns (Capital at Risk).
        """
        profits = [row[0] for row in data]

        # 1. Calculate Percentage Returns per trade
        returns_pct = []
        for p, b in data:
            if b > 0:
                returns_pct.append((p / b) * 100)
            else:
                returns_pct.append(0.0)

        total_trades = len(profits)
        if total_trades == 0:
            return self._default_metrics()

        # --- Basic Stats ---
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        total_net_profit = sum(profits)
        win_rate = (len(wins) / total_trades) * 100

        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 99.0

        # FIX 1: Explicit cast for average return
        average_return_pct = float(np.mean(returns_pct))

        # --- Advanced Risk Metrics (Sharpe & Sortino) ---
        returns_np = np.array(returns_pct)

        # FIX 2: Explicit cast for std_dev
        std_dev = float(np.std(returns_np, ddof=1)) if total_trades > 1 else 0.0

        # Sharpe Ratio
        if std_dev > 0:
            sharpe_ratio = float(average_return_pct / std_dev)
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio
        downside_returns = returns_np[returns_np < 0]
        if len(downside_returns) > 1:
            downside_std = float(np.std(downside_returns, ddof=1))
            sortino_ratio = float(average_return_pct / downside_std) if downside_std > 0 else 0.0
        else:
            sortino_ratio = 0.0

        # --- Max Drawdown % ---
        running_equity = [100.0]
        current_eq = 100.0

        for ret in returns_pct:
            current_eq = current_eq * (1 + (ret / 100))
            running_equity.append(current_eq)

        equity_curve = np.array(running_equity)
        peak_curve = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak_curve) / peak_curve

        # FIX 3: Explicit cast for Max Drawdown
        max_dd = float(np.min(drawdowns) * 100)

        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=abs(max_dd),
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_return_pct=average_return_pct,  # Now safely a float
            total_net_profit=total_net_profit
        )