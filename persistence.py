"""
persistence.py
Handles the SQLite database for storing trade history, calculating
performance metrics, and tracking detailed trade modifications (SL/TP changes).

Implements the ITradeRepository interface.
"""

import numpy as np
import sqlite3
import logging
import asyncio  # <-- NEW: Import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Callable, Any
from datetime import datetime
from market import PositionInfo

logger = logging.getLogger(__name__)


# ----------------------------- Domain Layer (DDD) -----------------------------
@dataclass(frozen=True)
class TradeModification:
    """Represents a change made to an active trade."""
    id: int
    ticket: int
    modification_type: str  # 'UPDATE_SL', 'UPDATE_TP', 'PARTIAL_CLOSE'
    old_value: float
    new_value: float
    reasoning: str
    timestamp: datetime

@dataclass(frozen=True)
class ClosedTrade:
    """Represents a single completed trade with full history for storage/retrieval."""
    ticket: int
    symbol: str
    action: str  # "BUY" or "SELL"
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    initial_sl: float
    initial_tp: float
    lot_size: float
    profit: float
    swap: float
    balance_at_entry: float
    # Context
    open_reasoning: Optional[str]
    close_reasoning: Optional[str]
    confidence: Optional[float]
    # History of changes
    modifications: List[TradeModification] = field(default_factory=list)

    @property
    def return_pct(self) -> float:
        if self.balance_at_entry > 0:
            return (self.profit / self.balance_at_entry) * 100
        return 0.0

@dataclass(frozen=True)
class PerformanceMetrics:
    """Holds calculated performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    average_return_pct: float
    total_net_profit: float


# --------------------------- Interfaces / Ports (SOLID) ---------------------------

class ITradeRepository(Protocol):
    """
    Interface for the persistence layer, handling trade history
    and performance metrics.
    """

    async def store_new_open_trade(self,
                                   position: PositionInfo,
                                   account_balance: float,
                                   initial_sl: float,
                                   initial_tp: float,
                                   reasoning: str,
                                   confidence: float) -> None:
        """Stores a new trade with its initial SL/TP context."""
        ...

    async def update_trade_as_closed(self,
                                     ticket: int,
                                     close_time: datetime,
                                     close_price: float,
                                     profit: float,
                                     swap: float,
                                     close_reasoning: str) -> None:
        """Updates a trade to CLOSED status with a reason."""
        ...

    async def log_modification(self,
                               ticket: int,
                               mod_type: str,
                               old_val: float,
                               new_val: float,
                               reasoning: str) -> None:
        """Logs a change to an active trade (SL/TP update or Partial Close)."""
        ...

    async def get_open_trade_tickets(self) -> List[int]:
        """Fetches all trade tickets currently marked as 'OPEN' in the DB."""
        ...

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculates performance metrics from all 'CLOSED' trades in the DB.
        """
        ...

    async def get_closed_trades_history(self, limit: int = 5) -> List[ClosedTrade]:
        """Fetches the last N closed trades with their full modification history."""
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

    async def _run_in_executor(self, blocking_func: Callable[..., Any], *args: Any) -> Any:
        """Runs a blocking function in asyncio's default thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, blocking_func, *args)

    def _connect(self) -> sqlite3.Connection:
        """
        Creates a new database connection.
        Each blocking task will create its own.
        """
        # Add a 10-second timeout to wait for locks
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _initialize_db(self):
        """Creates tables if they don't exist. Handles simple schema updates."""
        create_trades_sql = """
        CREATE TABLE IF NOT EXISTS trades (
            ticket INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            lot_size REAL NOT NULL,
            open_time TIMESTAMP NOT NULL,
            open_price REAL NOT NULL,
            initial_sl REAL DEFAULT 0.0,
            initial_tp REAL DEFAULT 0.0,
            balance_at_entry REAL NOT NULL DEFAULT 0.0,
            reasoning TEXT,
            confidence REAL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            close_time TIMESTAMP,
            close_price REAL,
            close_reasoning TEXT,
            profit REAL,
            swap REAL,
            peak_profit REAL DEFAULT 0.0,
            CHECK (status IN ('OPEN', 'CLOSED')),
            CHECK (action IN ('BUY', 'SELL'))
        );
        """

        create_mods_sql = """
        CREATE TABLE IF NOT EXISTS trade_modifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket INTEGER NOT NULL,
            modification_type TEXT NOT NULL, -- UPDATE_SL, UPDATE_TP, PARTIAL_CLOSE
            old_value REAL,
            new_value REAL,
            reasoning TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(ticket) REFERENCES trades(ticket)
        );
        """

        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(create_trades_sql)
            cursor.execute(create_mods_sql)

            # --- Quick Migration Check (Idempotent) ---
            # Checks if 'initial_sl' exists, if not, adds columns.
            # Useful if running against an older DB version.
            cursor.execute("PRAGMA table_info(trades)")
            columns = [info[1] for info in cursor.fetchall()]

            if 'initial_sl' not in columns:
                logger.info("Migrating DB: Adding initial_sl/tp and close_reasoning columns...")
                cursor.execute("ALTER TABLE trades ADD COLUMN initial_sl REAL DEFAULT 0.0")
                cursor.execute("ALTER TABLE trades ADD COLUMN initial_tp REAL DEFAULT 0.0")
                cursor.execute("ALTER TABLE trades ADD COLUMN close_reasoning TEXT")

            conn.commit()
            logger.info("Database initialized.")
        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")
        finally:
            if conn: conn.close()

    def _db_store_new_open_trade(self, params: tuple):
        """Blocking function to store a new trade."""
        sql = """
                INSERT INTO trades (
                    ticket, symbol, action, lot_size, open_time, open_price,
                    balance_at_entry, initial_sl, initial_tp, reasoning, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN');
                """
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Stored new open trade {params[0]}")
        except sqlite3.IntegrityError:
            logger.warning(f"Trade {params[0]} already exists.")
        except Exception as e:
            logger.exception(f"Failed to store trade {params[0]}: {e}")
        finally:
            if conn: conn.close()

    async def store_new_open_trade(self,
                                   position: PositionInfo,
                                   account_balance: float,
                                   initial_sl: float,
                                   initial_tp: float,
                                   reasoning: str,
                                   confidence: float) -> None:
        params = (
            position.ticket,
            position.symbol,
            "BUY" if position.type == 0 else "SELL",
            position.volume,
            position.open_time.isoformat(),
            position.open_price,
            account_balance,
            initial_sl,
            initial_tp,
            reasoning,
            confidence
        )
        await self._run_in_executor(self._db_store_new_open_trade, params)

    def _db_log_modification(self, params: tuple):
        sql = """
        INSERT INTO trade_modifications (ticket, modification_type, old_value, new_value, reasoning)
        VALUES (?, ?, ?, ?, ?);
        """
        conn = None
        try:
            conn = self._connect()
            conn.execute(sql, params)
            conn.commit()
            logger.info(f"Logged modification for ticket {params[0]}: {params[1]}")
        except Exception as e:
            logger.exception(f"Failed to log modification: {e}")
        finally:
            if conn: conn.close()

    async def log_modification(self, ticket: int, mod_type: str, old_val: float, new_val: float,
                               reasoning: str) -> None:
        await self._run_in_executor(self._db_log_modification, (ticket, mod_type, old_val, new_val, reasoning))

    def _db_update_trade_as_closed(self, params: tuple):
        sql = """
        UPDATE trades
        SET status = 'CLOSED', close_time = ?, close_price = ?, profit = ?, swap = ?, close_reasoning = ?
        WHERE ticket = ? AND status = 'OPEN';
        """
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if cursor.rowcount > 0:
                logger.info(f"Updated trade {params[5]} to CLOSED.")
            conn.commit()
        except Exception as e:
            logger.exception(f"Failed to close trade {params[5]}: {e}")
        finally:
            if conn: conn.close()

    async def update_trade_as_closed(self, ticket: int, close_time: datetime, close_price: float,
                                     profit: float, swap: float, close_reasoning: str) -> None:
        params = (close_time.isoformat(), close_price, profit, swap, close_reasoning, ticket)
        await self._run_in_executor(self._db_update_trade_as_closed, params)

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

    def _db_get_closed_trades_history(self, limit: int) -> List[ClosedTrade]:
        # 1. Fetch basic trade info
        sql_trades = """
        SELECT ticket, symbol, action, open_time, close_time, open_price, close_price,
               initial_sl, initial_tp, lot_size, profit, swap, balance_at_entry,
               reasoning, close_reasoning, confidence
        FROM trades
        WHERE status = 'CLOSED'
        ORDER BY close_time DESC
        LIMIT ?;
        """

        # 2. Fetch modifications for these tickets
        # (We do this in a loop or a join, but separate queries are often cleaner for object mapping in simple sqlite)

        conn = None
        results = []
        try:
            conn = self._connect()
            conn.row_factory = sqlite3.Row  # Access by name
            cursor = conn.cursor()

            cursor.execute(sql_trades, (limit,))
            rows = cursor.fetchall()

            for row in rows:
                ticket = row['ticket']

                # Fetch mods
                cursor.execute("""
                    SELECT id, modification_type, old_value, new_value, reasoning, timestamp
                    FROM trade_modifications
                    WHERE ticket = ?
                    ORDER BY timestamp ASC
                """, (ticket,))
                mod_rows = cursor.fetchall()

                mods = []
                for mr in mod_rows:
                    mods.append(TradeModification(
                        id=mr['id'],
                        ticket=ticket,
                        modification_type=mr['modification_type'],
                        old_value=mr['old_value'] if mr['old_value'] else 0.0,
                        new_value=mr['new_value'] if mr['new_value'] else 0.0,
                        reasoning=mr['reasoning'],
                        timestamp=datetime.fromisoformat(mr['timestamp']) if mr['timestamp'] else datetime.utcnow()
                    ))

                # Parse timestamps
                t_open = datetime.fromisoformat(row['open_time'])
                t_close = datetime.fromisoformat(row['close_time']) if row['close_time'] else datetime.utcnow()

                trade = ClosedTrade(
                    ticket=ticket,
                    symbol=row['symbol'],
                    action=row['action'],
                    open_time=t_open,
                    close_time=t_close,
                    open_price=row['open_price'],
                    close_price=row['close_price'],
                    initial_sl=row['initial_sl'] if row['initial_sl'] else 0.0,
                    initial_tp=row['initial_tp'] if row['initial_tp'] else 0.0,
                    lot_size=row['lot_size'],
                    profit=row['profit'],
                    swap=row['swap'],
                    balance_at_entry=row['balance_at_entry'],
                    open_reasoning=row['reasoning'],
                    close_reasoning=row['close_reasoning'],
                    confidence=row['confidence'],
                    modifications=mods
                )
                results.append(trade)

            return results
        except Exception as e:
            logger.exception(f"Failed to fetch trade history: {e}")
            return []
        finally:
            if conn: conn.close()

    async def get_closed_trades_history(self, limit: int = 5) -> List[ClosedTrade]:
        return await self._run_in_executor(self._db_get_closed_trades_history, limit)

    def _db_get_metrics(self) -> PerformanceMetrics:
        sql = "SELECT profit, balance_at_entry FROM trades WHERE status = 'CLOSED';"
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

            if not rows:
                return self._default_metrics()

            profits = [r[0] for r in rows]
            # Returns calc: Avoid div/0
            returns_pct = [(r[0] / r[1] * 100) if r[1] > 0 else 0.0 for r in rows]

            total_trades = len(profits)
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]

            total_net = sum(profits)
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
            profit_factor = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else 99.0

            avg_return = float(np.mean(returns_pct))
            std_dev = float(np.std(returns_pct, ddof=1)) if total_trades > 1 else 0.0

            sharpe = (avg_return / std_dev) if std_dev > 0 else 0.0

            downside = [r for r in returns_pct if r < 0]
            down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
            sortino = (avg_return / down_std) if down_std > 0 else 0.0

            # Max DD
            equity = [100.0]
            curr = 100.0
            for r in returns_pct:
                curr *= (1 + r / 100)
                equity.append(curr)

            eq_arr = np.array(equity)
            peak = np.maximum.accumulate(eq_arr)
            dds = (eq_arr - peak) / peak
            max_dd = float(np.min(dds) * 100)

            return PerformanceMetrics(sharpe, sortino, abs(max_dd), total_trades, win_rate, profit_factor, avg_return,
                                      total_net)

        except Exception as e:
            logger.error(f"Metrics calc failed: {e}")
            return self._default_metrics()
        finally:
            if conn: conn.close()

    async def get_performance_metrics(self) -> PerformanceMetrics:
        return await self._run_in_executor(self._db_get_metrics)

    def _default_metrics(self):
        return PerformanceMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

    def _db_sync_states(self) -> dict:
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("SELECT ticket, peak_profit FROM trades WHERE status = 'OPEN'")
            return {row[0]: (row[1] or 0.0) for row in cursor.fetchall()}
        finally:
            if conn: conn.close()

    async def sync_open_trade_states(self) -> dict[int, float]:
        return await self._run_in_executor(self._db_sync_states)

    def _db_update_peak(self, args):
        conn = None
        try:
            conn = self._connect()
            conn.execute("UPDATE trades SET peak_profit = ? WHERE ticket = ?", args)
            conn.commit()
        finally:
            if conn: conn.close()

    async def update_peak_profit(self, ticket: int, new_peak: float) -> None:
        await self._run_in_executor(self._db_update_peak, (new_peak, ticket))
