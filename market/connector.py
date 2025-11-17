"""
MT5 Connection Manager
----------------------

This file contains the `MT5Connector` class, which has the
Single Responsibility of managing the MetaTrader5 library's
connection lifecycle and its associated thread pool.

All blocking C-library calls are run in this executor to
avoid blocking the main asyncio event loop.
"""

import asyncio
import concurrent.futures
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import MetaTrader5 but handle the case where it's unavailable
try:
    import MetaTrader5 as mt5  # type: ignore
except Exception as e:
    mt5 = None  # type: ignore
    logger.warning(f"MetaTrader5 package not found or failed to import: {e}")
    logger.warning("MT5-dependent functionality will not work.")


class MT5Connector:
    """
    Handles the lifecycle of the MetaTrader5 connection.
    Uses ThreadPoolExecutor to run blocking C-lib calls asynchronously.
    """
    def __init__(self):
        if mt5 is None:
            raise ImportError("MetaTrader5 package not found. Cannot instantiate MT5Connector.")

        self._connected = False
        self._lock = asyncio.Lock()
        # ThreadPoolExecutor for converting blocking MT5 calls to async-friendly calls
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="MT5Worker"
        )
        self.mt5 = mt5  # Expose the mt5 library module

    async def __aenter__(self):
        """Allows using the connector as an async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Shuts down the connection on context exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Initializes the MT5 connection."""
        async with self._lock:
            if self._connected:
                return

            loop = asyncio.get_running_loop()

            # mt5.initialize() is blocking
            ok = await loop.run_in_executor(
                self._executor,
                self.mt5.initialize
            )

            if not ok:
                await self._log_mt5_error("MT5 initialize failed")
                raise ConnectionError("Failed to initialize MT5. Check terminal path/login.")

            self._connected = True
            logger.info("MT5 initialized and login successful.")

    async def disconnect(self) -> None:
        """Shuts down the MT5 connection."""
        async with self._lock:
            if not self._connected:
                self._connected = False
                return

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.mt5.shutdown)
            self._connected = False
            logger.info("MT5 shutdown completed.")
            # Note: We don't shut down the executor here,
            # as it might be reused on a reconnect.
            # It will be garbage collected when MT5Connector is.

    @property
    def connected(self) -> bool:
        """Returns True if the connector is initialized."""
        return self._connected

    def executor(self) -> concurrent.futures.Executor:
        """Provides access to the executor for services to use."""
        return self._executor

    async def _log_mt5_error(self, message: str):
        """Helper to log the last MT5 error."""
        loop = asyncio.get_running_loop()
        last_error = await loop.run_in_executor(self._executor, self.mt5.last_error)
        logger.error(f"{message}: {last_error}")

    async def run_in_executor(self, blocking_func, *args, **kwargs):
        """
        Helper to run a blocking function in the connector's thread pool.
        This simplifies calls from the adapter classes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: blocking_func(*args, **kwargs)
        )