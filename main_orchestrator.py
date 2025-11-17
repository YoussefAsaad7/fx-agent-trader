"""
main_orchestrator.py
The main entry point and orchestrator for the Forex Trading Agent.

This module is responsible for:
1. Loading configuration.
2. Setting up all services (Dependency Injection).
3. Running the main trading loop via a scheduler.
4. Handling graceful shutdown.
"""

import asyncio
import logging
import sys
from typing import Optional, Tuple
import os
# Import all our custom modules
from config import load_config, Config
from market import (
    MT5Connector,
    MT5MarketDataRepository,
    MT5TradeExecutionService,
    PandasTAIndicatorService
)
from agent_context_builder import AgentContextBuilder
from persistence import SQLiteTradeRepository
from llm_client import (
    ILLMClient,
    GeminiClient,
    DeepSeekClient,
    QwenClient
)
from engine import ForexAgentEngine
from trade_reconciler import TradeReconciler, ITradeReconciler

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    Manages the main trading loop, calling the engine at a fixed interval.
    This is an OOP implementation of the "customizable scheduler."
    """

    def __init__(self,
                 engine: ForexAgentEngine,
                 config: Config,
                 reconciler):
        self._engine = engine
        self._config = config
        self._task: Optional[asyncio.Task] = None
        self._reconciler: ITradeReconciler = reconciler
        self._running = False

    async def start(self):
        """Starts the scheduler loop."""
        self._running = True
        logger.info(
            f"TradingScheduler starting. Run interval: {self._config.run_interval_seconds}s, Dry Run: {self._config.dry_run}")
        self._task = asyncio.create_task(self._run_loop())
        await self._task

    async def _run_loop(self):
        """The main execution loop."""
        while self._running:
            try:
                logger.info("--- Starting new decision cycle ---")
                # 0. Reconcile trades first!
                # This ensures our view of the world is accurate *before*
                # we fetch metrics or make decisions.
                reconciled_count = await self._reconciler.reconcile_trades()
                logger.info(f"Reconciliation found and closed {reconciled_count} ghost trades.")

                # 1. Decide
                full_decision = await self._engine.decide(self._config.watch_list)
                logger.info(f"LLM CoT: {full_decision.cot_reasoning}...")

                # 2. Execute
                if self._config.dry_run:
                    logger.warning("--- DRY RUN MODE: No trades will be executed. ---")

                for decision in full_decision.decisions:
                    if decision._validation_error:
                        logger.error(
                            f"Decision INVALID: {decision.symbol} {decision.action} - {decision._validation_error}")
                    else:
                        logger.info(
                            f"Decision VALID: {decision.symbol} {decision.action} (Lot: {decision._calculated_lot_size})")

                        if not self._config.dry_run:
                            await self._engine.execute(decision, full_decision.context, full_decision.cot_reasoning)
                        else:
                            logger.info(f"[DRY RUN] Would execute {decision.action} for {decision.symbol}")

                logger.info(f"--- Decision cycle complete. Waiting {self._config.run_interval_seconds}s ---")

            except Exception as e:
                logger.exception(f"Critical error in run loop: {e}")
                # Don't crash the loop, wait and try again

            await asyncio.sleep(self._config.run_interval_seconds)

    async def stop(self):
        """Stops the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("TradingScheduler stopped.")


async def setup_dependencies(cfg: Config) -> Tuple[ForexAgentEngine, MT5Connector, ILLMClient, ITradeReconciler]:
    """
    Initializes all services and wires them together (Dependency Injection).
    """
    logger.info("Setting up dependencies...")

    # 1. MT5 Connection
    # We must pass the login info to the connector now
    connector = MT5Connector()
    # Must connect before proceeding
    await connector.connect()

    # 2. Core Services
    repo = MT5MarketDataRepository(connector)
    indicator_svc = PandasTAIndicatorService()
    builder = AgentContextBuilder(repo, indicator_svc)
    executor = MT5TradeExecutionService(connector, repo)

    # 3. Persistence
    persistence = SQLiteTradeRepository(db_path="trade_history.db")

    reconciler = TradeReconciler(
        trade_repo=persistence,
        market_repo=repo
    )
    # 4. LLM Client (Strategy Pattern)
    llm_client: ILLMClient
    if cfg.llm_provider == "deepseek":
        llm_client = DeepSeekClient(api_key=cfg.deepseek_api_key)
    elif cfg.llm_provider == "qwen":
        llm_client = QwenClient(api_key=os.getenv("QWEN_API_KEY", ""))  # Example
    else:
        llm_client = GeminiClient(api_key=cfg.gemini_api_key)

    # 5. System Prompt
    try:
        with open(cfg.system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except FileNotFoundError:
        logger.error(f"FATAL: system_prompt.txt not found at path: {cfg.system_prompt_path}")
        logger.error("Please create this file and add the agent's prompt.")
        sys.exit(1)  # Cannot run without a prompt

    # 6. Main Engine
    engine = ForexAgentEngine(
        context_builder=builder,
        trade_executor=executor,
        persistence_repo=persistence,
        llm_client=llm_client,
        market_repo=repo,  # Pass repo again for validation
        system_prompt_template=system_prompt
    )

    logger.info("All dependencies initialized successfully.")
    return engine, connector, llm_client, reconciler


async def main():
    """Main application entry point."""
    connector: Optional[MT5Connector] = None
    llm_client: Optional[ILLMClient] = None
    scheduler: Optional[TradingScheduler] = None

    try:
        # 1. Load Config
        cfg = load_config()

        # 2. Setup
        engine, connector, llm_client, reconciler  = await setup_dependencies(cfg)

        # 3. Run
        scheduler = TradingScheduler(engine, cfg, reconciler)
        await scheduler.start()

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.exception(f"Application failed to start: {e}")
    finally:
        # 4. Graceful Shutdown
        if scheduler:
            await scheduler.stop()
        if connector and connector.connected:
            await connector.disconnect()
        if llm_client and hasattr(llm_client, 'close'):
            await llm_client.close()  # Close httpx client
        logger.info("Application shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")