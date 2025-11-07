"""
engine.py
ForexAgentEngine: Business process orchestrator.

- Uses AgentContextBuilder to build the full market context.
- Uses Persistence module to get historical performance (Sharpe Ratio).
- Builds the System and User prompts for the LLM.
- Calls the LLM (ILLMClient).
- Parses and validates the LLM's response (CoT and JSON decisions).
- (Optional) Executes decisions via MT5TradeExecutionService.
- (Optional) Stores closed trade results into the Persistence module.
"""

import asyncio
import logging
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Dict
from datetime import datetime

# Import all our custom modules
from mt5 import (
    IMarketDataRepository,
    ITradeExecutionService,
    SymbolInfo,
    AccountState,
    PositionInfo,
    calculate_lot_size,
    OrderResult,
    TechnicalIndicators, MarketCandle, ITechnicalIndicatorService
)
from agent_context_builder import (
    AgentContextBuilder,
    AgentFullContext,
    SymbolWatchConfig
)
from persistence import (
    ITradeRepository,
    PerformanceMetrics,
    ClosedTrade, SQLiteTradeRepository
)

logger = logging.getLogger(__name__)


# ----------------------------- Domain Layer (DDD) -----------------------------

@dataclass
class AgentDecision:
    """
    Structured representation of a single decision from an LLM response.
    Matches the fields in the agent prompt.
    """
    symbol: str
    action: str  # 'buy_to_enter', 'sell_to_enter', 'hold', 'close'
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    invalidation_condition: Optional[str] = None
    confidence: Optional[float] = None

    # lot_size is suggested by LLM, but ultimately calculated by engine during validation
    lot_size: Optional[float] = None

    # Populated by the engine during validation
    _calculated_lot_size: float = 0.0
    _validation_error: Optional[str] = None


@dataclass
class LLMFullDecision:
    """
    Stores the complete inputs, outputs, and raw data for a single LLM decision cycle.
    """
    system_prompt: str
    user_prompt: str
    raw_response: str
    cot_reasoning: str
    decisions: List[AgentDecision]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # We add the context here so the executor can use it
    context: Optional[AgentFullContext] = None


# --------------------------- Interfaces / Ports (SOLID) ---------------------------

class ILLMClient(Protocol):
    """
    Interface (Port) for the LLM client.
    """

    async def call(self, system: str, user: str) -> str:
        """Calls the LLM and returns a raw string response."""
        ...


# -------------------------- Engine (Orchestrator) ---------------------------

# Regex ported from Go example for robust LLM output parsing
RE_JSON_FENCE = re.compile(r'(?is)' + r"```json\s*(\[\s*\{.*?\}\s*\])\s*```")
RE_JSON_ARRAY = re.compile(r'(?is)\[\s*\{.*?\}\s*\]')
RE_INVISIBLE = re.compile(r'[\u200B\u200C\u200D\uFEFF]')
RE_REASONING_TAG = re.compile(r'(?s)<reasoning>(.*?)</reasoning>')


class ForexAgentEngine:
    """
    The core engine that orchestrates data flow, LLM calls, and trade execution.
    """

    def __init__(self,
                 context_builder: AgentContextBuilder,
                 trade_executor: ITradeExecutionService,
                 persistence_repo: ITradeRepository,
                 llm_client: ILLMClient,
                 market_repo: IMarketDataRepository,
                 system_prompt_template: str):

        self._builder = context_builder
        self._executor = trade_executor
        self._persistence = persistence_repo
        self._llm = llm_client
        self._market_repo = market_repo  # For live price validation
        self._system_prompt_template = system_prompt_template

        self.allowed_actions = {"buy_to_enter", "sell_to_enter", "hold", "close"}

    def _build_system_prompt(self, metrics: PerformanceMetrics) -> str:
        """
        Builds the system prompt, dynamically injecting the current Sharpe Ratio.
        """
        prompt = self._system_prompt_template

        # Inject dynamic metrics
        replacement_text = (
            f"Current Performance (Sharpe Ratio: {metrics.sharpe_ratio:.2f})\n"
            f"Total Trades: {metrics.total_trades}, Win Rate: {metrics.win_rate:.1f}%\n"
            f"---"
        )
        prompt = prompt.replace(
            "You will receive your Sharpe Ratio at each invocation:",
            replacement_text
        )
        return prompt

    def _build_user_prompt(self, context: AgentFullContext) -> str:
        """
        Formats the rich AgentFullContext object into a text input for the LLM.
        """
        lines = []

        # 1. Account State
        acc = context.account_state
        lines.append(f"## Account State")
        lines.append(f"Balance: {acc.balance:.2f} {acc.account_currency}")
        lines.append(f"Equity: {acc.equity:.2f}")
        lines.append(f"Free Margin: {acc.free_margin:.2f}")
        lines.append("---")

        # 2. Upcoming Economic Events
        lines.append("## Upcoming Economic Events")
        if context.economic_events:
            for event in context.economic_events:
                lines.append(
                    f"- {event.get('time_until_event', 'N/A')}: {event.get('event', 'N/A')} (Impact: {event.get('impact', 'N/A')})")
        else:
            lines.append("No high-impact events scheduled.")
        lines.append("---")

        # 3. Open Positions
        lines.append("## Open Positions")
        if context.open_positions:
            for pos in context.open_positions:
                lines.append(f"- {pos.symbol} {pos.type.upper()} {pos.volume} lots")
                lines.append(f"  Entry: {pos.open_price:.5f}, Current: {pos.current_price:.5f}")
                lines.append(f"  Profit: {pos.profit:.2f}, Swap: {pos.swap:.2f}")
                lines.append(f"  SL: {pos.stop_loss:.5f}, TP: {pos.take_profit:.5f} (Ticket: {pos.ticket})")
        else:
            lines.append("No open positions.")
        lines.append("---")

        # 4. Market Data (Critical)
        lines.append("## Market Data & Indicators (Oldest -> Newest)")
        for symbol, sym_context in context.market_context.items():
            info = sym_context.symbol_info
            lines.append(f"\n### Symbol: {symbol}")
            lines.append(f"Spread: {info.spread} points | Swap: {info.swap_long} (L) / {info.swap_short} (S)")
            lines.append(f"Tick Value: {info.trade_tick_value} {acc.account_currency} | Point: {info.point}")

            for tf_data in sym_context.market_data:
                lines.append(f"\n  Timeframe: {tf_data.timeframe} (Last 10 data points)")

                # Slicing: Only show the last 10 data points
                slice_len = -10

                # Format candles
                candles_str = [
                    f"C: {c.close:.5f} H: {c.high:.5f} L: {c.low:.5f}"
                    for c in tf_data.candles[slice_len:]
                ]
                lines.append(f"    Candles: [{', '.join(candles_str)}]")

                # Format indicators
                inds = tf_data.indicators
                if inds.ema_fast:
                    lines.append(f"    EMA_Fast: [{', '.join([f'{x:.5f}' for x in inds.ema_fast[slice_len:]])}]")
                if inds.ema_slow:
                    lines.append(f"    EMA_Slow: [{', '.join([f'{x:.5f}' for x in inds.ema_slow[slice_len:]])}]")
                if inds.macd:
                    lines.append(f"    MACD:     [{', '.join([f'{x:.5f}' for x in inds.macd[slice_len:]])}]")
                if inds.macd_signal:
                    lines.append(f"    Signal:   [{', '.join([f'{x:.5f}' for x in inds.macd_signal[slice_len:]])}]")
                if inds.rsi:
                    lines.append(f"    RSI:      [{', '.join([f'{x:.2f}' for x in inds.rsi[slice_len:]])}]")
                if inds.atr:
                    lines.append(f"    ATR:      [{', '.join([f'{x:.5f}' for x in inds.atr[slice_len:]])}]")

        lines.append("\n---\nAnalyze the data and provide your decision JSON.")
        return "\n".join(lines)

    def _extract_cot(self, raw: str) -> str:
        """
        Extracts the Chain of Thought (CoT) using the <reasoning> tag or all text before the JSON.
        """
        # 1. Try <reasoning> tag
        match = RE_REASONING_TAG.search(raw)
        if match:
            return match.group(1).strip()

        # 2. Fallback: get all content before JSON
        json_match = RE_JSON_FENCE.search(raw) or RE_JSON_ARRAY.search(raw)
        if json_match:
            return raw[:json_match.start()].strip()

        # 3. If no JSON, all is CoT
        return raw.strip()

    def _clean_json_str(self, s: str) -> str:
        """Cleans common LLM JSON errors."""
        s = RE_INVISIBLE.sub("", s)  # Remove invisible characters
        # Ported from Go example: replace full-width characters
        s = s.replace('\u201c', '"').replace('\u201d', '"')
        s = s.replace('［', '[').replace('］', ']')
        s = s.replace('｛', '{').replace('｝', '}')
        s = s.replace('：', ':').replace('，', ',')
        return s

    def _extract_decisions(self, raw: str) -> List[AgentDecision]:
        """
        Extracts a list of structured AgentDecision objects from the LLM's raw response.
        """
        s = self._clean_json_str(raw)

        json_text = ""
        # 1. Try ```json fence
        match = RE_JSON_FENCE.search(s)
        if match:
            json_text = match.group(1)
        else:
            # 2. Fallback: find first JSON array
            match = RE_JSON_ARRAY.search(s)
            if match:
                json_text = match.group(0)

        if not json_text:
            logger.warning("No JSON array found in LLM response.")
            return []

        try:
            # Parse JSON
            data = json.loads(json_text)
            decisions = []
            for item in data:
                # Check if it's a valid dict
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid JSON item: {item}")
                    continue

                decisions.append(AgentDecision(
                    symbol=item.get("symbol"),
                    action=item.get("action"),
                    profit_target=item.get("profit_target"),
                    stop_loss=item.get("stop_loss"),
                    invalidation_condition=item.get("invalidation_condition"),
                    confidence=float(item.get("confidence", 0.0)),
                    lot_size=float(item.get("lot_size", 0.0))  # LLM suggestion
                ))
            return decisions
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nOriginal JSON text:\n{json_text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing decisions: {e}")
            return []

    async def _validate_decision(self, d: AgentDecision, ctx: AgentFullContext):
        """
        Validates a single decision.
        Core responsibility: Calculate and validate `lot_size`.
        """
        if d.action not in self.allowed_actions:
            d._validation_error = f"Invalid action: {d.action}"
            return

        if d.action in ("buy_to_enter", "sell_to_enter"):
            # 1. Check basic requirements
            if not all([d.symbol, d.stop_loss, d.profit_target, d.confidence]):
                d._validation_error = "Open decision is missing symbol, stop_loss, profit_target, or confidence"
                return

            # 2. Get required context
            acc = ctx.account_state
            sym_ctx = ctx.market_context.get(d.symbol)
            if not sym_ctx:
                d._validation_error = f"Market context for Symbol {d.symbol} is not available"
                return

            info = sym_ctx.symbol_info

            # 3. Get live price (for SL points calculation)
            try:
                # We need MarketDataRepo to get the *current* quote
                # Note: get_symbol_info in our module returns SymbolInfo,
                # but we need a quote. We should have a get_tick() method in the repo.
                # For demonstration, we'll use the latest candle's close price
                if not sym_ctx.market_data:
                    d._validation_error = "No candle data available to get entry price"
                    return

                # Use last close of H1 data (or M5, whichever exists)
                live_price = sym_ctx.market_data[0].candles[-1].close

            except Exception as e:
                d._validation_error = f"Failed to get live price: {e}"
                return

            # 4. Calculate risk parameters
            stop_loss_points = abs(live_price - d.stop_loss) / info.point
            if stop_loss_points == 0:
                d._validation_error = "Stop loss points calculated to 0 (SL price equals market price)"
                return

            # 5. Risk Percentage (based on LLM confidence)
            # Match mapping in prompt
            if d.confidence < 0.3:
                risk_percent = 0.0025  # 0.25% (very low confidence, no trade)
            elif d.confidence < 0.6:
                risk_percent = 0.005  # 0.5%
            elif d.confidence < 0.8:
                risk_percent = 0.01  # 1.0%
            else:
                risk_percent = 0.015  # 1.5% (high confidence)

            if d.confidence < 0.3:
                d._validation_error = f"Confidence ({d.confidence}) is too low to trade"
                return

            # 6. Calculate Lot Size
            calculated_lot = calculate_lot_size(
                account_balance=acc.balance,  # Use balance, or free_margin
                risk_percent=risk_percent,
                stop_loss_pips=stop_loss_points,  # This is in "points"
                pip_value_per_lot=info.trade_tick_value,  # This is "value per point"
                volume_step=info.volume_step
            )

            # 7. Validate Lot Size
            if calculated_lot < info.volume_min:
                d._validation_error = f"Calculated lot {calculated_lot} is below minimum {info.volume_min}"
                return

            d._calculated_lot_size = calculated_lot
            logger.info(f"Decision validated: {d.symbol} {d.action}, Calculated Lot: {calculated_lot}")

    async def decide(self, watch_list: List[SymbolWatchConfig]) -> LLMFullDecision:
        """
        Executes one full decision cycle.
        """
        try:
            # 1. Build context
            logger.info("Building context...")
            context = await self._builder.build_context(watch_list)

            # 2. Get performance metrics
            logger.info("Getting performance metrics...")
            metrics = await self._persistence.get_performance_metrics()

            # 3. Build prompts
            logger.info("Building prompts...")
            system_prompt = self._build_system_prompt(metrics)
            user_prompt = self._build_user_prompt(context)

            # 4. Call LLM
            logger.info("Calling LLM...")
            raw_response = await self._llm.call(system_prompt, user_prompt)

            # 5. Parse response
            logger.info("Parsing LLM response...")
            cot = self._extract_cot(raw_response)
            decisions = self._extract_decisions(raw_response)

            # 6. Validate decisions (and calculate lot size)
            logger.info("Validating decisions...")
            validation_tasks = [self._validate_decision(d, context) for d in decisions]
            await asyncio.gather(*validation_tasks)

            return LLMFullDecision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw_response,
                cot_reasoning=cot,
                decisions=decisions,
                context=context  # Attach context for the executor
            )

        except Exception as e:
            logger.exception("Critical error in decision cycle")
            return LLMFullDecision(
                system_prompt="ERROR", user_prompt="ERROR", raw_response=str(e),
                cot_reasoning=f"Engine failed: {e}", decisions=[]
            )

    async def execute(self, decision: AgentDecision, context: AgentFullContext):
        """
        Executes a single validated decision.
        """
        if decision._validation_error:
            logger.error(
                f"Rejecting execution of invalid decision: {decision.symbol} {decision.action} - {decision._validation_error}")
            return

        action = decision.action

        try:
            if action == "buy_to_enter" or action == "sell_to_enter":
                lot = decision._calculated_lot_size
                if lot == 0.0:
                    logger.error("Rejecting execution: calculated lot is 0.0")
                    return

                logger.info(f"--- EXECUTING {action.upper()} {decision.symbol} @ {lot} lots ---")
                result = await self._executor.place_market_order(
                    symbol=decision.symbol,
                    action="buy" if action == "buy_to_enter" else "sell",
                    lot=lot,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.profit_target,
                    comment=f"Agent v1 | {decision.confidence:.0%}"
                )
                logger.info(f"Execution result: {result}")

            elif action == "close":
                logger.info(f"--- EXECUTING CLOSE {decision.symbol} ---")
                # Find open position for this symbol
                pos_to_close = None
                for pos in context.open_positions:
                    if pos.symbol == decision.symbol:
                        pos_to_close = pos
                        break

                if not pos_to_close:
                    logger.warning(f"Wanted to close {decision.symbol}, but no open position was found.")
                    return

                result = await self._executor.close_position(pos_to_close.ticket)
                logger.info(f"Close result: {result}")

                # Critical: If successful, log to persistence
                if result.success:
                    # We need the open time. pos_to_close doesn't have it, PositionInfo needs modification
                    # To fix this, mt5_module.py's get_open_positions should also fetch `pos.time`
                    # Assuming we can get open_time...
                    open_time = datetime.utcnow()  # Mock: Replace with actual pos.time

                    closed_trade = ClosedTrade(
                        id=None,
                        symbol=pos_to_close.symbol,
                        profit=pos_to_close.profit,  # MT5 PositionInfo has P/L
                        open_time=open_time,
                        close_time=datetime.utcnow(),
                        volume=pos_to_close.volume,
                        llm_reasoning=f"Close decision: {decision.invalidation_condition}",
                        llm_confidence=decision.confidence or 0.5
                    )
                    await self._persistence.store_trade(closed_trade)

            elif action == "hold":
                logger.info(f"--- EXECUTING HOLD {decision.symbol} ---")

        except Exception as e:
            logger.exception(f"Execution failed for {action} {decision.symbol}")


# -------------------------- Mocking and Example Usage ----------------------------------

class MockLLMClient(ILLMClient):
    """Mock LLM client for testing the engine."""

    async def call(self, system: str, user: str) -> str:
        logger.info("--- MOCK LLM CALL (System) ---")
        logger.info(system[:200] + "...")
        logger.info("--- MOCK LLM CALL (User) ---")
        logger.info(user[:500] + "...")

        # Mock an LLM response
        mock_json = """
        [
            {
                "symbol": "EURUSD",
                "action": "buy_to_enter",
                "profit_target": 1.10500,
                "stop_loss": 1.09500,
                "invalidation_condition": "H1 close below 1.09500",
                "confidence": 0.75,
                "lot_size": 0.1
            },
            {
                "symbol": "USDJPY",
                "action": "hold",
                "invalidation_condition": "Waiting for breakout",
                "confidence": 0.5
            }
        ]
        """
        return f"<reasoning>\nAnalyzed EURUSD and USDJPY.\nEURUSD looks bullish, RSI just bounced from oversold.\nUSDJPY is consolidating, deciding to hold.\n</reasoning>\n```json\n{mock_json}\n```"


async def main():
    """Demonstrates the full usage of the engine."""

    # This requires a running MT5 terminal
    # and requires `pip install pandas pandas-ta numpy`

    logger.info("--- Starting Full Engine Cycle (Mock) ---")

    # 1. Mock dependencies
    # (In a real scenario, these would be the real services)
    mock_llm = MockLLMClient()

    # To make the example run, we need to mock MT5 and the Builder
    # In a real scenario, we would init them like in agent_context_builder.py
    class MockRepo(IMarketDataRepository):
        async def get_symbol_info(self, symbol: str) -> SymbolInfo:
            return SymbolInfo('EURUSD', 0.00001, 5, 10, 100000, 0.01, 0.01, 1.0, -1.0, -0.5, 'EUR', 'USD', 'EUR')

        async def get_last_candles(self, s, t, c) -> List: return [
            MarketCandle(datetime.utcnow(), 1.1, 1.1, 1.1, 1.1, 100)]

        async def get_account_state(self) -> AccountState:
            return AccountState(123, 10000.0, 10000.0, 10000.0, 100.0, 'USD')

        async def get_open_positions(self) -> List: return []

    class MockIndicatorSvc(ITechnicalIndicatorService):
        async def calculate_indicators(self, *args, **kwargs):
            return TechnicalIndicators()  # Return empty indicators

    class MockExecutor(ITradeExecutionService):
        async def place_market_order(self, *args, **kwargs):
            logger.info(f"[MockExecutor] PLACE ORDER: {kwargs}")
            return OrderResult(True, 12345, "Mock order placed", {})

        async def close_position(self, *args, **kwargs):
            logger.info(f"[MockExecutor] CLOSE POSITION: {kwargs}")
            return OrderResult(True, 12346, "Mock position closed", {})

    # 2. Set up real dependencies
    repo = MockRepo()  # In real scenario: MT5MarketDataRepository(connector)
    indicator_svc = MockIndicatorSvc()  # In real scenario: PandasTAIndicatorService()
    builder = AgentContextBuilder(repo, indicator_svc)
    executor = MockExecutor()  # In real scenario: MT5TradeExecutionService(connector)
    persistence = SQLiteTradeRepository(db_path="ex.db")
    await persistence.initialize()

    # 3. Define our desired data
    watch_list = [
        SymbolWatchConfig(symbol="EURUSD", timeframes=[]),
        SymbolWatchConfig(symbol="USDJPY", timeframes=[]),
    ]

    # 4. Initialize the engine
    engine = ForexAgentEngine(
        context_builder=builder,
        trade_executor=executor,
        persistence_repo=persistence,
        llm_client=mock_llm,
        market_repo=repo,  # Pass in the repo
        system_prompt_template="[Your long system prompt goes here]\nYou will receive your Sharpe Ratio at each invocation:"
    )

    # 5. Execute a decision cycle
    full_decision = await engine.decide(watch_list)

    logger.info(f"\n--- Decision Cycle Complete ---")
    logger.info(f"CoT: {full_decision.cot_reasoning}")

    # 6. Execute decisions
    for decision in full_decision.decisions:
        logger.info(f"Validated decision: {decision}")
        if not decision._validation_error:
            # Mock execution
            await engine.execute(decision, full_decision.context)
            logger.info(f"Would execute: {decision.action} {decision.symbol}")
        else:
            logger.error(f"Decision invalid: {decision._validation_error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Note: In a real scenario, you'd need a running asyncio loop
    asyncio.run(main())
    logger.info("To run the engine.py example, uncomment main() and ensure MT5/pandas are installed.")

