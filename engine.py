"""
engine.py
The core Trading Agent Engine.
Connects all services to make decisions.
"""

import asyncio
import logging
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set
from datetime import datetime, timezone, timedelta

# Import all our custom modules
from market import (
    IMarketDataRepository,
    ITradeExecutionService,
    PositionInfo,
    calculate_lot_size
)
from agent_context_builder import (
    AgentContextBuilder,
    AgentFullContext,
    SymbolWatchConfig
)
from persistence import (
    ITradeRepository,
    PerformanceMetrics,
    ClosedTrade
)
from llm_client import ILLMClient
logger = logging.getLogger(__name__)


# ----------------------------- Domain Layer (DDD) -----------------------------

@dataclass
class AgentDecision:
    """
    Structured representation of a single decision from an LLM response.
    Matches the fields in the agent prompt.
    """
    symbol: str
    action: str  # 'buy_to_enter', 'sell_to_enter','update_stop_loss', 'update_take_profit', 'partial_close', 'hold', 'close', wait
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    invalidation_condition: Optional[str] = None
    confidence: Optional[float] = None

    risk_percent: Optional[float] = None
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    close_percent: Optional[int] = None
    reasoning: Optional[str] = None
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
    context: Optional[AgentFullContext] = None


# -------------------------- Engine (Orchestrator) ---------------------------

RE_JSON_FENCE = re.compile(r'(?is)' + r"```json\s*(\[\s*\{.*?\}\s*\])\s*```")# Find Array
RE_JSON_ARRAY = re.compile(r'(?is)\[\s*\{.*?\}\s*\]') # Find Array
RE_JSON_FENCE_OBJ = re.compile(r'(?is)' + r"```json\s*(\{.*?\})\s*```") # Find object
RE_JSON_OBJ = re.compile(r'(?is)(\{.*?\})') # Find object

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

        self.allowed_actions = {"buy_to_enter", "sell_to_enter", "partial_close", "update_stop_loss", "update_take_profit", "hold", "close", "wait"}
        # Runtime Tracking ---
        self._start_time = datetime.utcnow()
        self._cycle_count = 0
        # Time Synchronization ---
        self._server_time_offset = timedelta(0)
        self._is_time_synced = False

    async def _run_time_sync(self):
        """
        Performs the one-time network call to calculate and set the time offset.
        """
        try:
            # We assume self._market_repo.get_tick is available and async
            tick = await self._market_repo.get_tick("EURUSD")

            if tick and hasattr(tick, 'time'):
                broker_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                self._server_time_offset = broker_time - now_utc
                logger.info(f"Time Sync Complete. Broker is {self._server_time_offset} from UTC.")
            else:
                logger.warning("Could not sync time. Defaulting to UTC+0.")

        except Exception as e:
            logger.error(f"Time sync failed: {e}")

    def _get_estimated_server_time(self) -> datetime:
        """
        Returns the current Broker Server Time using the cached offset.
        Zero API calls. Extremely fast.
        """
        return datetime.now(timezone.utc) + self._server_time_offset

    def _build_system_prompt(self) -> str:
        """
        Builds the system prompt, dynamically injecting the current Sharpe Ratio.
        """
        prompt = self._system_prompt_template

        prompt += (
            "\n\n"  # Added a newline for separation if prompt isn't empty
            "# Hard Constraints (Risk Controls)\n\n"
            "1. Risk-reward ratio: must be ≥ 1:3 (e.g., risk 1% to target 3%+ reward)\n"
            "2. Maximum open positions: 3 assets (quality over quantity)\n"
            "3. Margin usage: Total usage ≤ 90%\n"
            "# Output Format (Strictly follow)\n\n"
            "**You MUST use the XML tags <reasoning> and <decision> to separate the chain-of-thought and the decision JSON to avoid parsing errors.**\n\n"
            "## Format Requirements\n\n"
            "<reasoning>\n"
            "Your chain-of-thought analysis...\n"
            "- Keep the analysis concise and focused\n"
            "</reasoning>\n\n"
            "<decision>\n"
            "```json\n[\n"
            "{\"symbol\": \"EURUSD\", \"action\": \"sell_to_enter\", \"risk_percent\": 0.01, \"stop_loss\": 1.59594, \"take_profit\": 1.59002, \"confidence\": 85, \"reasoning\": \"example\"},\n"
            "{\"symbol\": \"XAUUSD\", \"action\": \"update_stop_loss\", \"new_stop_loss\": 4000.50, \"reasoning\": \"move stop to breakeven\"},\n"
            "{\"symbol\": \"USDJPY\", \"action\": \"close\", \"reasoning\": \"take profit\"}\n"
            "]\n```\n"
            "</decision>\n\n"
            "## Field Descriptions\n\n"
            "- `action`: buy_to_enter | sell_to_enter | close | update_stop_loss | update_take_profit | partial_close | hold | wait\n"
            "- `confidence`: 0-100 (recommend ≥75 for open position suggestions)\n"
            "- Required for opens: risk_percent, stop_loss, take_profit, confidence, reasoning\n"
            "- For update_stop_loss: provide `new_stop_loss` (not `stop_loss`)\n"
            "- For update_take_profit: provide `new_take_profit` (not `take_profit`)\n"
            "- For partial_close: provide `close_percent` (0-100)\n\n"
        )

        return prompt

    def _build_user_prompt(self, context: AgentFullContext, metrics: PerformanceMetrics, closed_trades: List[ClosedTrade]) -> str:
        """
        Formats the rich AgentFullContext object into a text input for the LLM.
        """
        server_time = self._get_estimated_server_time()
        lines = []
        processed_symbols: Set[str] = set()
        # --- System Status Header ---
        now = datetime.utcnow()
        active_sessions = self._get_market_sessions(now)
        elapsed = now - self._start_time
        runtime_minutes = int(elapsed.total_seconds() / 60)
        lines.append("## System Status")
        lines.append(
            f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
            f"Sessions: [{active_sessions}] | "
            f"Cycle: #{self._cycle_count} | "
            f"Runtime: {runtime_minutes} minutes"
        )
        lines.append("---")
        # ---------------------------------
        # --- 2. Performance Metrics ---
        if metrics.total_trades > 0:
            lines.append("## 📈 Performance Metrics (Closed Trades)")
            lines.append(
                f"Risk-Adj. Returns: Sharpe {metrics.sharpe_ratio:.2f} | "
                f"Sortino {metrics.sortino_ratio:.2f} | "
                f"Max DD {metrics.max_drawdown_pct:.2f}%"
            )
            lines.append(
                f"Profit Factor: {metrics.profit_factor:.2f} | "
                f"Win Rate: {metrics.win_rate:.1f}% | "
                f"Avg. Return: {metrics.average_return_pct:.2f}%"
            )
            lines.append(
                f"Total Trades: {metrics.total_trades} | "
                f"Net PnL: {metrics.total_net_profit:+.2f}"
            )
            lines.append("---")

        if closed_trades:
            lines.append(f"## Recent Trade History (Last {len(closed_trades)})")

            for i, trade in enumerate(closed_trades):
                # Basic Trade Info
                duration = trade.close_time - trade.open_time
                dur_str = str(duration).split('.')[0]  # Remove microseconds

                header = (
                    f"{i + 1}. {trade.symbol} ({trade.action}) | "
                    f"Profit: {trade.profit:+.2f} ({trade.return_pct:+.2f}%) | "
                    f"Entry: {trade.open_price:.5f} | Exit: {trade.close_price:.5f} | "
                    f"Duration: {dur_str}"
                )
                lines.append(header)

                # Context & Reasoning
                lines.append(f"   - Open Time: {trade.open_time.strftime('%Y-%m-%d %H:%M')}")
                lines.append(f"   - Open Reason: {trade.open_reasoning or 'N/A'}")
                lines.append(f"   - Initial Plan: SL {trade.initial_sl:.5f} | TP {trade.initial_tp:.5f}")
                lines.append(f"   - Close Time: {trade.close_time.strftime('%Y-%m-%d %H:%M')}")
                lines.append(f"   - Close Reason: {trade.close_reasoning or 'N/A'}")

                # Modifications History
                if trade.modifications:
                    lines.append("   - Modifications:")
                    for mod in trade.modifications:
                        mod_time = mod.timestamp.strftime('%H:%M')
                        val_change = f"{mod.old_value:.5f} -> {mod.new_value:.5f}"
                        lines.append(f"     [{mod_time}] {mod.modification_type}: {val_change} | Why: {mod.reasoning}")

                lines.append("")  # Spacing
            lines.append("---")
        # ---------------------------------------------

        # --- 4. Account & Positions ---
        acc = context.account_state
        equity = acc.equity if acc.equity > 0 else 1.0
        balance_pct = (acc.balance / equity) * 100
        pnl_pct = ((acc.equity - acc.balance) / acc.balance) * 100 if acc.balance > 0 else 0
        margin_used_pct = ((acc.equity - acc.free_margin) / equity) * 100

        lines.append(f"## Account")
        lines.append(
            f"Account: Equity {acc.equity:.2f} | "
            f"Balance {acc.balance:.2f} ({balance_pct:.1f}%) | "
            f"PnL {pnl_pct:+.2f}% | "
            f"Margin {margin_used_pct:.1f}% | "
            f"Positions {len(context.open_positions)}"
        )
        lines.append("---")

        # --- 3. Positions (with full market data) ---
        if context.open_positions:
            lines.append("## Current Positions")

            for i, pos in enumerate(context.open_positions):
                # A. Calculate Holding Duration
                # Assumes pos.open_time is a datetime object. If it's a timestamp, use datetime.fromtimestamp(pos.open_time)
                duration_str = ""
                if pos.open_time:
                    # Ensure pos.open_time is compatible (aware vs naive)
                    # If you get "can't subtract offset-naive", normalize them:
                    t_open = pos.open_time
                    if t_open.tzinfo is None and server_time.tzinfo is not None:
                        t_open = t_open.replace(tzinfo=timezone.utc)
                    elif t_open.tzinfo is not None and server_time.tzinfo is None:
                        server_time = server_time.replace(tzinfo=timezone.utc)

                    # Now calculate
                    delta = server_time - t_open
                    total_minutes = int(delta.total_seconds() / 60)

                    # Handle edge case where tick time is slightly behind open time (rare execution lag)
                    if total_minutes < 0:
                        total_minutes = 0

                    if total_minutes < 60:
                        duration_str = f" | Holding {total_minutes} minutes"
                    else:
                        hours = total_minutes // 60
                        mins = total_minutes % 60
                        duration_str = f" | Holding {hours} hours {mins} minutes"

                # B. Calculate Metrics (Forex Adaptation)
                pos_type = "BUY" if pos.type == 0 else "SELL"

                # 1. The Tier Metric (PnL based on Account Balance)
                # This decides WHICH Take Profit rule applies (Tier 1, Tier 2, etc.)
                balance = context.account_state.balance if context.account_state.balance > 0 else 1.0
                nav_pnl_pct = (pos.profit / balance) * 100

                # 2. The Trigger Metric (Retracement from Peak)
                # This decides IF the rule triggers
                valid_peak = max(pos.peak_profit, pos.profit) # TODO instant peak pull update

                # Avoid division by zero if peak is 0 or negative
                if valid_peak > 0:
                    retracement_pct = ((valid_peak - pos.profit) / valid_peak) * 100
                else:
                    retracement_pct = 0.0

                # 3. Calculate R-Multiple (Risk Efficiency)
                # "For every dollar I risked, how many did I make?"
                r_multiple_str = "N/A"
                if pos.stop_loss > 0:
                    # Calculate distance based on direction
                    if pos.type == 0:  # BUY
                        risk_dist = pos.open_price - pos.stop_loss
                        gain_dist = pos.current_price - pos.open_price
                    else:  # SELL
                        risk_dist = pos.stop_loss - pos.open_price
                        gain_dist = pos.open_price - pos.current_price

                    # Avoid division by zero
                    if risk_dist > 0:
                        r_val = gain_dist / risk_dist
                        r_multiple_str = f"{r_val:.2f}R"

                # Calculate Peak NAV PCT for the prompt
                peak_nav_pct = (valid_peak / balance) * 100

                pos_line = (
                    f"{i + 1}. {pos.symbol} {pos_type} | "
                    f"Entry {pos.open_price:.5f} Current {pos.current_price:.5f} | "
                    f"Lots {pos.volume:.2f} | "
                    f"R-Mult: {r_multiple_str} | "
                    f"PnL: {nav_pnl_pct:+.2f}% (Peak {peak_nav_pct:.2f}%)| "  # Matches your "Profit 1-3%" tiers
                    f"Retracement: {retracement_pct:.1f}% | "  # Matches your "50% retracement" triggers
                    f"PnL Amt: {pos.profit:+.2f} (Peak: {valid_peak:.2f}) | "
                    f"Margin {pos.margin:+.0f} | "
                    f"Swap {pos.swap:+.2f} | "
                    f"SL {pos.stop_loss:.5f} TP {pos.take_profit:.5f}" 
                    f"{duration_str}"
                )

                lines.append(pos_line)

                # --- EMBEDDED MARKET DATA FOR THIS POSITION ---
                # This ensures the agent sees the chart data immediately after the position data
                sym_context = context.market_context.get(pos.symbol)
                if sym_context:
                    processed_symbols.add(pos.symbol)  # Mark as shown
                    info = sym_context.symbol_info

                    # Exact format from Market Data section
                    lines.append(f"\n   ### Symbol: {pos.symbol}")
                    lines.append(
                        f"   Spread: {info.spread} points | Swap: {info.swap_long} (L) / {info.swap_short} (S)")
                    lines.append(
                        f"   Tick Value: {info.trade_tick_value} {acc.currency} | Point: {info.point} | Pip: {info.point * 10} (1 Pip = 10 Points)")

                    for tf_data in sym_context.market_data:
                        tf_label = self.get_timeframe_label(tf_data.timeframe)
                        lines.append(f"\n     Timeframe: {tf_label} (Last 10 data points)")

                        pattern_slice = -100
                        mid_prices_list = [c.close for c in tf_data.candles[pattern_slice:]]
                        mid_prices_str = ", ".join([f"{p:.5f}" for p in mid_prices_list])
                        lines.append(f"     Mid_Prices (Last {len(mid_prices_list)}): [{mid_prices_str}]")

                        # Volume
                        vol_lookback = 20
                        volumes = [c.tick_volume for c in tf_data.candles]
                        if len(volumes) >= vol_lookback:
                            last_20_vols = volumes[-(vol_lookback + 1):-1]
                            if len(last_20_vols) > 0:
                                avg_vol = sum(last_20_vols) / len(last_20_vols)
                            else:
                                avg_vol = 1
                            current_vol = volumes[-1]
                            if avg_vol > 0:
                                vol_ratio = current_vol / avg_vol
                            else:
                                vol_ratio = 0.0
                            vol_str = f"     Volume_Analysis: [Current: {current_vol}, Avg_20: {int(avg_vol)}, Ratio: {vol_ratio:.2f}x]"
                            lines.append(vol_str)
                        else:
                            lines.append("     Volume_Analysis: [Not enough data for Avg]")

                        # Candles & Indicators
                        slice_len = -10
                        candles_str = [
                            f"C: {c.close:.5f} H: {c.high:.5f} L: {c.low:.5f} O: {c.open:.5f} V: {c.tick_volume}"
                            for c in tf_data.candles[slice_len:]
                        ]
                        lines.append(f"      Candles: [{', '.join(candles_str)}]")

                        inds = tf_data.indicators
                        if inds.ema_fast:
                            lines.append(
                                f"      EMA_Fast: [{', '.join([f'{x:.5f}' for x in inds.ema_fast[slice_len:]])}]")
                        if inds.ema20:
                            lines.append(
                                f"      EMA (20-Period): [{', '.join([f'{x:.5f}' for x in inds.ema20[slice_len:]])}]")
                        if inds.ema_slow:
                            lines.append(
                                f"      EMA_Slow: [{', '.join([f'{x:.5f}' for x in inds.ema_slow[slice_len:]])}]")
                        if inds.ema50:
                            lines.append(
                                f"      EMA (50-Period): [{', '.join([f'{x:.5f}' for x in inds.ema50[slice_len:]])}]")
                        if inds.macd:
                            lines.append(f"      MACD:     [{', '.join([f'{x:.5f}' for x in inds.macd[slice_len:]])}]")
                        if inds.macd_signal:
                            lines.append(
                                f"      MACD Signal:   [{', '.join([f'{x:.5f}' for x in inds.macd_signal[slice_len:]])}]")
                        if inds.macd_histogram:
                            lines.append(
                                f"      MACD Histogram:   [{', '.join([f'{x:.5f}' for x in inds.macd_histogram[slice_len:]])}]")
                        if inds.rsi:
                            lines.append(f"      RSI:      [{', '.join([f'{x:.2f}' for x in inds.rsi[slice_len:]])}]")
                        if inds.rsi7:
                            lines.append(
                                f"      RSI (7-Period):      [{', '.join([f'{x:.2f}' for x in inds.rsi7[slice_len:]])}]")
                        if inds.rsi14:
                            lines.append(
                                f"      RSI (14-Period):      [{', '.join([f'{x:.2f}' for x in inds.rsi14[slice_len:]])}]")
                        if inds.atr:
                            lines.append(
                                f"      ATR (14-Period):      [{', '.join([f'{x:.5f}' for x in inds.atr[slice_len:]])}]")

                lines.append("")  # Empty line between positions

        else:
            lines.append("## Current Positions")
            lines.append("Current Positions: None\n")

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

        # --- Market Data (Remaining Symbols) ---
        lines.append("## Market Data & Indicators (Oldest -> Newest)")
        for symbol, sym_context in context.market_context.items():
            # Skip if we already showed this symbol in the positions section
            if symbol in processed_symbols:
                continue
            info = sym_context.symbol_info
            lines.append(f"\n### Symbol: {symbol}")
            lines.append(f"Spread: {info.spread} points | Swap: {info.swap_long} (L) / {info.swap_short} (S)")
            lines.append(f"Tick Value: {info.trade_tick_value} {acc.currency} | Point: {info.point} | Pip: {info.point*10} (1 Pip = 10 Points)")

            for tf_data in sym_context.market_data:
                tf_label = self.get_timeframe_label(tf_data.timeframe)
                lines.append(f"\n  Timeframe: {tf_label} (Last 10 data points)")

                # 1. Define Lookback: Use last 100 for pattern recognition (balanced token usage)
                # If you want the full 200, change this to -200
                pattern_slice = -100

                # 2. Extract Close Prices (User defined "Mid_Prices" as Close prices)
                # We use proper slicing to ensure it works even if you have fewer than 100 candles
                mid_prices_list = [c.close for c in tf_data.candles[pattern_slice:]]

                # 3. Format: Create a clean, comma-separated string inside brackets
                # formatting to .5f keeps the precision needed for Forex/Indices
                mid_prices_str = ", ".join([f"{p:.5f}" for p in mid_prices_list])

                # 4. Append to prompt lines
                lines.append(f"   Mid_Prices (Last {len(mid_prices_list)}): [{mid_prices_str}]")

                # --- VOLUME ANALYSIS BLOCK ---

                # 1. Define Volume Lookback (Standard = 20)
                vol_lookback = 20

                # 2. Get the list of volumes (ensure we have enough data)
                volumes = [c.tick_volume for c in tf_data.candles]

                if len(volumes) >= vol_lookback:
                    # Calculate Average of the last 20 candles (excluding the current incomplete one if you prefer,
                    # but usually including the immediate history is fine)
                    # Note: tf_data.candles[-21:-1] gets the previous 20 closed candles
                    # (safest to avoid current forming candle skewing the average)
                    last_20_vols = volumes[-(vol_lookback + 1):-1]

                    # Safety check for division by zero
                    if len(last_20_vols) > 0:
                        avg_vol = sum(last_20_vols) / len(last_20_vols)
                    else:
                        avg_vol = 1

                    current_vol = volumes[-1]  # The volume of the latest candle

                    # Calculate Ratio (RVol)
                    if avg_vol > 0:
                        vol_ratio = current_vol / avg_vol
                    else:
                        vol_ratio = 0.0

                    # 3. Create the string
                    # We provide the raw average AND the explicit ratio for the AI
                    vol_str = f"   Volume_Analysis: [Current: {current_vol}, Avg_20: {int(avg_vol)}, Ratio: {vol_ratio:.2f}x]"

                    # 4. Append BEFORE candles
                    lines.append(vol_str)

                else:
                    # Fallback if not enough data
                    lines.append("   Volume_Analysis: [Not enough data for Avg]")

                # --- END VOLUME BLOCK ---

                # Slicing: Only show the last 10 data points
                slice_len = -10

                # Format candles
                candles_str = [
                    f"C: {c.close:.5f} H: {c.high:.5f} L: {c.low:.5f} O: {c.open:.5f} V: {c.tick_volume}"
                    for c in tf_data.candles[slice_len:]
                ]
                lines.append(f"    Candles: [{', '.join(candles_str)}]")

                # Format indicators
                inds = tf_data.indicators
                if inds.ema_fast:
                    lines.append(f"    EMA_Fast: [{', '.join([f'{x:.5f}' for x in inds.ema_fast[slice_len:]])}]")
                if inds.ema20:
                    lines.append(f"    EMA (20-Period): [{', '.join([f'{x:.5f}' for x in inds.ema20[slice_len:]])}]")
                if inds.ema_slow:
                    lines.append(f"    EMA_Slow: [{', '.join([f'{x:.5f}' for x in inds.ema_slow[slice_len:]])}]")
                if inds.ema50:
                    lines.append(f"    EMA (50-Period): [{', '.join([f'{x:.5f}' for x in inds.ema50[slice_len:]])}]")
                if inds.macd:
                    lines.append(f"    MACD:     [{', '.join([f'{x:.5f}' for x in inds.macd[slice_len:]])}]")
                if inds.macd_signal:
                    lines.append(f"    MACD Signal:   [{', '.join([f'{x:.5f}' for x in inds.macd_signal[slice_len:]])}]")
                if inds.macd_histogram:
                    lines.append(f"    MACD Histogram:   [{', '.join([f'{x:.5f}' for x in inds.macd_histogram[slice_len:]])}]")
                if inds.rsi:
                    lines.append(f"    RSI:      [{', '.join([f'{x:.2f}' for x in inds.rsi[slice_len:]])}]")
                if inds.rsi7:
                    lines.append(f"    RSI (7-Period):      [{', '.join([f'{x:.2f}' for x in inds.rsi7[slice_len:]])}]")
                if inds.rsi14:
                    lines.append(f"    RSI (14-Period):      [{', '.join([f'{x:.2f}' for x in inds.rsi14[slice_len:]])}]")
                if inds.atr:
                    lines.append(f"    ATR (14-Period):      [{', '.join([f'{x:.5f}' for x in inds.atr[slice_len:]])}]")


        lines.append("\n---\nAnalyze the data. Provide your reasoning in <reasoning> tags, then provide a JSON array of decisions.")
        print("\n".join(lines))
        return "\n".join(lines)

    def get_timeframe_label(self,tf_int):
        tf_map = {
            1: "M1",
            3: "M3",
            5: "M5",
            15: "M15",
            30: "M30",
            60: "H1",
            16385: "H1",  # MT5 specific integer for H1
            240: "H4",
            16388: "H4",  # MT5 specific integer for H4
            1440: "D1",
            16408: "D1",  # MT5 specific integer for D1
            43200: "MN1"
        }
        # Returns the label if found, otherwise returns the original number as a string
        return tf_map.get(tf_int, str(tf_int))

    def _get_market_sessions(self, current_time_utc):
        h = current_time_utc.hour
        s = []
        if h >= 21 or h < 6: s.append("Sydney")
        if 0 <= h < 9: s.append("Tokyo")
        if 7 <= h < 16: s.append("London")
        if 12 <= h < 21: s.append("New York")
        return ", ".join(s) if s else "Quiet"

    def _extract_cot(self, raw: str) -> str:
        """
        Extracts the Chain of Thought (CoT) using the <reasoning> tag or all text before the JSON.
        """
        # 1. Try <reasoning> tag
        match = RE_REASONING_TAG.search(raw)
        if match:
            return match.group(1).strip()

        # 2. Fallback: get all content before any JSON block (array or object)
        json_match = (
                RE_JSON_FENCE.search(raw) or
                RE_JSON_ARRAY.search(raw) or
                RE_JSON_FENCE_OBJ.search(raw) or
                RE_JSON_OBJ.search(raw)
        )
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
        is_object = False

        # 1. Try ```json [ ... ]``` (fenced array)
        match = RE_JSON_FENCE.search(s)
        if match:
            json_text = match.group(1)
        else:
            # 2. Try [ ... ] (raw array)
            match = RE_JSON_ARRAY.search(s)
            if match:
                json_text = match.group(0)
            else:
                # 3. Try ```json { ... }``` (fenced object)
                match = RE_JSON_FENCE_OBJ.search(s)
                if match:
                    json_text = match.group(1)
                    is_object = True
                else:
                    # 4. Try { ... } (raw object)
                    match = RE_JSON_OBJ.search(s)
                    if match:
                        json_text = match.group(0)
                        is_object = True
        if not json_text:
            logger.warning("No JSON block (array or object) found in LLM response.")
            return []

        try:
            # Parse JSON
            data = json.loads(json_text)
            decisions = []

            # If it was a single object, wrap it in a list
            if is_object:
                data = [data]

            for item in data:
                # Check if it's a valid dict
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid JSON item: {item}")
                    continue

                decisions.append(AgentDecision(
                    symbol=item.get("symbol"),
                    action=item.get("action"),
                    take_profit=item.get("take_profit"),
                    stop_loss=item.get("stop_loss"),
                    invalidation_condition=item.get("invalidation_condition"),
                    confidence=item.get("confidence", 0.0),
                    lot_size=item.get("lot_size", 0.0),  # LLM suggestion
                    new_stop_loss=item.get("new_stop_loss"),
                    new_take_profit=item.get("new_take_profit"),
                    risk_percent=item.get("risk_percent"),
                    close_percent=item.get("close_percent"),
                    reasoning=item.get("reasoning")
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
            if not all([d.symbol, d.stop_loss, d.take_profit, d.confidence]):
                d._validation_error = "Open decision is missing symbol, stop_loss, profit_target, or confidence"
                return

            # 2. Get required context
            acc = ctx.account_state
            sym_ctx = ctx.market_context.get(d.symbol)
            if not sym_ctx:
                d._validation_error = f"Market context for Symbol {d.symbol} is not available"
                return

            info = sym_ctx.symbol_info

            # 3. Get live tick (for SL points calculation)
            try:
                tick = await self._market_repo.get_tick(d.symbol)
                if not tick:
                    d._validation_error = f"Failed to get live tick for {d.symbol}"
                    return

                # Use bid/ask depending on direction
                live_price = tick.ask if d.action == "buy_to_enter" else tick.bid


            except Exception as e:
                d._validation_error = f"Failed to get live price: {e}"
                return

            # 4. Calculate risk parameters
            stop_loss_points = abs(live_price - d.stop_loss) / info.point
            if stop_loss_points == 0:
                d._validation_error = "Stop loss points calculated to 0 (SL price equals market price)"
                return



            # 6. Calculate Lot Size
            calculated_lot = calculate_lot_size(
                account_balance=acc.balance,  # Use balance, or free_margin
                risk_percent=d.risk_percent,
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
        if not self._is_time_synced:
            logger.info("First cycle detected. Running time synchronization...")
            await self._run_time_sync()  # Await the network call
            self._is_time_synced = True  # Set flag so it never runs again
        # ----------------------------------------------------
        # --- NEW: Increment Cycle ---
        self._cycle_count += 1
        try:
            # 1. Build context
            logger.info("Building context...")
            context = await self._builder.build_context(watch_list)

            # --- New: Fetch Full Data ---
            logger.info("Fetching history...")
            metrics_task = self._persistence.get_performance_metrics()
            history_task = self._persistence.get_closed_trades_history(limit=10)

            metrics, closed_trades = await asyncio.gather(metrics_task, history_task)

            # 3. Build prompts
            logger.info("Building prompts...")
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(context, metrics, closed_trades)

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

    async def execute(self, decision: AgentDecision, context: AgentFullContext, reasoning: str): # TODO rename reasoning to CoT and store it in db
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
                    take_profit=decision.take_profit,
                    comment=f"Agent v1 | {decision.confidence:.0%}"
                )
                logger.info(f"Execution result: {result}")
                if result.success and result.ticket:
                    # We must wait for the position to appear in MT5
                    await asyncio.sleep(2.0)  # Wait 2s for broker propagation
                    pos = await self._find_position_by_ticket(result.ticket, context)
                    if pos:
                        await self._persistence.store_new_open_trade(
                            position=pos,
                            account_balance=context.account_state.balance,
                            initial_sl=decision.stop_loss,
                            initial_tp=decision.take_profit,
                            reasoning=decision.reasoning,
                            confidence=decision.confidence
                        )
                    else:
                        logger.error(f"Could not find newly opened position {result.ticket} to store in DB!")
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

                result = await self._executor.close_position(pos_to_close, comment="") # TODO
                logger.info(f"Close result: {result}")
                if result.success:
                    # The trade is closed. We must wait for the broker
                    # to process the closure before fetching history.
                    await asyncio.sleep(2.0)  # Wait 2s

                    # Fetch the historical trade data
                    closed_trade_history = await self._market_repo.get_historical_trade(pos_to_close.ticket)

                    if closed_trade_history:
                        await self._persistence.update_trade_as_closed(
                            ticket=pos_to_close.ticket,
                            close_time=closed_trade_history.close_time,
                            close_price=closed_trade_history.close_price,
                            profit=closed_trade_history.profit,
                            swap=closed_trade_history.swap,
                            close_reasoning= decision.reasoning
                        )
                        logger.info(f"Successfully closed and logged trade {pos_to_close.ticket}")
                    else:
                        logger.error(f"Closed trade {pos_to_close.ticket} but could not fetch its history to log.")

            elif action == "hold":
                logger.info(f"--- EXECUTING HOLD {decision.symbol} ---")

            elif action == "update_stop_loss":
                if decision.new_stop_loss is None:
                    logger.error(f"Missing new_stop_loss for {decision.symbol}")
                    return

                # Find position logic...
                pos = next((p for p in context.open_positions if p.symbol == decision.symbol), None)
                if pos:
                    logger.info(f"Updating SL for {decision.symbol} to {decision.new_stop_loss} Reason: {decision.reasoning}")
                    await self._executor.update_position_sl(pos.ticket, stop_loss=decision.new_stop_loss)
                    # --- NEW: Log Modification ---
                    await self._persistence.log_modification(
                        pos.ticket, "UPDATE_SL", pos.stop_loss, decision.new_stop_loss, decision.reasoning
                    )

            elif action == "update_take_profit":
                if decision.new_take_profit is None:
                    logger.error(f"Missing new_take_profit for {decision.symbol}")
                    return

                # Find position logic...
                pos = next((p for p in context.open_positions if p.symbol == decision.symbol), None)
                if pos:
                    logger.info(f"Updating TP for {decision.symbol} to {decision.new_take_profit} Reason: {decision.reasoning}")
                    await self._executor.update_position_tp(pos.ticket, take_profit=decision.new_take_profit)
                    # --- NEW: Log Modification ---
                    await self._persistence.log_modification(
                        pos.ticket, "UPDATE_TP", pos.take_profit, decision.new_take_profit, decision.reasoning
                    )

            elif action == "partial_close":
                if decision.close_percent is None:
                    logger.error(f"Missing close_percent for {decision.symbol}")
                    return

                # Find position logic...
                pos = next((p for p in context.open_positions if p.symbol == decision.symbol), None)
                if pos:
                    logger.info(f"Partial close {decision.symbol} by {decision.close_percent} Reason: {decision.reasoning}")
                    lot_to_close = pos.volume * decision.close_percent / 100
                    await self._executor.partial_close_position(pos.ticket, lot_to_close=lot_to_close, comment=f"{decision.reasoning}")
                    # --- NEW: Log Modification ---
                    await self._persistence.log_modification(
                        pos.ticket, "PARTIAL_CLOSE", pos.volume, pos.volume - lot_to_close,
                        f"Closed {decision.close_percent}%: {decision.reasoning}"
                    )

        except Exception as e:
            logger.exception(f"Execution failed for {action} {decision.symbol}")

    async def _find_position_by_ticket(self, ticket: int, context: AgentFullContext) -> Optional[PositionInfo]:
        """
        Finds a position by its ticket.
        This is tricky because the context is *stale*. We must re-fetch.
        """
        # Re-fetch open positions
        open_positions = await self._market_repo.get_open_positions()

        for pos in open_positions:
            if pos.ticket == ticket:
                return pos

        # It might be in the original context if we're fast enough
        for pos in context.open_positions:
            if pos.ticket == ticket:
                return pos

        return None
