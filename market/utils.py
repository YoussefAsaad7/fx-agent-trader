"""
Utility Functions
-----------------

This file contains stateless, pure helper functions used
throughout the package, primarily for calculations and
data sanitization.
"""

import logging
import math
import re
from decimal import Decimal

logger = logging.getLogger(__name__)


def round_to_step(value: float, step: float) -> float:
    """
    Rounds a value *down* to the nearest valid step size.
    e.g., round_to_step(0.128, 0.01) -> 0.12
    e.g., round_to_step(0.12, 0.05) -> 0.10
    e.g., round_to_step(0.14, 0.05) -> 0.10
    e.g., round_to_step(0.15, 0.05) -> 0.15
    """
    if step <= 0:
        return value

    try:
        # Get number of decimal places from the step
        step_str = str(step)
        if 'e-' in step_str:
            decimals = int(step_str.split('e-')[-1])
        elif '.' in step_str:
            decimals = len(step_str.split('.')[-1])
        else:
            decimals = 0

        # Use floor to always round down to the nearest step
        quantized = math.floor(value / step) * step

        # Round to the precision of the step to avoid floating point issues
        return round(quantized, decimals)

    except Exception:
        # Fallback for complex steps
        return math.floor(value / step) * step


def calculate_lot_size(account_balance: float,
                       risk_percent: float,
                       stop_loss_pips: float,
                       pip_value_per_lot: float,
                       volume_step: float) -> float:
    """
    Deterministic lot size calculation.

    Args:
        account_balance (float): Current account balance.
        risk_percent (float): Desired risk (e.g., 0.01 for 1%).
        stop_loss_pips (float): Stop loss distance in pips (e.g., 50.0).
                                   FOR MT5, THIS IS `point` units.
        pip_value_per_lot (float): The value of 1 pip for 1.0 lot.
                                   FOR MT5, THIS IS `symbol_info.trade_tick_value`.
        volume_step (float): The minimum lot step (e.g., 0.01 or 1.0).

    Returns:
        float: The correctly rounded lot size.
    """
    if stop_loss_pips <= 0 or pip_value_per_lot <= 0:
        logger.warning(f"Invalid inputs to calculate_lot_size: SL_pips={stop_loss_pips}, PipVal={pip_value_per_lot}")
        return 0.0
    if volume_step <= 0:
        raise ValueError("volume_step must be positive")

    try:
        risk_usd = Decimal(str(account_balance)) * Decimal(str(risk_percent))
        denom = Decimal(str(stop_loss_pips)) * Decimal(str(pip_value_per_lot))

        if denom == 0:
            logger.error("Lot size calculation failed: denominator is zero.")
            return 0.0

        lot = float(risk_usd / denom)
    except Exception as e:
        logger.error(f"Error in lot size calculation: {e}")
        return 0.0

    # Use the robust round_to_step function to round *down*
    return round_to_step(lot, volume_step)


def safe_comment(comment: str) -> str:
    """Ensure MT5 comment field is ASCII-only and <=31 bytes."""
    if not comment:
        return ""
    # Convert to str in case it’s not
    comment = str(comment)
    # Remove non-printable/non-ASCII characters
    comment = re.sub(r"[^\x20-\x7E]", "", comment)
    # Truncate to max 31 bytes
    while len(comment.encode("ascii", "ignore")) > 31:
        comment = comment[:-1]
    return comment