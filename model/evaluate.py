"""
Evaluation utilities: constraint satisfaction rate (CSR).
"""

from datetime import datetime
from typing import Tuple

from tokenizer import DAY_MAP, MONTH_MAP, is_valid_date


def check_constraints(
    generated_date: Tuple[int, int, int],
    expected_dow: str,
    expected_month: str,
    expected_leap: bool,
    expected_decade: int,
) -> bool:
    """
    Returns True if the generated date satisfies all four conditions.

    Args:
        generated_date : (day, month, year)
        expected_dow   : e.g. "MON"
        expected_month : e.g. "DEC"
        expected_leap  : True / False
        expected_decade: e.g. 196 (means 1960-1969)
    """
    day, month, year = generated_date

    if not is_valid_date(day, month, year):
        return False

    dt = datetime(year, month, day)

    weekday     = dt.strftime("%a").upper()[:3]
    month_name  = dt.strftime("%b").upper()
    is_leap     = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
    decade      = year // 10

    return (
        weekday    == expected_dow    and
        month_name == expected_month  and
        is_leap    == expected_leap   and
        decade     == expected_decade
    )


def constraint_satisfaction_rate(predictions, conditions) -> float:
    """
    Compute CSR over a list of predictions.

    Args:
        predictions : list of (day, month, year) tuples
        conditions  : list of (dow, month_name, leap_bool, decade_int) tuples
    Returns:
        float in [0, 1]
    """
    assert len(predictions) == len(conditions)
    passed = sum(
        check_constraints(pred, *cond)
        for pred, cond in zip(predictions, conditions)
    )
    return passed / len(predictions)
