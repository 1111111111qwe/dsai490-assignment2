"""
Shared tokenizer and condition encoder for all models.
"""

import re
import numpy as np
from datetime import datetime
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

DAY_MAP = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6,
}

MONTH_MAP = {
    "JAN": 1, "FEB": 2,  "MAR": 3,  "APR": 4,
    "MAY": 5, "JUN": 6,  "JUL": 7,  "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

REVERSE_DAY_MAP   = {v: k for k, v in DAY_MAP.items()}
REVERSE_MONTH_MAP = {v: k for k, v in MONTH_MAP.items()}

# ---------------------------------------------------------------------------
# Vocabulary (used by Seq2Seq and Transformer)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>"]
DAY_TOKENS     = list(DAY_MAP.keys())
MONTH_TOKENS   = list(MONTH_MAP.keys())
LEAP_TOKENS    = ["True", "False"]
DECADE_TOKENS  = [str(i) for i in range(180, 221)]   # 180 .. 220
CHAR_TOKENS    = list("0123456789-")

ALL_TOKENS = (
    SPECIAL_TOKENS
    + DAY_TOKENS
    + MONTH_TOKENS
    + LEAP_TOKENS
    + DECADE_TOKENS
    + CHAR_TOKENS
)

TOKEN_TO_ID: dict = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN: dict = {i: tok for tok, i in TOKEN_TO_ID.items()}

VOCAB_SIZE    = len(TOKEN_TO_ID)
PAD_ID        = TOKEN_TO_ID["<PAD>"]
START_ID      = TOKEN_TO_ID["<START>"]
END_ID        = TOKEN_TO_ID["<END>"]

MAX_INPUT_LEN  = 4   # encoder: DOW MONTH LEAP DECADE
MAX_OUTPUT_LEN = 13  # decoder: <START> + 10 chars (dd-mm-yyyy) + <END> padded to 13


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_PATTERN = re.compile(r"\[(.*?)\]\s\[(.*?)\]\s\[(.*?)\]\s\[(.*?)\]\s(.*)")


def parse_raw_line(line: str) -> Tuple[str, str, str, int, str]:
    """Return (dow, month_name, leap_str, decade_int, date_str)."""
    m = _PATTERN.match(line.strip())
    if not m:
        raise ValueError(f"Cannot parse line: {line!r}")
    dow        = m.group(1)
    month_name = m.group(2)
    leap_str   = m.group(3)
    decade     = int(m.group(4))
    date_str   = m.group(5)
    return dow, month_name, leap_str, decade, date_str


# ---------------------------------------------------------------------------
# Condition vector (CVAE / CGAN: 21-dim float)
# ---------------------------------------------------------------------------

CONDITION_DIM = 21
NOISE_DIM     = 16   # CGAN noise vector size


def build_condition_vector(
    dow: str,
    month_name: str,
    leap_str: str,
    decade: int,
) -> np.ndarray:
    """Return a 21-dim float32 condition vector."""
    dow_vec = np.zeros(7, dtype=np.float32)
    dow_vec[DAY_MAP[dow]] = 1.0

    month_vec = np.zeros(12, dtype=np.float32)
    month_vec[MONTH_MAP[month_name] - 1] = 1.0

    leap_val   = np.float32(1.0 if leap_str == "True" else 0.0)
    decade_val = np.float32((decade - 180) / 40.0)

    return np.concatenate([dow_vec, month_vec, [leap_val], [decade_val]])


# ---------------------------------------------------------------------------
# Sequence tokens (Seq2Seq / Transformer)
# ---------------------------------------------------------------------------

def build_seq_tokens(
    dow: str,
    month_name: str,
    leap_str: str,
    decade: int,
    date_str: str,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns:
        encoder_tokens  : list of 4 token IDs
        decoder_input   : list of MAX_OUTPUT_LEN token IDs (teacher-forcing input)
        decoder_target  : list of MAX_OUTPUT_LEN token IDs (shifted target)
    """
    encoder_tokens = [
        TOKEN_TO_ID[dow],
        TOKEN_TO_ID[month_name],
        TOKEN_TO_ID[leap_str],
        TOKEN_TO_ID[str(decade)],
    ]

    decoder_input  = [START_ID] + [TOKEN_TO_ID[ch] for ch in date_str]
    decoder_target = [TOKEN_TO_ID[ch] for ch in date_str] + [END_ID]

    # Pad to MAX_OUTPUT_LEN
    while len(decoder_input)  < MAX_OUTPUT_LEN:
        decoder_input.append(PAD_ID)
    while len(decoder_target) < MAX_OUTPUT_LEN:
        decoder_target.append(PAD_ID)

    return encoder_tokens, decoder_input, decoder_target


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

MAX_DAYS_IN_DECADE = 3660


def get_offset_in_decade(day: int, month: int, year: int) -> int:
    """Days from the start of the decade containing `year`."""
    decade_start = (year // 10) * 10
    return (datetime(year, month, day) - datetime(decade_start, 1, 1)).days


def offset_to_date(offset: int, decade: int) -> Tuple[int, int, int]:
    """Convert a day-offset within a decade back to (day, month, year)."""
    from datetime import timedelta
    start = datetime(decade * 10, 1, 1)
    dt    = start + timedelta(days=int(offset))
    return dt.day, dt.month, dt.year


def denormalize_date(pred: np.ndarray) -> Tuple[int, int, int]:
    """Convert CVAE/CGAN [0,1] output back to (day, month, year)."""
    day   = int(round(float(pred[0]) * 31))
    month = int(round(float(pred[1]) * 12))
    year  = int(round(float(pred[2]) * 400 + 1800))
    day   = max(1, min(day,   31))
    month = max(1, min(month, 12))
    year  = max(1800, min(year, 2200))
    return day, month, year


def is_valid_date(day: int, month: int, year: int) -> bool:
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False


def format_date(day: int, month: int, year: int) -> str:
    return f"{day}-{month}-{year}"
