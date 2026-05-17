"""
Inference entry point for DSAI 490 Assignment 2.

Usage:
    python predict.py -i path/to/input.txt -o path/to/output.txt

The default model used is the Transformer (best token accuracy).
Switch MODEL_CHOICE below to use a different model.
"""

import argparse
import sys
import os
import re

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Paths
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from tokenizer import (
    parse_raw_line,
    build_condition_vector, build_seq_tokens,
    VOCAB_SIZE, TOKEN_TO_ID, ID_TO_TOKEN,
    PAD_ID, START_ID, END_ID,
    MAX_INPUT_LEN, MAX_OUTPUT_LEN,
    CONDITION_DIM, NOISE_DIM,
    denormalize_date, offset_to_date, is_valid_date, format_date,
    DAY_MAP, MONTH_MAP,
)

# ---------------------------------------------------------------------------
# Choose which model to use for inference: "transformer" | "seq2seq" | "cvae" | "cgan"
MODEL_CHOICE = "transformer"

# Weight paths
TRANSFORMER_WEIGHTS = os.path.join(_DIR, "transformer", "weights", "transformer.weights.h5")
SEQ2SEQ_WEIGHTS     = os.path.join(_DIR, "seq2seq",     "weights", "seq2seq.weights.h5")
CVAE_WEIGHTS        = os.path.join(_DIR, "cvae",        "weights", "cvae_decoder.weights.h5")
CGAN_WEIGHTS        = os.path.join(_DIR, "cgan",        "weights", "generator.weights.h5")

NOISE_DIM_CGAN  = 16
LATENT_DIM_CVAE = 8

# ---------------------------------------------------------------------------


def _build_seq_model(model_type: str) -> tf.keras.Model:
    if model_type == "transformer":
        from transformer.model import build_transformer
        model = build_transformer(
            vocab_size=VOCAB_SIZE,
            max_input_len=MAX_INPUT_LEN,
            max_output_len=MAX_OUTPUT_LEN,
        )
        # Warm-up build
        dummy_enc = np.zeros((1, MAX_INPUT_LEN),  dtype=np.int32)
        dummy_dec = np.zeros((1, MAX_OUTPUT_LEN), dtype=np.int32)
        model([dummy_enc, dummy_dec])
        model.load_weights(TRANSFORMER_WEIGHTS)
    else:  # seq2seq
        from seq2seq.model import build_seq2seq
        model = build_seq2seq(vocab_size=VOCAB_SIZE)
        dummy_enc = np.zeros((1, MAX_INPUT_LEN),  dtype=np.int32)
        dummy_dec = np.zeros((1, MAX_OUTPUT_LEN), dtype=np.int32)
        model([dummy_enc, dummy_dec])
        model.load_weights(SEQ2SEQ_WEIGHTS)
    return model


def _generate_seq(model, dow: str, month_name: str,
                  leap_str: str, decade: int) -> str:
    """Autoregressive decoding for Seq2Seq / Transformer."""
    enc_tokens = np.array([[
        TOKEN_TO_ID[dow],
        TOKEN_TO_ID[month_name],
        TOKEN_TO_ID[leap_str],
        TOKEN_TO_ID[str(decade)],
    ]], dtype=np.int32)

    dec_tokens = [START_ID]

    for _ in range(MAX_OUTPUT_LEN):
        dec_array = np.array([
            dec_tokens + [PAD_ID] * (MAX_OUTPUT_LEN - len(dec_tokens))
        ], dtype=np.int32)

        preds = model([enc_array := enc_tokens, dec_array], training=False)
        next_id = int(np.argmax(preds[0, len(dec_tokens) - 1, :]))

        if next_id == END_ID or next_id == PAD_ID:
            break
        dec_tokens.append(next_id)

    result = "".join(ID_TO_TOKEN.get(t, "") for t in dec_tokens[1:])
    return result


def _generate_cvae(decoder, dow: str, month_name: str,
                   leap_str: str, decade: int) -> str:
    cond = build_condition_vector(dow, month_name, leap_str, decade)
    cond = cond.reshape(1, -1)
    z    = np.random.normal(size=(1, LATENT_DIM_CVAE)).astype(np.float32)
    pred = decoder([z, cond], training=False).numpy()[0]
    day, month, year = denormalize_date(pred)
    return format_date(day, month, year)


def _generate_cgan(generator, dow: str, month_name: str,
                   leap_str: str, decade: int) -> str:
    from tokenizer import MAX_DAYS_IN_DECADE
    cond   = build_condition_vector(dow, month_name, leap_str, decade)
    cond   = cond.reshape(1, -1)
    noise  = np.random.normal(size=(1, NOISE_DIM_CGAN)).astype(np.float32)
    offset_norm = float(generator([noise, cond], training=False).numpy()[0, 0])
    offset = int(round(offset_norm * MAX_DAYS_IN_DECADE))
    day, month, year = offset_to_date(offset, decade)
    return format_date(day, month, year)


# ---------------------------------------------------------------------------
# Weekday correction (post-processing)
# ---------------------------------------------------------------------------

def _fix_weekday(date_str: str, expected_dow: str) -> str:
    """
    Nudge the date ±7 days until the weekday matches, staying within the decade.
    Falls back to the original if nothing works within 6 steps.
    """
    from datetime import datetime, timedelta

    try:
        day, month, year = map(int, date_str.split("-"))
        if not is_valid_date(day, month, year):
            return date_str

        dt        = datetime(year, month, day)
        decade    = year // 10
        dec_start = decade * 10
        dec_end   = dec_start + 9

        for delta in range(-6, 7):
            candidate = dt + timedelta(days=delta)
            if candidate.year < dec_start or candidate.year > dec_end:
                continue
            if candidate.strftime("%a").upper()[:3] == expected_dow:
                return format_date(candidate.day, candidate.month, candidate.year)
    except Exception:
        pass

    return date_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(input_path: str, output_path: str):
    # Load model
    if MODEL_CHOICE in ("transformer", "seq2seq"):
        model = _build_seq_model(MODEL_CHOICE)
        gen_fn = lambda d, m, l, dec: _generate_seq(model, d, m, l, dec)
    elif MODEL_CHOICE == "cvae":
        from cvae.model import build_decoder
        decoder = build_decoder()
        dummy_z   = np.zeros((1, LATENT_DIM_CVAE), dtype=np.float32)
        dummy_c   = np.zeros((1, CONDITION_DIM),   dtype=np.float32)
        decoder([dummy_z, dummy_c])
        decoder.load_weights(CVAE_WEIGHTS)
        gen_fn = lambda d, m, l, dec: _generate_cvae(decoder, d, m, l, dec)
    else:  # cgan
        from cgan.model import build_generator
        generator = build_generator(NOISE_DIM_CGAN, CONDITION_DIM)
        dummy_n = np.zeros((1, NOISE_DIM_CGAN), dtype=np.float32)
        dummy_c = np.zeros((1, CONDITION_DIM),  dtype=np.float32)
        generator([dummy_n, dummy_c])
        generator.load_weights(CGAN_WEIGHTS)
        gen_fn = lambda d, m, l, dec: _generate_cgan(generator, d, m, l, dec)

    # Read input
    with open(input_path) as f:
        lines = f.readlines()

    results = []
    for line in lines:
        line = line.strip()
        if not line:
            results.append("")
            continue

        dow, month_name, leap_str, decade, _ = parse_raw_line(line + " 1-1-1800")

        date_str = gen_fn(dow, month_name, leap_str, decade)

        # Post-process: fix weekday
        date_str = _fix_weekday(date_str, dow)

        # Reconstruct full output line: [DOW] [MONTH] [LEAP] [DECADE] date
        results.append(
            f"[{dow}] [{month_name}] [{leap_str}] [{decade}] {date_str}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(results) + "\n")

    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  required=True,  help="Path to input file")
    parser.add_argument("-o", "--output", required=True,  help="Path to output file")
    args = parser.parse_args()
    predict(args.input, args.output)
