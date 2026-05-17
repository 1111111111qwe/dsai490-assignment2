"""
Training script for Model 1: CVAE.

Usage:
    python train.py --data ../../data/data.txt
"""

import argparse
import sys
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer import (
    parse_raw_line, build_condition_vector,
    MONTH_MAP, DAY_MAP,
)
from model import build_encoder, build_decoder, CVAE

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONDITION_DIM = 21
DATE_DIM      = 3
LATENT_DIM    = 8
BATCH_SIZE    = 128
EPOCHS        = 50
BETA          = 0.001
WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "weights", "cvae_decoder.weights.h5")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    X_conditions = []
    Y_dates      = []

    for line in lines:
        dow, month_name, leap_str, decade, date_str = parse_raw_line(line)
        condition = build_condition_vector(dow, month_name, leap_str, decade)

        day, month, year = map(int, date_str.split("-"))
        day_norm   = day   / 31.0
        month_norm = month / 12.0
        year_norm  = (year - 1800) / 400.0

        X_conditions.append(condition)
        Y_dates.append([day_norm, month_norm, year_norm])

    return (
        np.array(X_conditions, dtype=np.float32),
        np.array(Y_dates,      dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str):
    print("Loading dataset...")
    X, Y = load_dataset(data_path)
    print(f"  Samples: {len(X)}, Condition dim: {X.shape[1]}, Target dim: {Y.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    encoder = build_encoder(CONDITION_DIM, DATE_DIM, LATENT_DIM)
    decoder = build_decoder(CONDITION_DIM, DATE_DIM, LATENT_DIM)
    cvae    = CVAE(encoder, decoder, beta=BETA)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    print("Training CVAE...")
    history = cvae.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
    )

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    decoder.save_weights(WEIGHTS_PATH)
    print(f"Decoder weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../../data/data.txt")
    args = parser.parse_args()
    main(args.data)
