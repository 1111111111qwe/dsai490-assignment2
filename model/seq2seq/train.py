"""
Training script for Model 3: Seq2Seq LSTM.

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
    parse_raw_line, build_seq_tokens,
    VOCAB_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN,
)
from model import build_seq2seq

# ---------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
LATENT_DIM    = 128
BATCH_SIZE    = 128
EPOCHS        = 30
WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "weights", "seq2seq.weights.h5")


def load_dataset(path: str):
    with open(path) as f:
        lines = f.readlines()

    enc_inputs, dec_inputs, dec_targets = [], [], []

    for line in lines:
        dow, month_name, leap_str, decade, date_str = parse_raw_line(line)
        enc, dec_in, dec_tar = build_seq_tokens(
            dow, month_name, leap_str, decade, date_str
        )
        enc_inputs.append(enc)
        dec_inputs.append(dec_in)
        dec_targets.append(dec_tar)

    return (
        np.array(enc_inputs,  dtype=np.int32),
        np.array(dec_inputs,  dtype=np.int32),
        np.array(dec_targets, dtype=np.int32),
    )


def main(data_path: str):
    print("Loading dataset...")
    enc_in, dec_in, dec_tar = load_dataset(data_path)
    print(f"  Samples: {len(enc_in)}")

    (
        X_enc_train, X_enc_test,
        X_dec_train, X_dec_test,
        y_train,     y_test,
    ) = train_test_split(enc_in, dec_in, dec_tar, test_size=0.2, random_state=42)

    model = build_seq2seq(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        latent_dim=LATENT_DIM,
        max_input_len=MAX_INPUT_LEN,
        max_output_len=MAX_OUTPUT_LEN,
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("Training Seq2Seq...")
    model.fit(
        [X_enc_train, X_dec_train], y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_enc_test, X_dec_test], y_test),
    )

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    model.save_weights(WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../../data/data.txt")
    args = parser.parse_args()
    main(args.data)
