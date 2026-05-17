"""
Training script for Model 2: CGAN.

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
    get_offset_in_decade, MAX_DAYS_IN_DECADE,
)
from model import build_generator, build_discriminator

# ---------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
CONDITION_DIM    = 21
NOISE_DIM        = 16
BATCH_SIZE       = 128
EPOCHS           = 100
GEN_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "generator.weights.h5")


# ---------------------------------------------------------------------------

def load_dataset(path: str):
    with open(path) as f:
        lines = f.readlines()

    X_conditions = []
    Y_offsets    = []

    for line in lines:
        dow, month_name, leap_str, decade, date_str = parse_raw_line(line)
        condition = build_condition_vector(dow, month_name, leap_str, decade)

        day, month, year = map(int, date_str.split("-"))
        offset      = get_offset_in_decade(day, month, year)
        offset_norm = offset / MAX_DAYS_IN_DECADE

        X_conditions.append(condition)
        Y_offsets.append([offset_norm])

    return (
        np.array(X_conditions, dtype=np.float32),
        np.array(Y_offsets,    dtype=np.float32),
    )


# ---------------------------------------------------------------------------

@tf.function
def train_step(real_conditions, real_offsets,
               generator, discriminator,
               gen_opt, disc_opt, bce, noise_dim):

    batch_size = tf.shape(real_conditions)[0]

    # --- Discriminator step ---
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as disc_tape:
        fake_offsets      = generator([noise, real_conditions], training=True)
        real_preds        = discriminator([real_offsets, real_conditions], training=True)
        fake_preds        = discriminator([fake_offsets, real_conditions], training=True)

        real_loss  = bce(tf.ones_like(real_preds),  real_preds)
        fake_loss  = bce(tf.zeros_like(fake_preds), fake_preds)
        disc_loss  = real_loss + fake_loss

    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # --- Generator step ---
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape:
        gen_offsets = generator([noise, real_conditions], training=True)
        preds       = discriminator([gen_offsets, real_conditions], training=True)
        gen_loss    = bce(tf.ones_like(preds), preds)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))

    return gen_loss, disc_loss


def main(data_path: str):
    print("Loading dataset...")
    X, Y = load_dataset(data_path)
    print(f"  Samples: {len(X)}")

    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)

    generator     = build_generator(NOISE_DIM, CONDITION_DIM)
    discriminator = build_discriminator(CONDITION_DIM)

    bce      = tf.keras.losses.BinaryCrossentropy()
    gen_opt  = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train))
        .shuffle(10000, seed=42)
        .batch(BATCH_SIZE)
    )

    print("Training CGAN...")
    for epoch in range(EPOCHS):
        gen_losses, disc_losses = [], []
        for cond_batch, offset_batch in train_dataset:
            g_loss, d_loss = train_step(
                cond_batch, offset_batch,
                generator, discriminator,
                gen_opt, disc_opt, bce, NOISE_DIM,
            )
            gen_losses.append(float(g_loss))
            disc_losses.append(float(d_loss))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"G Loss: {np.mean(gen_losses):.4f} | "
                f"D Loss: {np.mean(disc_losses):.4f}"
            )

    os.makedirs(os.path.dirname(GEN_WEIGHTS_PATH), exist_ok=True)
    generator.save_weights(GEN_WEIGHTS_PATH)
    print(f"Generator weights saved to {GEN_WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../../data/data.txt")
    args = parser.parse_args()
    main(args.data)
