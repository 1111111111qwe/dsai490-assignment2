"""
Model 2: Conditional GAN (CGAN)
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_generator(noise_dim: int = 16,
                    condition_dim: int = 21) -> tf.keras.Model:
    """
    Takes (noise, condition) → normalized date offset in [0, 1].
    The offset represents days elapsed from the start of the target decade.
    """
    noise_input     = layers.Input(shape=(noise_dim,),     name="gen_noise")
    condition_input = layers.Input(shape=(condition_dim,), name="gen_condition")

    x = layers.Concatenate()([noise_input, condition_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64,  activation="relu")(x)
    x = layers.Dense(32,  activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="offset_output")(x)

    return tf.keras.Model(
        [noise_input, condition_input],
        output,
        name="generator",
    )


def build_discriminator(condition_dim: int = 21) -> tf.keras.Model:
    """
    Takes (offset, condition) → P(real).
    """
    offset_input    = layers.Input(shape=(1,),             name="disc_offset")
    condition_input = layers.Input(shape=(condition_dim,), name="disc_condition")

    x = layers.Concatenate()([offset_input, condition_input])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="real_fake")(x)

    return tf.keras.Model(
        [offset_input, condition_input],
        output,
        name="discriminator",
    )
