"""
Model 1: Conditional Variational Autoencoder (CVAE)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------

class Sampling(layers.Layer):
    """Samples z ~ N(z_mean, exp(0.5 * z_log_var))."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(condition_dim: int = 21,
                  date_dim: int = 3,
                  latent_dim: int = 8) -> tf.keras.Model:

    condition_input = layers.Input(shape=(condition_dim,), name="enc_condition")
    date_input      = layers.Input(shape=(date_dim,),      name="enc_date")

    x = layers.Concatenate()([condition_input, date_input])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling()([z_mean, z_log_var])

    return tf.keras.Model(
        [condition_input, date_input],
        [z_mean, z_log_var, z],
        name="encoder",
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def build_decoder(condition_dim: int = 21,
                  date_dim: int = 3,
                  latent_dim: int = 8) -> tf.keras.Model:

    latent_input    = layers.Input(shape=(latent_dim,),    name="dec_z")
    condition_input = layers.Input(shape=(condition_dim,), name="dec_condition")

    x = layers.Concatenate()([latent_input, condition_input])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    output = layers.Dense(date_dim, activation="sigmoid", name="date_output")(x)

    return tf.keras.Model(
        [latent_input, condition_input],
        output,
        name="decoder",
    )


# ---------------------------------------------------------------------------
# CVAE wrapper
# ---------------------------------------------------------------------------

class CVAE(tf.keras.Model):
    """Conditional VAE with custom train/test steps using tf.GradientTape."""

    def __init__(self, encoder: tf.keras.Model,
                 decoder: tf.keras.Model,
                 beta: float = 0.001):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta    = beta

    def call(self, inputs):
        condition, target = inputs
        _, _, z = self.encoder([condition, target])
        return self.decoder([z, condition])

    def train_step(self, data):
        condition, target = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([condition, target])
            reconstruction       = self.decoder([z, condition])

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(target - reconstruction), axis=1)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss":               total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss":             kl_loss,
        }

    def test_step(self, data):
        condition, target = data
        z_mean, z_log_var, z = self.encoder([condition, target])
        reconstruction       = self.decoder([z, condition])

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(target - reconstruction), axis=1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss":               total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss":             kl_loss,
        }
