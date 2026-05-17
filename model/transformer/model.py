"""
Model 4: Transformer (Encoder-Decoder with multi-head attention)
"""

import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Positional Embedding
# ---------------------------------------------------------------------------

class PositionalEmbedding(layers.Layer):
    """Combines token embeddings with learned positional embeddings."""

    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_embeddings    = layers.Embedding(vocab_size,       embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length,  embed_dim)
        self.sequence_length     = sequence_length

    def call(self, inputs):
        length    = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return (
            self.token_embeddings(inputs)
            + self.position_embeddings(positions)
        )


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerEncoder(layers.Layer):

    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int):
        super().__init__()
        self.attention   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj  = tf.keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_out  = self.attention(inputs, inputs)
        x         = self.layernorm_1(inputs + attn_out)
        proj_out  = self.dense_proj(x)
        return self.layernorm_2(x + proj_out)


# ---------------------------------------------------------------------------
# Transformer Decoder Block
# ---------------------------------------------------------------------------

class TransformerDecoder(layers.Layer):

    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int):
        super().__init__()
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj  = tf.keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def _causal_mask(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len    = tf.shape(inputs)[1]
        i    = tf.range(seq_len)[:, tf.newaxis]
        j    = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        return tf.tile(mask, [batch_size, 1, 1])

    def call(self, inputs, encoder_outputs):
        causal_mask = self._causal_mask(inputs)

        attn1_out = self.attention_1(
            query=inputs, value=inputs, key=inputs,
            attention_mask=causal_mask,
        )
        out1 = self.layernorm_1(inputs + attn1_out)

        attn2_out = self.attention_2(
            query=out1, value=encoder_outputs, key=encoder_outputs,
        )
        out2 = self.layernorm_2(out1 + attn2_out)

        proj_out = self.dense_proj(out2)
        return self.layernorm_3(out2 + proj_out)


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

def build_transformer(
    vocab_size: int,
    max_input_len: int  = 4,
    max_output_len: int = 13,
    embed_dim: int      = 128,
    dense_dim: int      = 256,
    num_heads: int      = 4,
) -> tf.keras.Model:

    enc_input = layers.Input(shape=(max_input_len,),  dtype="int32", name="enc_input")
    dec_input = layers.Input(shape=(max_output_len,), dtype="int32", name="dec_input")

    enc_emb  = PositionalEmbedding(max_input_len,  vocab_size, embed_dim)(enc_input)
    dec_emb  = PositionalEmbedding(max_output_len, vocab_size, embed_dim)(dec_input)

    enc_out  = TransformerEncoder(embed_dim, dense_dim, num_heads)(enc_emb)
    dec_out  = TransformerDecoder(embed_dim, dense_dim, num_heads)(dec_emb, enc_out)

    output   = layers.Dense(vocab_size, activation="softmax", name="output_dense")(dec_out)

    return tf.keras.Model([enc_input, dec_input], output, name="transformer")
