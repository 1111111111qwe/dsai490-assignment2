"""
Model 3: Seq2Seq with LSTM (Encoder-Decoder)
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_seq2seq(vocab_size: int,
                  embedding_dim: int = 64,
                  latent_dim: int = 128,
                  max_input_len: int = 4,
                  max_output_len: int = 13) -> tf.keras.Model:
    """
    Returns the full teacher-forcing training model.
    Encoder LSTM reads the 4-token condition sequence.
    Decoder LSTM generates the date character-by-character.
    """
    # Encoder
    encoder_inputs   = layers.Input(shape=(max_input_len,),  name="enc_input")
    enc_emb          = layers.Embedding(vocab_size, embedding_dim, name="enc_embedding")(encoder_inputs)
    _, state_h, state_c = layers.LSTM(latent_dim, return_state=True, name="enc_lstm")(enc_emb)
    encoder_states   = [state_h, state_c]

    # Decoder
    decoder_inputs   = layers.Input(shape=(max_output_len,), name="dec_input")
    dec_emb_layer    = layers.Embedding(vocab_size, embedding_dim, name="dec_embedding")
    dec_emb          = dec_emb_layer(decoder_inputs)

    dec_lstm         = layers.LSTM(latent_dim, return_sequences=True,
                                   return_state=True, name="dec_lstm")
    dec_outputs, _, _= dec_lstm(dec_emb, initial_state=encoder_states)

    dec_dense        = layers.Dense(vocab_size, activation="softmax", name="output_dense")
    decoder_outputs  = dec_dense(dec_outputs)

    model = tf.keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="seq2seq",
    )
    return model
