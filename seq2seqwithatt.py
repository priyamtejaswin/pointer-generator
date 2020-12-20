#!/usr/bin/env python
"""
created at: Sat 19 Dec 2020 05:22:17 AM EST
created by: Priyam Tejaswin

Implementing the (baseline) Seq2Seq+Attention model.
References:
Section 2 in <https://arxiv.org/pdf/1704.04368.pdf>
"""


import numpy as np
import tensorflow as tf


class BiLSTMEncoder(tf.keras.Model):
    def __init__(self, lstm_embed_size=256):
        super(BiLSTMEncoder, self).__init__()

        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_embed_size, return_sequences=True)
        )


    def call(self, x):
        bilstm_embeds = self.encoder(x)
        return bilstm_embeds

    
class LSTMDecoder(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size = 50000):
        super(LSTMDecoder, self).__init__()

        self.decoder = tf.keras.layers.LSTMCell(lstm_embed_size)

        self.henc_dense = tf.keras.layers.Dense(word_embed_size)
        self.sdec_dense = tf.keras.layers.Dense(word_embed_size)

        glorot_uniform = tf.keras.initializers.glorot_uniform()
        self.vatt = tf.Variable(glorot_uniform(shape=(word_embed_size, 1)))

        self.linear_first = tf.keras.layers.Dense(4096)  # This is arbitrary.
        self.linear_vocab = tf.keras.layers.Dense(vocab_size)


    def call(self, seq_outs, step_input, prev_states):
        """
        seq_outs: Encoder output for all input tokens.
        step_input: Input to the decoder for current timestep t.
        prev_states: [prev_h, prev_carry]
        """
        _, (hidden, carry) = self.decoder(step_input, prev_states)

        # Compute attention.
        scores = tf.tanh(self.henc_dense(seq_outs) + tf.expand_dims(self.sdec_dense(carry), 1))
        logits = tf.linalg.matmul(scores, self.vatt)
        attention = tf.nn.softmax(logits, axis=1)

        # Context vector.
        context = tf.reduce_sum(seq_outs * attention, axis=1)
        combined = tf.concat([context, carry], axis=1)

        return combined, hidden, carry


LSTM_EMBED_SIZE = 5
WORD_EMBED_SIZE = 7
VOCAB_SIZE = 17

encoder = BiLSTMEncoder(lstm_embed_size=LSTM_EMBED_SIZE)
decoder = LSTMDecoder(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE)

x = tf.convert_to_tensor(np.random.rand(32, 9, WORD_EMBED_SIZE))  # [Batch, Time, Features]
enc_embeds = encoder(x)
prev_states = [tf.random.normal([32, LSTM_EMBED_SIZE]) , tf.zeros([32, LSTM_EMBED_SIZE])]
dec_combined, dec_hidden, dec_carry = decoder(enc_embeds, x[:, 0, :], prev_states)

print("passed.")