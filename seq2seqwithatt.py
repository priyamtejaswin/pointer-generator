#!/usr/bin/env python
"""
created at: Sat 19 Dec 2020 05:22:17 AM EST
created by: Priyam Tejaswin

Implementing the (baseline) Seq2Seq+Attention model.
References:
Section 2 in <https://arxiv.org/pdf/1704.04368.pdf>
"""


import os
import io
import re
from tqdm import tqdm
import unicodedata
import numpy as np
import tensorflow as tf


class BiLSTMEncoder(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000):
        super(BiLSTMEncoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, word_embed_size)

        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_embed_size, return_sequences=True)
        )


    def call(self, x):
        embeds = self.embedding(x)
        outs = self.encoder(embeds)
        return outs

    
class LSTMDecoder(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000):
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


def main():
    LSTM_EMBED_SIZE = 5
    WORD_EMBED_SIZE = 7
    VOCAB_SIZE = 17

    encoder = BiLSTMEncoder(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE) 
    decoder = LSTMDecoder(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE)

    x = tf.convert_to_tensor(np.random.randint(0, VOCAB_SIZE, [32, 9]))  # [Batch, TimeIndices]
    print(x.shape)
    print(x)
    enc_embeds = encoder(x)
    prev_states = [tf.random.normal([32, LSTM_EMBED_SIZE]) , tf.zeros([32, LSTM_EMBED_SIZE])]
    dec_combined, dec_hidden, dec_carry = decoder(enc_embeds, encoder.embedding(x[:, 0]), prev_states)

    print("passed.")


def unicode_to_ascii(s):
    # Normalize and remove accents.
    # https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    return ''.join(c for c in unicodedata.normalize('NFD', s) \
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = w.replace('<unk>', 'UNK')

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^0-9#a-zA-Z?.!,]+", " ", w)

    w = w.strip()
    w = w.replace('UNK', '<unk>')

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    words = [preprocess_sentence(l) for l in  tqdm(lines[:num_examples])]
    return words


def datastuff():
    maindir = '/projects/metis1/users/ptejaswi/multistep-retrieve-summarize/data/gigawords/org_data'
    path_train_src = os.path.join(maindir, 'train.src.txt')
    path_train_tgt = os.path.join(maindir, 'train.tgt.txt')

    source = create_dataset(path_train_src, 10000)
    target = create_dataset(path_train_tgt, 10000)

    print("source:", len(source))
    print("target:", len(target))

    assert len(source) == len(target)

    top_k = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>', filters=' ')
    
    tokenizer.fit_on_texts(source + target)
    # Set <pad> AFTER fitting on texts!
    tokenizer.index_word[0] = '<pad>'
    tokenizer.word_index['<pad>'] = 0

    source_seqs = tokenizer.texts_to_sequences(source)
    source_vecs = tf.keras.preprocessing.sequence.pad_sequences(source_seqs, padding='post')

    target_seqs = tokenizer.texts_to_sequences(target)
    target_vecs = tf.keras.preprocessing.sequence.pad_sequences(target_seqs, padding='post')

    print("passed")


if __name__ == '__main__':
    # main()
    datastuff()