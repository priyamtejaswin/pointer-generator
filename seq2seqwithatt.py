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
import random
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
        return self.encoder(embeds)  # Output

    
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

        self.lstm_embed_size = lstm_embed_size


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

        # Distribution over vocab.
        pvocab = self.linear_vocab(self.linear_first(combined))

        return pvocab, hidden, carry

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.lstm_embed_size))


class S2SModel(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000, batch_size=32):
        super(S2SModel, self).__init__()

        self.encoder = BiLSTMEncoder(lstm_embed_size=lstm_embed_size, word_embed_size=word_embed_size, vocab_size=vocab_size)
        self.decoder = LSTMDecoder(lstm_embed_size=lstm_embed_size, word_embed_size=word_embed_size, vocab_size=vocab_size)
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    @tf.function
    def train_step(self, inp, targ, tokenizer):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output = self.encoder(inp)
            dec_hidden = [self.decoder.reset_state(self.batch_size)] * 2

            starts = tf.expand_dims([tokenizer.word_index['<start>']] * self.batch_size, 1)
            dec_input = tf.squeeze(self.encoder.embedding(starts))

            for t in range(1, targ.shape[1]):
                dec_preds, dec_output, dec_carry = self.decoder(enc_output, dec_input, dec_hidden)
                loss += self.loss_function(targ[:, t], dec_preds)

                dec_input = tf.squeeze(self.encoder.embedding(tf.expand_dims(targ[:, t], 1)))
                dec_hidden = [dec_output, dec_carry]
            
        batch_loss = loss * 1.0 / int(targ.shape[1])
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


def main():
    LSTM_EMBED_SIZE = 256
    WORD_EMBED_SIZE = 128
    VOCAB_SIZE = 50000
    BATCH_SIZE = 32
    EPOCHS = 10

    dataset, tokenizer, steps_per_epoch = datastuff(top_k=VOCAB_SIZE, num_examples=1000000, batch_size=BATCH_SIZE)
    model = S2SModel(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        total_loss = 0

        progbar = tqdm(enumerate(dataset.take(steps_per_epoch)), total=steps_per_epoch, desc='description')

        for batch, (inp, targ) in progbar:
            batch_loss = model.train_step(inp, targ, tokenizer)
            total_loss += batch_loss
            progbar.set_description("batch loss: %.3f" % batch_loss.numpy())

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


def datastuff(top_k, num_examples=None, batch_size=None):
    maindir = '/projects/metis1/users/ptejaswi/multistep-retrieve-summarize/data/gigawords/org_data'
    path_train_src = os.path.join(maindir, 'train.src.txt')
    path_train_tgt = os.path.join(maindir, 'train.tgt.txt')

    source = create_dataset(path_train_src, num_examples)
    target = create_dataset(path_train_tgt, num_examples)

    print("source:", len(source))
    print("target:", len(target))

    assert len(source) == len(target)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>', filters=' ')
    
    tokenizer.fit_on_texts(source + target)
    # Set <pad> AFTER fitting on texts!
    tokenizer.index_word[0] = '<pad>'
    tokenizer.word_index['<pad>'] = 0

    source_seqs = tokenizer.texts_to_sequences(source)
    source_vecs = tf.keras.preprocessing.sequence.pad_sequences(source_seqs, padding='post')

    target_seqs = tokenizer.texts_to_sequences(target)
    target_vecs = tf.keras.preprocessing.sequence.pad_sequences(target_seqs, padding='post')

    indices = list(range(len(source)))
    random.shuffle(indices)
    slice_index = int(len(indices) * 0.8)

    X_train, X_test = source_vecs[:slice_index], source_vecs[slice_index:]
    y_train, y_test = target_vecs[:slice_index], target_vecs[slice_index:]

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=5)

    return dataset, tokenizer, len(X_train)//batch_size


if __name__ == '__main__':
    main()