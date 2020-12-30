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
import datetime
from tqdm import tqdm
import unicodedata
import numpy as np
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)


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


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    
class LSTMDecoder(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000):
        super(LSTMDecoder, self).__init__()

        self.decoder = tf.keras.layers.LSTM(lstm_embed_size, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(512)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.lstm_embed_size = lstm_embed_size


    def call(self, enc_output, step_input, hidden, initial_state):
        """
        seq_outs: Encoder output for all input tokens.
        step_input: Input to the decoder for current timestep t.
        initial_state: [prev_mem, prev_carry]
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = tf.concat([tf.expand_dims(context_vector, 1), step_input], axis=-1)
        dec_output, final_memory, final_carry = self.decoder(x, initial_state=initial_state)

        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
        vocab = self.fc(dec_output)

        return vocab, final_memory, final_carry

        # # Compute attention.
        # scores = tf.tanh(self.henc_dense(seq_outs) + tf.expand_dims(self.sdec_dense(carry), 1))
        # logits = tf.linalg.matmul(scores, self.vatt)
        # attention = tf.nn.softmax(logits, axis=1)

        # # Context vector.
        # context = tf.reduce_sum(seq_outs * attention, axis=1)
        # combined = tf.concat([context, carry], axis=1)

        # # Distribution over vocab.
        # pvocab = tf.nn.softmax(self.linear_vocab(self.linear_first(combined)), axis=1)

        # return pvocab, hidden, carry

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
    def train_step(self, inp, targ, tokenizer, update=True):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output = self.encoder(inp)
            dec_hidden = self.decoder.reset_state(self.batch_size)

            starts = tf.expand_dims([tokenizer.word_index['<start>']] * self.batch_size, 1)
            dec_input = self.encoder.embedding(starts)
            initial_state = [dec_hidden, dec_hidden]

            for t in range(1, targ.shape[1]):
                dec_preds, dec_hidden, dec_carry = self.decoder(enc_output, dec_input, dec_hidden, initial_state)
                loss += self.loss_function(targ[:, t], dec_preds)

                # For next timestep!
                dec_input = self.encoder.embedding(tf.expand_dims(targ[:, t], 1))
                initial_state = [dec_hidden, dec_carry]
            
        batch_loss = loss * 1.0 / int(targ.shape[1])
        if update:
            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    
    def evaluate(self, inp, tokenizer, max_len):
        """
        `max_len` is max test sequence length during training.
        """
        assert len(inp.shape) == 2, "Should be of size [n, length]"
        enc_output = self.encoder(inp)
        dec_hidden = self.decoder.reset_state(len(inp))

        starts = tf.expand_dims([tokenizer.word_index['<start>']] * len(inp), 1)
        dec_input = self.encoder.embedding(starts)
        initial_state = [dec_hidden, dec_hidden]

        result = ''
        
        for t in range(max_len-2):
            dec_preds, dec_hidden, dec_carry = self.decoder(enc_output, dec_input, dec_hidden, initial_state)
            pred_id = tf.argmax(dec_preds[0]).numpy()
            word = tokenizer.index_word[pred_id]
            if word == '<end>':
                return result.strip()
            else:
                result += word + ' '

            # For next timestep!
            dec_input = self.encoder.embedding(tf.expand_dims([pred_id], 0))
            initial_state = [dec_hidden, dec_carry]

        return result


def main():
    LSTM_EMBED_SIZE = 256
    WORD_EMBED_SIZE = 128
    VOCAB_SIZE = 50000
    BATCH_SIZE = 32
    EPOCHS = 5

    dataset, tokenizer, steps_per_epoch, max_targ_len, (X_test, y_test) = datastuff(top_k=VOCAB_SIZE, num_examples=1000000, batch_size=BATCH_SIZE)
    model = S2SModel(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)

    tboard_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_train_log_dir = 'logs/' + tboard_time + '/train'
    tboard_train_writer = tf.summary.create_file_writer(tboard_train_log_dir)

    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        total_loss = 0

        progbar = tqdm(enumerate(dataset.take(steps_per_epoch)), total=steps_per_epoch, desc='avg loss: ')

        for batch, (inp, targ) in progbar:
            batch_loss = model.train_step(inp, targ, tokenizer)
            total_loss += batch_loss.numpy()
            avg_loss = total_loss/(batch+1)
            progbar.set_description("avg loss: %.3f" % avg_loss)

            with tboard_train_writer.as_default():
                tf.summary.scalar('train-loss', avg_loss, step=(epoch * steps_per_epoch + batch))

            if (batch+1)%1000 == 0:
                print(tokenizer.sequences_to_texts(X_test[0:1]))
                eval_ans = model.evaluate(X_test[0:1], tokenizer, max_targ_len)
                print(tokenizer.sequences_to_texts(y_test[0:1]))
                print()
                print(eval_ans)
                checkpoint.save(file_prefix = checkpoint_prefix + 'e%dstep%d'%(epoch, batch))

                with tboard_train_writer.as_default():
                    test_loss = 0
                    norm = 0
                    for i in range(0, min(2000, len(X_test)), BATCH_SIZE):
                        x = X_test[i : i+BATCH_SIZE]
                        y = y_test[i : i+BATCH_SIZE]
                        test_loss += model.train_step(x, y, tokenizer, update=False).numpy()
                        norm += 1

                    tf.summary.scalar('test-loss', test_loss*1.0/norm, step=(epoch * steps_per_epoch + batch))

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
    maindir = '/projects/ogma2/users/ptejaswi/org_data'
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
    dataset = dataset.prefetch(buffer_size=10)

    return dataset, tokenizer, len(X_train)//batch_size, target_vecs.shape[1], (X_test, y_test)


if __name__ == '__main__':
    main()
