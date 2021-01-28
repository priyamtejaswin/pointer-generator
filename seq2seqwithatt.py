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
import sys
import pickle
import random
import datetime
from tqdm import tqdm
import unicodedata
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tokenizers import Tokenizer
from evaluate_checkpoint import generate_output


physical_devices = tf.config.list_physical_devices('GPU')
for phydev in physical_devices:
    tf.config.experimental.set_memory_growth(phydev, enable=True)


class BiLSTMEncoder(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000, pretr_src_embeds=None):
        super(BiLSTMEncoder, self).__init__()

        if pretr_src_embeds is None:
            initializer = 'uniform'
        else:
            initializer = tf.keras.initializers.Constant(pretr_src_embeds)
            print("Setting pre-trained embeddings for Encoder.")

        self.embedding = tf.keras.layers.Embedding(vocab_size, word_embed_size, embeddings_initializer=initializer)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_embed_size, return_sequences=True, return_state=True)
        )
        self.hidden_map = tf.keras.layers.Dense(lstm_embed_size)
        self.carry_map = tf.keras.layers.Dense(lstm_embed_size)


    def call(self, x):
        embeds = self.embedding(x)
        # return self.encoder(embeds)  # Output
        sequences, fwhid, fwcar, bwhid, bwcar = self.encoder(embeds)
        cathid = tf.concat([fwhid, bwhid], axis=-1)
        catcar = tf.concat([fwcar, bwcar], axis=-1)

        return sequences, self.hidden_map(cathid), self.carry_map(catcar)


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
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000, pretr_tgt_embeds=None):
        super(LSTMDecoder, self).__init__()

        if pretr_tgt_embeds is None:
            initializer = 'uniform'
        else:
            initializer = tf.keras.initializers.Constant(pretr_tgt_embeds)
            print("Setting pre-trained embeddings for Decoder.")

        self.embedding = tf.keras.layers.Embedding(vocab_size, word_embed_size, embeddings_initializer=initializer)
        self.decoder = tf.keras.layers.LSTM(lstm_embed_size, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(512)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.lstm_embed_size = lstm_embed_size


    def call(self, enc_output, step_token, hidden, initial_state):
        """
        seq_outs: Encoder output for all input tokens.
        step_input: Input to the decoder for current timestep t.
        initial_state: [prev_mem, prev_carry]
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)

        step_input = self.embedding(step_token)
        x = tf.concat([tf.expand_dims(context_vector, 1), step_input], axis=-1)
        dec_output, final_memory, final_carry = self.decoder(x, initial_state=initial_state)

        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
        vocab = self.fc(dec_output)

        return vocab, final_memory, final_carry


    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.lstm_embed_size))


class CellDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
        super(CellDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        #Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                                None, self.batch_sz*[35], self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)


    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                    self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs 
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if(attention_type=='bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[35-1])
        return outputs


class S2SModel(tf.keras.Model):
    def __init__(self, lstm_embed_size=256, word_embed_size=128, vocab_size=50000, batch_size=32, pretr_src_embeds=None, pretr_tgt_embeds=None):
        super(S2SModel, self).__init__()

        self.encoder = BiLSTMEncoder(lstm_embed_size=lstm_embed_size, word_embed_size=word_embed_size, vocab_size=vocab_size)
        # self.decoder = LSTMDecoder(lstm_embed_size=lstm_embed_size, word_embed_size=word_embed_size, vocab_size=vocab_size, pretr_tgt_embeds=pretr_tgt_embeds)
        self.decoder = CellDecoder(vocab_size=vocab_size, embedding_dim=word_embed_size, dec_units=lstm_embed_size, batch_sz=batch_size)
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
    def train_step(self, inp, targ, tgt_tokenizer, update=True):
        loss = 0

        # with tf.GradientTape() as tape:
        #     enc_output, dec_hidden, dec_carry = self.encoder(inp)
        #     initial_state = [dec_hidden, dec_carry]
        #     dec_input = tf.expand_dims([tgt_tokenizer.word_index['<start>']] * self.batch_size, 1)

        #     for t in range(1, targ.shape[1]):
        #         dec_preds, dec_hidden, dec_carry = self.decoder(enc_output, dec_input, dec_hidden, initial_state)
        #         loss += self.loss_function(targ[:, t], dec_preds)

        #         # For next timestep!
        #         dec_input = tf.expand_dims(targ[:, t], 1)
        #         initial_state = [dec_hidden, dec_carry]
            
        # batch_loss = loss * 1.0 / int(targ.shape[1])

        with tf.GradientTape() as tape:
            enc_output, dec_hidden, dec_carry = self.encoder(inp)
            dec_input = targ[:, :-1]
            real = targ[:, 1:]

            self.decoder.attention_mechanism.setup_memory(enc_output)
            decoder_initial_state = self.decoder.build_initial_state(self.batch_size, [dec_hidden, dec_carry], tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)

        if update:
            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    
    def evaluate(self, inp, tgt_tokenizer, max_len):
        """
        `max_len` is max test sequence length during training.
        """
        assert len(inp.shape) == 2, "Should be of size [n, length]"
        enc_output, dec_hidden, dec_carry = self.encoder(inp)
        initial_state = [dec_hidden, dec_carry]
        dec_input = tf.expand_dims([tgt_tokenizer.word_index['<start>']] * len(inp), 1)

        result = ''
        
        for t in range(max_len-2):
            dec_preds, dec_hidden, dec_carry = self.decoder(enc_output, dec_input, dec_hidden, initial_state)
            pred_id = tf.argmax(dec_preds[0]).numpy()
            word = tgt_tokenizer.index_word[pred_id]
            if word == '<end>':
                return result.strip()
            else:
                result += word + ' '

            # For next timestep!
            dec_input = tf.expand_dims([pred_id], 0)
            initial_state = [dec_hidden, dec_carry]

        return result


def create_glove_matrix(tokenizer, path_to_vectors, vocab_size, word_embed_size):
    lookup = {}
    with open(path_to_vectors) as fp:
        for line in fp.readlines():
            row = line.strip().split()
            word = row[0]
            vec = np.array([float(x) for x in row[1:]])
            lookup[word] = vec
            assert word == word.lower()

        assert word_embed_size == len(vec), "Pre-trained embed size and desired embed sizes mismatch."

    print("Found %d pre-trained words." % len(lookup))
    print("Found %d tokens in vocab." % len(tokenizer.word_index))
    ru_init = tf.random_uniform_initializer()
    matrix = ru_init(shape=[vocab_size, word_embed_size]).numpy()

    replaced = 0
    for w, ix in tokenizer.word_index.items():
        if w in lookup and ix < vocab_size:
            matrix[ix] = lookup[w]
            replaced += 1

    print("Replaced %d words in matrxi." % replaced)
    return matrix


def main():
    LSTM_EMBED_SIZE = 256
    WORD_EMBED_SIZE = 100
    VOCAB_SIZE = 50000
    BATCH_SIZE = 32
    EPOCHS = 5
    MAX_TARG_LEN = 35

    # Training data and tokenizers.    
    dataset, src_tokenizer, tgt_tokenizer, steps_per_epoch = wikibiodata(top_k=VOCAB_SIZE, num_examples=10000, batch_size=BATCH_SIZE)

    # Validation data.
    valid_dir = '/projects/metis1/users/ptejaswi/wikipedia-biography-dataset/wikipedia-biography-dataset/valid'
    valid_source = create_wikibio_source(load_text(os.path.join(valid_dir, 'valid.box')))
    valid_X = create_sequences(valid_source, src_tokenizer)
    valid_nbs = [int(x) for x in load_text(os.path.join(valid_dir, 'valid.nb'))]
    valid_target = create_wikibio_target(load_text(os.path.join(valid_dir, 'valid.sent')), valid_nbs)
    valid_y = create_sequences(valid_target, tgt_tokenizer)

    # Load and create Glove.
    embed_src = create_glove_matrix(src_tokenizer, '../glove/glove.6B.100d.txt', VOCAB_SIZE, WORD_EMBED_SIZE)
    embed_tgt = create_glove_matrix(tgt_tokenizer, '../glove/glove.6B.100d.txt', VOCAB_SIZE, WORD_EMBED_SIZE)

    # Create model ...
    model = S2SModel(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE,
                        pretr_src_embeds=embed_src, pretr_tgt_embeds=embed_tgt)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    tboard_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_train_log_dir = 'logs/' + tboard_time + '/train'
    tboard_train_writer = tf.summary.create_file_writer(tboard_train_log_dir)

    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        progbar = tqdm(enumerate(dataset.take(steps_per_epoch)), total=steps_per_epoch, desc='epoch: , avg loss: ')

        for batch, (inp, targ) in progbar:
            batch_loss = model.train_step(inp, targ, tgt_tokenizer)
            # total_loss += batch_loss.numpy()
            # avg_loss = total_loss/(batch+1)
            progbar.set_description("epoch: %d, avg loss: %.3f" % (epoch, batch_loss.numpy()))

            if (batch+1)%10 == 0:
                with tboard_train_writer.as_default():
                    tf.summary.scalar('train-loss', batch_loss.numpy(), step=(epoch * steps_per_epoch + batch))

            if (batch+1)%100 == 0:
                print(src_tokenizer.sequences_to_texts(valid_X[0:3]))
                # eval_ans = model.evaluate(valid_X[0:1], tgt_tokenizer, MAX_TARG_LEN)
                eval_ans = generate_output(valid_X[0:3], model.encoder, model.decoder, LSTM_EMBED_SIZE, 
                                            tgt_tokenizer.word_index['<start>'], tgt_tokenizer.word_index['<end>'])
                print(tgt_tokenizer.sequences_to_texts(valid_y[0:3]))
                print()
                print(tgt_tokenizer.sequences_to_texts(eval_ans))

                # checkpoint.save(file_prefix = checkpoint_prefix + 'e%dstep%d'%(epoch, batch))

                with tboard_train_writer.as_default():
                    test_loss = 0
                    norm = 0
                    for i in range(0, min(2000, len(valid_X)), BATCH_SIZE):
                        x = valid_X[i : i+BATCH_SIZE]
                        y = valid_y[i : i+BATCH_SIZE]

                        if len(x) == BATCH_SIZE:
                            test_loss += model.train_step(x, y, tgt_tokenizer, update=False).numpy()
                            norm += 1

                    tf.summary.scalar('valid-loss', test_loss*1.0/norm, step=(epoch * steps_per_epoch + batch))

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


def create_wikievent_source(path, num_examples=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')[:num_examples]
    entity_chain, sent_chain = [], []
    for line in tqdm(lines):
        entity, sentence = line.split('WIKISEP')
        entity = entity.strip()
        sentence = sentence.strip()

        entity_chain.append(preprocess_sentence(entity))
        sent_chain.append(preprocess_sentence(sentence))

    assert len(entity_chain) == len(sent_chain) > 0
    return entity_chain, sent_chain


def create_wikievent_target(path, num_examples=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')[:num_examples]
    assert len(lines) > 0
    return [preprocess_sentence(l) for l in tqdm(lines)]


def load_text(path, num_examples=None):
    return io.open(path, encoding='UTF-8').read().strip().split('\n')[:num_examples]


def create_tokenizer(data, top_k):
    tkzr = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>', filters=' ')
    tkzr.fit_on_texts(data)
    tkzr.index_word[0] = '<pad>'
    tkzr.word_index['<pad>'] = 0
    return tkzr


def create_sequences(data, tokenizer, pad=True):
    seqs = tokenizer.texts_to_sequences(data)
    if pad:
        vecs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post')
        return vecs
    else:
        return seqs


def create_tfdataset(elements, batch_size):
    """
    elements: array of items to go in the dataset.
    All should have same length.
    """
    assert len(elements) >= 2
    _first = elements[0]
    for e in elements:
        assert len(_first) == len(e)

    dataset = tf.data.Dataset.from_tensor_slices(elements)
    dataset.shuffle(len(_first), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def create_wikibio_target(lines, ids, truncate=True):
    assert len(lines) == sum(ids)
    data = []
    ix = 0
    for n in tqdm(ids):
        words = '<start> ' + lines[ix] + ' <end>'
        if truncate:
            words = ' '.join(words.split()[:35])
        
        data.append(words)
        ix += n

    return data

def create_wikibio_source(lines):
    sep = '<sep>'
    data = []
    for ix, l in tqdm(enumerate(lines), total=len(lines)):
        clean = []
        s, e = '', ''

        for word in l.split():
            if len(re.findall(':', word)) != 1:
                continue

            key, val = word.split(':')
            if val != '<none>':
                match = re.search(r'_[0-9]+$', key)
                if match is None:
                    clean.append(key)
                    clean.append(val)
                    clean.append(sep)

                    s = ''
                    e = ''
                else:
                    holder = key[:match.start()]
                    if s:
                        if s == holder:
                            e += val + ' '
                        else:
                            clean.append(s)
                            clean.append(e.strip())
                            clean.append(sep)

                            s = holder
                            e = val + ' '

                    else:
                        s = holder
                        e = val + ' '

        if s:
            clean.append(s)
            clean.append(e.strip())
            clean.append(sep)

        tokens = '<start> ' + ' '.join(clean).strip(sep).strip() + ' <end>'
        tokens = ' '.join(tokens.split()[:80])
        data.append(tokens)

    return data


def simple_load(path, num_examples=None):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    return ['<start> ' + l.strip().replace('#', '1') + ' <end>' for l in lines[:num_examples]]


def wikibiodata(top_k, num_examples=None, batch_size=32):
    maindir = '../wikipedia-biography-dataset/wikipedia-biography-dataset/'
    path_train_src = os.path.join(maindir, 'train', 'train.box')
    path_train_tgt = os.path.join(maindir, 'train', 'train.sent')
    path_train_nbs = os.path.join(maindir, 'train', 'train.nb')

    raw_source = load_text(path_train_src)
    raw_target = load_text(path_train_tgt)
    nbs = [int(x) for x in load_text(path_train_nbs)]
    
    source = create_wikibio_source(raw_source)
    target = create_wikibio_target(raw_target, nbs)

    print("Source:", len(source))
    print("Target:", len(target))
    assert len(source) == len(target)

    src_tokenizer = create_tokenizer(source, top_k)
    tgt_tokenizer = create_tokenizer(target, top_k)

    source_vecs = create_sequences(source, src_tokenizer)
    target_vecs = create_sequences(target, tgt_tokenizer)

    dataset = create_tfdataset((source_vecs, target_vecs), batch_size=batch_size)
    return dataset, src_tokenizer, tgt_tokenizer, len(source)//batch_size


def datastuff(top_k, num_examples=None, batch_size=None, use_hgft=False):
    """
    Returns
    dataset: the dataset iterator
    (src_tokenizer, tgt_tokenizer): source and target tokenizers
    steps_per_epoch
    max_target_seq_len
    (X_test, y_ytest) 
    """
    maindir = '../WikiEvent'
    path_train_src = os.path.join(maindir, 'train.src')
    path_train_tgt = os.path.join(maindir, 'train.tgt')

    if use_hgft is False:
        source, retrieved = create_wikievent_source(path_train_src, num_examples)
        target = create_wikievent_target(path_train_tgt, num_examples)

        print("source  :", len(source))
        print("retrived:", len(retrieved))
        print("target  :", len(target))

        assert len(source) == len(retrieved) == len(target)

        src_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>', filters=' ')
        src_tokenizer.fit_on_texts(source + retrieved)
        # Set <pad> AFTER fitting on texts!
        src_tokenizer.index_word[0] = '<pad>'
        src_tokenizer.word_index['<pad>'] = 0

        source_seqs = src_tokenizer.texts_to_sequences(source)
        source_vecs = tf.keras.preprocessing.sequence.pad_sequences(source_seqs, padding='post')

        retrvd_seqs = src_tokenizer.texts_to_sequences(retrieved)
        retrvd_vecs = tf.keras.preprocessing.sequence.pad_sequences(retrvd_seqs, padding='post')

        # Now, for the target ...
        tgt_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token='<unk>', filters=' ')
        tgt_tokenizer.fit_on_texts(target)
        # Set <pad> AFTER fitting on texts!
        tgt_tokenizer.index_word[0] = '<pad>'
        tgt_tokenizer.word_index['<pad>'] = 0
        target_seqs = tgt_tokenizer.texts_to_sequences(target)
        target_vecs = tf.keras.preprocessing.sequence.pad_sequences(target_seqs, padding='post')

        indices = list(range(len(source)))
        random.shuffle(indices)
        slice_index = int(len(indices) * 0.99)

        X_train, X_test = source_vecs[:slice_index], source_vecs[slice_index:]
        r_train, r_test = retrvd_vecs[:slice_index], retrvd_vecs[slice_index:]
        y_train, y_test = target_vecs[:slice_index], target_vecs[slice_index:]

        dataset = tf.data.Dataset.from_tensor_slices((X_train, r_train, y_train)).shuffle(len(X_train))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=batch_size)

        max_targ_len = target_vecs.shape[1]

        return dataset, (src_tokenizer, tgt_tokenizer), len(X_train)//batch_size, max_targ_len, (X_test, r_test, y_test)

    else:
        src_tokenizer = Tokenizer.from_file('hgf_tokenizers/tokenizer_train.src.txt_1.json')
        tgt_tokenizer = Tokenizer.from_file('hgf_tokenizers/tokenizer_train.tgt.txt_1.json')

        source = simple_load(path_train_src)
        target = simple_load(path_train_tgt)
        assert len(source) == len(target)
        print("source:", len(source))
        print("target:", len(target))

        # td_source = tf.data.TextLineDataset([path_train_src])
        # td_target = tf.data.TextLineDataset([path_train_tgt])
        
        indices = list(range(len(source)))
        random.shuffle(indices)
        slice_index = int(len(indices) * 0.9995)

        print("encoding now ...")
        source = [src_tokenizer.encode(r).ids for r in tqdm(source)]
        target = [tgt_tokenizer.encode(r).ids for r in tqdm(target)]

        source = tf.keras.preprocessing.sequence.pad_sequences(source, padding='post')
        target = tf.keras.preprocessing.sequence.pad_sequences(target, padding='post')

        X_train, X_test = source[:slice_index], source[slice_index:]
        y_train, y_test = target[:slice_index], target[slice_index:]

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=batch_size)

        max_targ_len = max([len(r) for r in y_test])

        return dataset, (src_tokenizer, tgt_tokenizer), len(X_train)//batch_size, max_targ_len, (X_test, y_test)


if __name__ == '__main__':
    main()
