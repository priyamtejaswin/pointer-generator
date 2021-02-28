#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_addons as tfa
import unicodedata
import re
import numpy as np
import os
import io
import time
import datetime
from tqdm import tqdm
from batcher import KerasBatcher
from data import Vocab


physical_devices = tf.config.list_physical_devices('GPU')
for phydev in physical_devices:
    tf.config.experimental.set_memory_growth(phydev, enable=True)


class NMTDataset:
    def __init__(self, problem_type='en-spa'):
        self.problem_type = 'en-spa'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
    

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
    
    def load_text(self, path, num_examples=None):
        return io.open(path, encoding='UTF-8').read().strip().split('\n')[:num_examples]

    def create_wikievent_source(self, lines):
        data = []
        for ix, l in enumerate(lines):
            clean = l.strip().lower()
            # entities, sentences = clean.split('WIKISEP')
            # entities = entities.strip().lower()
            # sentences = sentences.strip().lower()
            # entities = '<start> ' + entities + ' <end>'

            text = '<start> ' + ' '.join(clean.split()[:250]) + ' <end>'
            data.append(text)

        return data

    def create_wikievent_target(self, lines, truncate=True):
        data = []
        for ix, l in enumerate(lines):
            clean = l.strip().lower()
            if truncate:
                clean = ' '.join(clean.split()[:40])

            data.append('<start> ' + clean + ' <end>')

        return data
            
    
    def create_wikibio_target(self, lines, ids, truncate=True):
        assert len(lines) == sum(ids)
        data = []
        ix = 0
        for n in tqdm(ids):
            words = '<start> ' + lines[ix] + ' <end>'
            if truncate:
                words = ' '.join(words.split()[:35])  # 35

            data.append(words)
            ix += n

        return data
    
    def create_wikibio_source(self, lines, answers, indices):
        """
        Creates the dataset.
        lines: Original bio data.
        answers: Paragraphs retrieved using bio data as query.
        indices: Line numbers corresponding to the answers file.
        """
        assert len(indices) == len(lines)
        assert indices[-1] == len(answers)
        
        prev = 0
        sep = '<sep>'
        data = []
        for ix, l in enumerate(lines):#, total=len(lines)):
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

            tokens = '<start> ' + ' '.join(clean).strip(sep).strip()
            tokens = ' '.join(tokens.split()[:80])
            
            extra = answers[prev : indices[ix]]
            for ret in extra:
                tokens += ' <ret> ' + ret
                
            tokens = ' '.join(tokens.split()[:250]) + ' <end>'
            data.append(tokens)
            prev = indices[ix]

        return data
    

    ## Step 1 and Step 2 
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w
    
    def create_dataset(self, path, num_examples):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        sent_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        return zip(*sent_pairs)
        

    def create_wikibio_data(self, maindir, num_examples):
        """
        path: Dir location. Must contain train file.
        """
        path_train_src = os.path.join(maindir, 'train', 'train.box')
        path_train_tgt = os.path.join(maindir, 'train', 'train.sent')
        path_train_nbs = os.path.join(maindir, 'train', 'train.nb')
        
        nbs = [int(x) for x in self.load_text(path_train_nbs)[:num_examples]]
        all_targets = self.load_text(path_train_tgt)[:sum(nbs)]
        target = self.create_wikibio_target(all_targets, nbs)
        
        path_ans = os.path.join(maindir, 'train', 'train.ans')
        path_howmany = os.path.join(maindir, 'train', 'train.ans.ix')
        howmany = [int(x) for x in self.load_text(path_howmany)[:num_examples]]
        source = self.create_wikibio_source(self.load_text(path_train_src, num_examples=num_examples), 
                                            self.load_text(path_ans, num_examples=howmany[-1]),
                                            howmany)
        
        assert len(source) == len(target)
        return source, target

    
    def create_wikievent_data(self, maindir, subset, num_examples=None):
        """
        Path to train/dev/test files for WikiEvent.
        subset: {train, dev, test}
        """
        path_src = os.path.join(maindir, subset + '.src')
        path_tgt = os.path.join(maindir, subset + '.tgt')

        source = self.create_wikievent_source(self.load_text(path_src, num_examples=num_examples))
        target = self.create_wikievent_target(self.load_text(path_tgt, num_examples=num_examples))

        assert len(source) == len(target)
        return source, target


    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language
        
        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>', num_words=50000)
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path, subset, num_examples=None):
        """
        Creating cleaned input, output pairs.
        Subset: {train, dev, test}
        """
        # targ_lang, inp_lang = self.create_dataset(path, num_examples)
        # inp_lang, targ_lang = self.create_wikibio_data(path, num_examples)
        
        inp_lang, targ_lang = self.create_wikievent_data(maindir=path, subset=subset, num_examples=num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


    def call(self, path, subset, num_examples, BUFFER_SIZE, BATCH_SIZE):
        """
        Creates source, target tensors with tokenizers.
        Returns TF Dataset, tokenizers and steps_per_epoch.
        """
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(path, subset, num_examples)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
        train_dataset = train_dataset.shuffle(len(input_tensor), reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=BATCH_SIZE)

        return train_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer, len(input_tensor)//BATCH_SIZE


class HPS(object):
    """
    Container for all hyperparameters.
    """
    def __init__(self ):
        self.batch_size = 32
        self.max_enc_steps = 120
        self.max_dec_steps = 40
        self.pointer_gen = True


BUFFER_SIZE = 32
BATCH_SIZE = 32
# Let's limit the #training examples for faster training
# num_examples = None
# dataset_creator = NMTDataset('en-spa')
# train_dataset, inp_lang, targ_lang, steps_per_epoch = dataset_creator.call('../WikiEvent', 'train',
#                                                                             num_examples, BUFFER_SIZE, BATCH_SIZE)

hps = HPS()
vocab = Vocab(
    '../WikiEvent/train.vocab', 80000
)
sequence = KerasBatcher(
    '../WikiEvent/train.src',
    '../WikiEvent/train.tgt',
    vocab,
    hps
)
steps_per_epoch = len(sequence)
train_dataset = tf.keras.utils.OrderedEnqueuer(sequence, shuffle=True)

example_bo = next(iter(sequence))
example_input_batch, example_target_batch = example_bo.enc_batch, example_bo.dec_batch
print("Batch inputs ...")
print(example_input_batch.shape, example_target_batch.shape)

print("First input ids ...")
print(example_input_batch[0])
print("First input tokens ...")
print([vocab.id2word(i) for i in example_input_batch[0] if vocab.id2word(i) != vocab.PAD_TOKEN])

print()
print("First target ids ...")
print(example_target_batch[0])
print("First target tokens ...")
print([vocab.id2word(i) for i in example_target_batch[0] if vocab.id2word(i) != vocab.PAD_TOKEN])

# ### Some important parameters
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 256
units = 512

print("max_length_source, max_length_target, vocab_size")
print(max_length_input, max_length_output, vocab.size())


##### Encoder definitions 
class BiLSTMEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(BiLSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        ## -------- BiLSTM layer in Encoder -------- ##
        self.lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.enc_units, recurrent_initializer='glorot_uniform',
                                 return_sequences=True, return_state=True)
        )
        
        ## -------- Hidden dense -------- ##
        self.hid_dense = tf.keras.layers.Dense(self.enc_units)
        
        ## -------- Carry dense -------- ##
        self.car_dense = tf.keras.layers.Dense(self.enc_units)

        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, fh, fc, bh, bc = self.lstm_layer(x, initial_state=hidden)
        hid = tf.concat([fh, bh], axis=-1)
        car = tf.concat([fc, bc], axis=-1)
        return output, self.hid_dense(hid), self.car_dense(car)
    
    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))] * 2 


## Test Encoder Stack
# encoder = BiLSTMEncoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
encoder = BiLSTMEncoder(vocab.size(), embedding_dim, units, hps.batch_size)


# sample input
sample_hidden = encoder.initialize_hidden_state()
print ('Encoder init shapes: [fw_h, fw_c, bw_h, bw_c] {}'.format([_.shape for _ in sample_hidden]))
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
    super(Decoder, self).__init__()
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
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

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
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output])
    return outputs


# Test decoder stack

decoder = Decoder(vocab.size(), embedding_dim, units, hps.batch_size, 'luong')
sample_x = tf.random.uniform((hps.batch_size, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)


sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

# ## Define the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam()

@tf.function
def loss_function(real, pred):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
  mask = tf.cast(mask, dtype=loss.dtype)  
  loss = mask* loss
  loss = tf.reduce_mean(loss)
  return loss  


# ## Checkpoints (Object-based saving)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# ## One train_step operations
@tf.function
def train_step(batch, enc_hidden, update=True):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_h, enc_c = encoder(batch.enc_batch, enc_hidden)

    dec_input = batch.dec_batch  # Ignores [STOP] token
    real = batch.target_batch  # Ignores [START] token

    # Set the AttentionMechanism object with encoder_outputs
    decoder.attention_mechanism.setup_memory(enc_output)

    # Create AttentionWrapperState as initial_state for decoder
    decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
    pred = decoder(dec_input, decoder_initial_state)
    logits = pred.rnn_output
    loss = loss_function(real, logits)

  if update:
      variables = encoder.trainable_variables + decoder.trainable_variables
      gradients = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(gradients, variables))

  return loss


# ## Train the model
tboard_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_train_log_dir = 'logs/' + tboard_time + '/wikievent'
tboard_train_writer = tf.summary.create_file_writer(tboard_train_log_dir)

print("Starting main epoch loop ...")
train_dataset.start(workers=3, max_queue_size=BUFFER_SIZE)
EPOCHS = 10
TOTAL = steps_per_epoch * EPOCHS
enc_hidden = encoder.initialize_hidden_state()

progbar = tqdm(enumerate(train_dataset.get()), total=TOTAL, desc='avg loss: ')
for ix, batch in progbar:
    batch_loss = train_step(batch, enc_hidden, update=True)
    progbar.set_description("avg loss: %.3f" % batch_loss.numpy())
    
    if ix % 10 == 0:
        with tboard_train_writer.as_default():
            tf.summary.scalar('train-loss', batch_loss.numpy(), step=ix)
    
    if ix%1000 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    if (ix+1) == TOTAL:
        print("Completed.")
        break


# ## Use tf-addons BasicDecoder for decoding
def evaluate_sentence(sentence):
  """
  `sentence` is RAW!
  It is not `pre-processed`.
  `create_wikibio_source` is called IN THIS FUNCTION!
  """
#   sentence = dataset_creator.preprocess_sentence(sentence)
#   inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
#   inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                           maxlen=max_length_input,
#                                                           padding='post')

  if isinstance(sentence, str):
    sentence = [sentence]
  
#   inputs = []
#   for s in sentence:
#     inputs.append([inp_lang.word_index[i] for i in dataset_creator.preprocess_sentence(s).split(' ')])

  inputs = inp_lang.texts_to_sequences(dataset_creator.create_wikievent_source(sentence))
  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                          maxlen=max_length_input,
                                                          padding='post')
  inputs = tf.convert_to_tensor(inputs)
  inference_batch_size = inputs.shape[0]
  result = ''

  enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))] * 2
  enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

  dec_h = enc_h
  dec_c = enc_c

  start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
  end_token = targ_lang.word_index['<end>']

  greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

  # Instantiate BasicDecoder object
  decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc, maximum_iterations=40)
  # Setup Memory in decoder stack
  decoder.attention_mechanism.setup_memory(enc_out)

  # set decoder_initial_state
  decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)


  ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
  ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
  ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

  decoder_embedding_matrix = decoder.embedding.variables[0]
  
  outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
  return outputs.sample_id.numpy()

def translate(sentence):
  result = evaluate_sentence(sentence)
  print(result)
  result = targ_lang.sequences_to_texts(result)
  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))


# ## Restore the latest checkpoint and test

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# translate(u'hace mucho frio aqui.')
# translate('''type_1:pope	name_1:michael	name_2:iii	name_3:of	name_4:alexandria	title_1:56th	title_2:pope	title_3:of	title_4:alexandria	title_5:&	title_6:patriarch	title_7:of	title_8:the	title_9:see	title_10:of	title_11:st.	title_12:mark	image:<none>	caption:<none>	enthroned_1:25	enthroned_2:april	enthroned_3:880ended_1:16	ended_2:march	ended_3:907	predecessor_1:shenouda	predecessor_2:i	successor_1:gabriel	successor_2:i	ordination:<none>	consecration:<none>	birth_date:<none> 	birth_name:<none>	birth_place_1:egypt	death_date_1:16	death_date_2:march	death_date_3:907	buried_1:monastery	buried_2:of	buried_3:saint	buried_4:macarius  	buried_5:the	buried_6:great	nationality_1:egyptian	religion_1:coptic	religion_2:orthodox	religion_3:christian	residence_1:saint	residence_2:mark	residence_3:'s	residence_4:church	feast_day_1:16	feast_day_2:march	feast_day_3:-lrb-	feast_day_4:20	feast_day_5:baramhat	feast_day_6:in	feast_day_7:the	feast_day_8:coptic	feast_day_9:calendar	feast_day_10:-rrb-	alma_mater:<none>	signature:<none>	article_title_1:pope	article_title_2:michael	article_title_3:iii	article_title_4:of	article_title_5:alexandria''')


# translate('''name_1:paul\tname_2:f.\tname_3:whelan\timage:<none>\talt:<none>\tcaption:<none>\tbirth_name:<none>\tbirth_date:<none>\tbirth_place:<none>\tdeath_date:<none>\tdeath_place:<none>\tnationality:<none>\tother_names:<none>\tknown_for:<none>\toccupation_1:professor\toccupation_2:of\toccupation_3:computer\toccupation_4:vision\tarticle_title_1:paul\tarticle_title_2:f.\tarticle_title_3:whelan''')


# dataset_creator.create_wikibio_source(['''name_1:paul\tname_2:f.\tname_3:whelan\timage:<none>\talt:<none>\tcaption:<none>\tbirth_name:<none>\tbirth_date:<none>\tbirth_place:<none>\tdeath_date:<none>\tdeath_place:<none>\tnationality:<none>\tother_names:<none>\tknown_for:<none>\toccupation_1:professor\toccupation_2:of\toccupation_3:computer\toccupation_4:vision\tarticle_title_1:paul\tarticle_title_2:f.\tarticle_title_3:whelan'''])


# ## Use tf-addons BeamSearchDecoder 
# 

def beam_evaluate_sentence(sentence, beam_width=3):
  if isinstance(sentence, str):
    sentence = [sentence]

  inputs = inp_lang.texts_to_sequences(dataset_creator.create_wikievent_source(sentence))
  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                          maxlen=max_length_input,
                                                          padding='post')
  inputs = tf.convert_to_tensor(inputs)
  inference_batch_size = inputs.shape[0]
  result = ''

  enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))] * 2
  enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

  dec_h = enc_h
  dec_c = enc_c

  start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
  end_token = targ_lang.word_index['<end>']

  # From official documentation
  # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
  # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
  # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
  # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

  enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
  decoder.attention_mechanism.setup_memory(enc_out)
  # print("beam_with * [batch_size, max_length_input, rnn_units]:", enc_out.shape)

  # set decoder_inital_state which is an AttentionWrapperState considering beam_width
  hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
  decoder_initial_state = decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
  decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

  # Instantiate BeamSearchDecoder
  decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoder.rnn_cell,beam_width=beam_width, output_layer=decoder.fc, maximum_iterations=40)
  decoder_embedding_matrix = decoder.embedding.variables[0]

  # The BeamSearchDecoder object's call() function takes care of everything.
  outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
  # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object. 
  # The final beam predictions are stored in outputs.predicted_id
  # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
  # final_state = tfa.seq2seq.BeamSearchDecoderState object.
  # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated

  
  # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
  # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
  # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
  final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
  beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))
  
  return final_outputs.numpy(), beam_scores.numpy()


def beam_translate(sentence):
  result, beam_scores = beam_evaluate_sentence(sentence)
  # print(result.shape, beam_scores.shape)
  best = []
  for beam, score in zip(result, beam_scores):
    # print(beam.shape, score.shape)
    output = targ_lang.sequences_to_texts(beam)
    output = [a[:a.index('<end>')] if '<end>' in a else a for a in output]
    beam_score = [a.sum() for a in score]
    # print('Input: %s' % (sentence))
    # for i in range(len(output)):
    #   print('{} Predicted translation: {}  {}'.format(i+1, output[i], beam_score[i]))
    best.append(output[np.argmin(beam_score)])

  assert len(sentence) if type(sentence)==list else 1 == len(best)
  return best


# ## Batch Decoding ...

# testX = dataset_creator.load_text('../wikipedia-biography-dataset/wikipedia-biography-dataset/test/test.box')
# testIds = [int(x) for x in dataset_creator.load_text('../wikipedia-biography-dataset/wikipedia-biography-dataset/test/test.nb')]
# testTargs = dataset_creator.load_text('../wikipedia-biography-dataset/wikipedia-biography-dataset/test/test.sent', sum(testIds))
# testY = dataset_creator.create_wikibio_target(testTargs, testIds)

testX = dataset_creator.load_text('../WikiEvent/test.src')
testY = dataset_creator.load_text('../WikiEvent/test.tgt')
assert len(testX) == len(testY)


print(testX[0])


print(testY[0])


print(beam_translate(testX[0]))


write_preds = []
write_targs = []
INFER_SIZE = 4

print("Evaluating test set ...")
for i in tqdm(range(0, len(testX), INFER_SIZE)):
    x, y = testX[i: i+INFER_SIZE], testY[i: i+INFER_SIZE]
    if len(x) == INFER_SIZE:
        hypo = beam_translate(x)
        for text in hypo:
            clean = []
            for w in text.split():
                if w == '<start>':
                    pass
                elif w == '<end>':
                    break
                else:
                    clean.append(w)
                    
            write_preds.append(' '.join(clean))
            
        for text in y:
            clean = []
            for w in text.lower().split():
                if w == '<start>':
                    pass
                elif w == '<end>':
                    break
                else:
                    clean.append(w)
                    
            write_targs.append(' '.join(clean))


with open('./results/wikievent_noret_basicdecoder_hypos.txt', 'w') as fp:
    fp.write('\n'.join(write_preds) + '\n')
    
print("\nHypos written to disk.")

with open('./results/wikievent_noret_basicdecoder_targets.txt', 'w') as fp:
    fp.write('\n'.join(write_targs) + '\n')
    
print("\nTargets written to disk.")




