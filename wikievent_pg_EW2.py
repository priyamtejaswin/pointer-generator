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


# Some strange TF hack to avoid CUDNN errors ...
physical_devices = tf.config.list_physical_devices('GPU')
for phydev in physical_devices:
    tf.config.experimental.set_memory_growth(phydev, enable=True)


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

hps = HPS()
vocab = Vocab(
    '../WikiEvent/train.vocab', 85000
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
    
    def initialize_hidden_state(self, size=None):
        if size is None:
            size = self.batch_sz
        return [tf.zeros((size, self.enc_units)), tf.zeros((size, self.enc_units))] * 2 


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
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong', pointer_gen=False):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    self.pointer_gen = pointer_gen
    
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
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc if pointer_gen is False else None)

    # Dense layer for p_gen
    self.dense_gen = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    
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
    outputs, alignments, states, _a, _b = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output])
    if self.pointer_gen is False:
        return outputs  # Shape of output.rnn_output : [batch X time X vocab]
    else:
        # Shape of output.rnn_output : [batch X time X attention_units]
        # Shape of alignments: [batch X time X src_seq_len]
        
        # 1. Compute p_gens for each decoding timestep.
        p_gens = self.dense_gen(
            tf.concat(states + [x, outputs.rnn_output], axis=-1)
        )

        return


# Test decoder stack
decoder = Decoder(vocab.size(), embedding_dim, units, hps.batch_size, 'luong', pointer_gen=True)
sample_x = tf.random.uniform((hps.batch_size, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)

sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

# ## Define the optimizer and the loss function
optimizer = tf.keras.optimizers.Adam()

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
# @tf.function
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

train_dataset.stop()


# ## Restore the latest checkpoint and test
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def beam_evaluate_sentence(inputs, start_id, end_id, beam_width=3):
  inference_batch_size = inputs.shape[0]
  result = ''

  enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))] * 2
  enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

  start_tokens = tf.fill([inference_batch_size], start_id)

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
  outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_id, initial_state=decoder_initial_state)
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


def beam_translate(inputs, vocab):
  start_id = vocab.word2id(vocab.START_DECODING)
  end_id = vocab.word2id(vocab.STOP_DECODING)
  result, beam_scores = beam_evaluate_sentence(inputs, start_id, end_id)
  # print(result.shape, beam_scores.shape)
  best = []
  for beam, score in zip(result, beam_scores):
    # print(beam.shape, score.shape)
    output = [a[:np.where(a == end_id)[0][0]] if end_id in a else a for a in beam]
    beam_score = [a.sum() for a in score]
    # print('Input: %s' % (sentence))
    # for i in range(len(output)):
    #   print('{} Predicted translation: {}  {}'.format(i+1, output[i], beam_score[i]))
    best.append(output[np.argmin(beam_score)])

  assert len(inputs) == len(best)
  return [' '.join([vocab.id2word(x) for x in row]) for row in best]


# ## Batch Decoding ...
test_dataset = KerasBatcher(
    '../WikiEvent/test.src',
    '../WikiEvent/test.tgt',
    vocab,
    hps,
    batch_size=5
)

write_preds = []
write_targs = []
print("Evaluating test set ...")
for tbc in tqdm(test_dataset):
    x, y = tbc.enc_batch, tbc.original_abstracts
    hypo = beam_translate(x, vocab)
    
    for text in hypo:
        clean = []
        for w in text.split():
            if w == vocab.START_DECODING:
                pass
            elif w == vocab.STOP_DECODING:
                break
            else:
                clean.append(w)
                
        write_preds.append(' '.join(clean))
        
    for text in y:
        clean = []
        for w in text.lower().split():
            if w == vocab.START_DECODING:
                pass
            elif w == vocab.STOP_DECODING:
                break
            else:
                clean.append(w)
                
        write_targs.append(' '.join(clean))    


# Save preds and truth to disk ...
with open('./results/wikievent_noret_basicdecoder_hypos.txt', 'w') as fp:
    fp.write('\n'.join(write_preds) + '\n')
print("\nHypos written to disk.")

with open('./results/wikievent_noret_basicdecoder_targets.txt', 'w') as fp:
    fp.write('\n'.join(write_targs) + '\n')
print("\nTargets written to disk.")




