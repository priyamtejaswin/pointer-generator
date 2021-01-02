# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding"""

import tensorflow as tf
import numpy as np
# import data

# FLAGS = tf.app.flags.FLAGS
class Flags(object):
  def __init__(self):
    self.beam_size = 5
    self.max_dec_steps = 30
    self.min_dec_steps = 2

FLAGS = Flags()

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state

  def extend(self, token, log_prob, state):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)


def run_beam_search(model, vocab, batch):
  """Performs beam search decoding on the given example.

  Args:
    model: a seq2seq model
    vocab: TOKENIZER!!!
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  # enc_states, dec_in_state = model.run_encoder(batch)
  enc_states = model.encoder(batch)
  # dec_in_state is a LSTMStateTuple TODO: I missed this!! It should be a map of the BiLSTM hidden states ...
  dec_in_state = model.decoder.reset_state(1)
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[vocab.token_to_id('<start>')],
                     log_probs=[0.0],
                     state=dec_in_state) for _ in range(FLAGS.beam_size)]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)
  initial_state = [model.decoder.reset_state(len(batch))] * 2

  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
    latest_tokens = np.array([h.latest_token for h in hyps]) # latest token produced by each hypothesis
    # latest_tokens = [t if t in range(vocab.size()) else vocab.word2id('<unk>') for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    states = np.squeeze(np.array([h.state for h in hyps])) # list of current decoder states of the hypotheses

    # Run one step of the decoder to get the new info
    # (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(batch=batch,
    #                     latest_tokens=latest_tokens,
    #                     enc_states=enc_states,
    #                     dec_init_states=states,
    #                     prev_coverage=prev_coverage)

    step_input = tf.transpose(tf.expand_dims(latest_tokens, 0))
    vocab_logits, final_output, final_carry = model.decoder(enc_states, step_input, states, initial_state)
    initial_state = [final_output, final_carry]

    vocab_probs = tf.nn.softmax(vocab_logits).numpy()
    topk_ids = np.argsort(vocab_probs, axis=1)[:, -FLAGS.beam_size:][:, ::-1]
    topk_log_probs = np.log10([vocab_probs[i][row] for i, row in enumerate(topk_ids)])

    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    

    for i in range(num_orig_hyps):
      h, new_state = hyps[i], final_output[i]  # take the ith hypothesis and new decoder state info
      for j in range(FLAGS.beam_size):  # for each of the top 2*beam_size hyps: TODO: WHY IS THIS 2*BEAMSIZE?
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i, j],
                           log_prob=topk_log_probs[i, j],
                           state=new_state)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == vocab.token_to_id('<end>'): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted[0]

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
