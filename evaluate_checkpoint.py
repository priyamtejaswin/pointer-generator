#!/usr/bin/env python
"""
created at: Thu 28 Jan 2021 03:55:04 AM EST
created by: Priyam Tejaswin (ptejaswi)

Script to score the model checkpoint.
Usage:
`python evaluate_checkpoint.py ckpts_dir ckpt-number`

Script also implements Greedy and BeamSearch decoding.
"""


import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa


def generate_output(seqs, encoder, decoder, units, start_index, end_index):
    """
    Expects a sequence/vector of token-ids.
    encoder: encoder object.
    decoder: decoder object.
    start_index: the token index for `<start>`
    Minimum batch size for `seqs` is 1!
    """
    assert len(seqs.shape) == 2
    
    inputs = tf.convert_to_tensor(seqs)
    inference_batch_size = inputs.shape[0]

    enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size, units))]
    enc_out, enc_h, enc_c = encoder(inputs)

    start_tokens = tf.fill([inference_batch_size], start_index)

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
    
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    decoder.attention_mechanism.setup_memory(enc_out)
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

    decoder_embedding_matrix = decoder.embedding.variables[0]
    
    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token=end_index, initial_state=decoder_initial_state)
    
    return outputs.sample_id.numpy()


def main(ckpt_dir, ckpt_name):
    assert os.path.isdir(ckpt_dir)

