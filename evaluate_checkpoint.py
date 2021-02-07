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
# from seq2seqwithatt import S2SModel, wikibiodata, create_glove_matrix
# from seq2seqwithatt import create_wikibio_source, create_wikibio_target, create_sequences, load_text


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

    LSTM_EMBED_SIZE = 256
    WORD_EMBED_SIZE = 100
    VOCAB_SIZE = 50000
    BATCH_SIZE = 32
    EPOCHS = 5
    MAX_TARG_LEN = 35

    _, src_tokenizer, tgt_tokenizer, _ = wikibiodata(top_k=VOCAB_SIZE, num_examples=None, batch_size=BATCH_SIZE, withdata=False)
    embed_src = create_glove_matrix(src_tokenizer, '../glove/glove.6B.100d.txt', VOCAB_SIZE, WORD_EMBED_SIZE)
    embed_tgt = create_glove_matrix(tgt_tokenizer, '../glove/glove.6B.100d.txt', VOCAB_SIZE, WORD_EMBED_SIZE)

    # Create model ...
    model = S2SModel(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE,
                        pretr_src_embeds=embed_src, pretr_tgt_embeds=embed_tgt)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Validation data.
    valid_dir = '../wikipedia-biography-dataset/wikipedia-biography-dataset/test'
    valid_source = create_wikibio_source(load_text(os.path.join(valid_dir, 'test.box')))
    valid_X = create_sequences(valid_source, src_tokenizer)
    valid_nbs = [int(x) for x in load_text(os.path.join(valid_dir, 'test.nb'))]
    valid_target = create_wikibio_target(load_text(os.path.join(valid_dir, 'test.sent')), valid_nbs)
    valid_y = create_sequences(valid_target, tgt_tokenizer)

    # print("Doesn't work without this ...", model.train_step(
    #     valid_X[:BATCH_SIZE, :], valid_X[:BATCH_SIZE, :35], tgt_tokenizer, update=False))

    towrite = []
    INFER_SIZE = 5
    progbar = tqdm(range(0, 25, INFER_SIZE), total=len(valid_X)//INFER_SIZE, desc='loss: ')
    for i in progbar:
        x = valid_X[i : i+INFER_SIZE]
        y = valid_y[i : i+INFER_SIZE]

        batch_loss = model.train_step(x, y, update=False)
        progbar.set_description("loss: %.3f" % (batch_loss.numpy()))

        if len(x) == INFER_SIZE:
            eval_ans = generate_output(x, model.encoder, model.decoder, LSTM_EMBED_SIZE,
                                                tgt_tokenizer.word_index['<start>'], tgt_tokenizer.word_index['<end>'])
            preds = tgt_tokenizer.sequences_to_texts(eval_ans)
            towrite.extend(preds)

    
if __name__ == '__main__':
    ckpt_dir, ckpt_name = sys.argv[1], sys.argv[2]
    main(ckpt_dir, ckpt_name)
