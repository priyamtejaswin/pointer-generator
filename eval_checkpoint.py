#!/usr/bin/env python
"""
created at: Mon 28 Dec 2020 08:25:53 AM EST
created by: Priyam Tejaswin (ptejaswi)

Evaluate a checkpoint.
`python script.py ckpt_dir ckpt_prefix`
"""


import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from seq2seqwithatt import S2SModel, datastuff, create_dataset
from beam_search import run_beam_search


ckpt_dir, ckpt_prefix = sys.argv[1], sys.argv[2]
assert os.path.isdir(ckpt_dir)

LSTM_EMBED_SIZE = 256
WORD_EMBED_SIZE = 128
VOCAB_SIZE = 50000
BATCH_SIZE = 64
EPOCHS = 3

# dataset, (src_tokenizer, tgt_tokenizer), steps_per_epoch, max_targ_len, (X_test, y_test) = datastuff(top_k=VOCAB_SIZE, num_examples=None, batch_size=BATCH_SIZE)
max_targ_len = 25
with open('hgf_tokenizers/src_tokenizer.cpkl', 'rb') as fp:
    src_tokenizer = pickle.load(fp)
with open('hgf_tokenizers/tgt_tokenizer.cpkl', 'rb') as fp:
    tgt_tokenizer = pickle.load(fp)

model = S2SModel(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)

checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
restorepath = os.path.join(ckpt_dir, ckpt_prefix)
checkpoint.restore(restorepath)

source = create_dataset('../org_data/test.src.txt')
target = create_dataset('../org_data/test.tgt.txt')
assert len(source) == len(target)

towrite = []
counter = 0
for s, t in tqdm(zip(source, target), total=len(source)):
    sequence = np.array(src_tokenizer.texts_to_sequences([s]*5))
    # pred = model.evaluate(sequence[:1], tokenizer, max_targ_len)

    # Trying beam-search ...
    beamids = run_beam_search(model, tgt_tokenizer, sequence).tokens[1:-1]
    pred = tgt_tokenizer.sequences_to_texts([beamids])[0]

    towrite.append(pred.strip())

    if counter%5 == 0:
        print(t)
        print(pred)

    counter += 1

with open(restorepath+'.hypo', 'w') as fp:
    fp.write('\n'.join(towrite) + '\n')

# with open(restorepath+'.targ', 'w') as fp:
#     fp.write('\n'.join([l.lstrip('<start>').rstrip('<end>').strip() for l in target]) + '\n')

print("saved to disk.")