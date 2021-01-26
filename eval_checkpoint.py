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
from seq2seqwithatt import S2SModel, datastuff, create_dataset, create_wikibio_source, create_wikibio_target, wikibiodata, create_glove_matrix
from beam_search import run_beam_search


ckpt_dir, ckpt_prefix = sys.argv[1], sys.argv[2]
assert os.path.isdir(ckpt_dir)

LSTM_EMBED_SIZE = 256
WORD_EMBED_SIZE = 100
VOCAB_SIZE = 50000
BATCH_SIZE = 32
EPOCHS = 5

maindir = '../wikipedia-biography-dataset/wikipedia-biography-dataset/'
path_train_src = os.path.join(maindir, 'test', 'test.box')
path_train_tgt = os.path.join(maindir, 'test', 'test.sent')
path_train_nbs = os.path.join(maindir, 'test', 'test.nb')

source = create_wikibio_source(path_train_src, None)
target = create_wikibio_target(path_train_tgt, path_train_nbs, num_examples=None, truncate=False)
assert len(source) == len(target)

# dataset, (src_tokenizer, tgt_tokenizer), steps_per_epoch, max_targ_len, (X_test, y_test) = wikibiodata(top_k=VOCAB_SIZE, 
#                                                                                                             num_examples=None, 
#                                                                                                             batch_size=BATCH_SIZE)
with open('hgf_tokenizers/src_tokenizer.cpkl', 'rb') as fp:
    src_tokenizer = pickle.load(fp)
with open('hgf_tokenizers/tgt_tokenizer.cpkl', 'rb') as fp:
    tgt_tokenizer = pickle.load(fp)

max_targ_len = 30
# Load and create Glove ...
embed_src = create_glove_matrix(src_tokenizer, '../glove/glove.6B.100d.txt', VOCAB_SIZE, WORD_EMBED_SIZE)
embed_tgt = create_glove_matrix(tgt_tokenizer, '../glove/glove.6B.100d.txt', VOCAB_SIZE, WORD_EMBED_SIZE)

# Create model ...
model = S2SModel(lstm_embed_size=LSTM_EMBED_SIZE, word_embed_size=WORD_EMBED_SIZE, vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE,
                    pretr_src_embeds=embed_src, pretr_tgt_embeds=embed_tgt)

checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
restorepath = os.path.join(ckpt_dir, ckpt_prefix)
checkpoint.restore(restorepath)

towrite = []
counter = 0
for s, t in tqdm(zip(source, target), total=len(source)):
    sequence = np.array([src_tokenizer.texts_to_sequences([s]) for _ in range(5)]).squeeze()
    # pred = model.evaluate(sequence[:1], tokenizer, max_targ_len)

    # Trying beam-search ...
    beamids = run_beam_search(model, tgt_tokenizer, sequence)
    # pred = tgt_tokenizer.decode(beamids)
    pred = tgt_tokenizer.sequences_to_texts([beamids.tokens[1:-1]])[0]

    towrite.append(pred.strip())

    if counter%5 == 0:
        print(t)
        print(pred)

    counter += 1

    with open(restorepath+'.hypo', 'a') as fp:
        fp.write(pred + '\n')

# with open(restorepath+'.targ', 'w') as fp:
#     fp.write('\n'.join([l.lstrip('<start>').rstrip('<end>').strip() for l in target]) + '\n')

print("saved to disk.")