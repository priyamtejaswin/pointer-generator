#!/usr/bin/env python
"""
created at: Mon 28 Dec 2020 05:14:17 AM EST
created by: Priyam Tejaswin (ptejaswi)

Script to create the HugginFace WordPiece Tokenizer.
!pip install tokenizers
https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
"""


import os
import sys
from tqdm import tqdm


datadir = sys.argv[1]
assert os.path.isdir(datadir), "Source dir does not exist."

filenames = sys.argv[2:]
datafiles = [os.path.join(datadir, f) for f in filenames]
assert len(datafiles)>0, "No source files provided."
for path in datafiles:
    assert os.path.isfile(path), "File %s does not exist." % path


from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(WordPiece())
trainer = WordPieceTrainer(special_tokens=['<pad>', '<start>', '<end>', '<unk>'])
# These special tokens SHOULD be encoded as 0, 1, 2, 3 in the tokenizer!

# So that no chunk is bigger than a word...
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer ...
lines = []
for path in datafiles:
    with open(path) as fp:
        lines.extend([l.strip().replace('#', '1') for l in tqdm(fp.readlines())])

tokenizer.train_from_iterator(iterator=lines, trainer=trainer)

# Check for the dir ...
savepath = './hgf_tokenizers'
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Some weird hacking for unknowns ...
files = tokenizer.model.save(savepath)
tokenizer.model = WordPiece.from_file(*files, unk_token='<unk>')

# Enable padding ...
tokenizer.enable_padding(pad_token='<pad>')
before = tokenizer.encode_batch(['<start> one word <end>', '<start> ZZZ word more <end>'])
assert before[0].ids[0] == 1
assert before[0].ids[3] == 2
assert before[0].ids[4] == 0
assert before[1].ids[0] == 1
assert before[1].ids[4] == 2
assert before[1].ids[1] == 3
print("Before!")
for row in before:
    print(row.tokens)
    print(row.ids)
print()

# Save and reload to check ...
savename = "tokenizer_%s_%d.json" % (filenames[0], len(filenames))
tokenizer.save(os.path.join(savepath, savename))
tokenizer = Tokenizer.from_file(os.path.join(savepath, savename))

# Check ...
after = tokenizer.encode_batch(['<start> one word <end>', '<start> ZZZ word more <end>'])
print("After!")
for row in after:
    print(row.tokens)
    print(row.ids)
print()

assert len(before) == len(after) == 2
for b, a in zip(before, after):
    assert b.tokens == a.tokens
    assert b.ids == a.ids

print("tested and saved. %s" % savename)