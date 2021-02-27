#!/usr/bin/env python
"""
created at: Sat 27 Feb 2021 08:58:19 AM EST
created by: Priyam Tejaswin

Make the Vocab file for See's batcher code.
Format:
```
word1,freq
word2,freq
...
```
Words should be sorted in dec. order of frequency.
"""


import csv
from collections import Counter
import plac


@plac.pos('files', "Filepaths,comma,separated")
@plac.pos('outpath', "Path to save to.")
def main(files, outpath):
    """
    Make the Vocab file for See's batcher code.
    Words should be sorted in dec. order of frequency.
    ```
    word1,freq
    word2,freq
    ...
    ```
    """
    inpaths = [f.strip() for f in files.split(',')]
    vocab = Counter()
    for path in inpaths:
        with open(path) as fp:
            text = fp.read().lower().strip().replace('\n', ' ')
            words = text.strip().split()
            vocab.update(words)
            print("Updted with %d words." % len(words))

    print("Writing to %s ..." % outpath)
    with open(outpath, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(vocab.most_common())
    
    print("Done.")


if __name__ == '__main__':
    plac.call(main)
