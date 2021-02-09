This repository aims to implement reliable baseline models for sequence-to-sequence tasks in `Tf2.1` -- most implementations available online are outdated (`Tf1`) and cannot be used/merged easily with other codebases, extensions and ideas. The [original codebase](https://github.com/abisee/pointer-generator) serves as a good reference point, although some of it is extremely outdated and slow (especially the inference).

## Major changes and updates
* Uses `tf.data` for all i/o, pre-processing and data pipelines.
* Model code is highly modular, and extensible -- `tk.keras.Layer` abstractions.
* Inference (greedy and Beam) is fast and supports a batch of samples -- uses abstractions from [Tf Addons](https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/BasicDecoder).
* Tested on Tf 2.1
* Performance verified on different datasets.

## Models
* [x] Seq2Seq + Attention [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf)
* [ ] Seq2Seq + Attention + Coverage [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf)

## Performance
Models were trained from scratch on the entire train set. Results were reported on the test set.

### WikiBio
Table-to-text generation task, [Lebret et al. (2016)](https://arxiv.org/pdf/1603.07771.pdf).
* Src/tgt vocabs at 70k; src text limited to 80 tokens, and target text to 35 tokens.
* Model trained for 3 epochs, batch-size of 32, data shuffled after each epoch.
* Text generated using Greedy decoding.
```
score_type,low,mid,high
rouge1-R,0.630135,0.631606,0.633259
rouge1-P,0.712288,0.713883,0.715423
rouge1-F,0.650755,0.652152,0.653549
rouge2-R,0.476047,0.477881,0.479600
rouge2-P,0.537347,0.539204,0.541024
rouge2-F,0.490887,0.492712,0.494322
rougeL-R,0.610680,0.612184,0.613858
rougeL-P,0.689643,0.691277,0.692798
rougeL-F,0.630374,0.631814,0.633269
```
