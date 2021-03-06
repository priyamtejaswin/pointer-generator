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

### WikiEvent
Open-domain event text generation task.

#### S2S-E
Considers the entities only.
```
score_type,low,mid,high
rouge1-R,0.313428,0.319884,0.326313
rouge1-P,0.350305,0.357162,0.364144
rouge1-F,0.324943,0.330882,0.337045
rouge2-R,0.135235,0.140901,0.147425
rouge2-P,0.153485,0.159741,0.166801
rouge2-F,0.140610,0.146707,0.153264
rouge4-R,0.040359,0.045020,0.049642
rouge4-P,0.046750,0.051670,0.056944
rouge4-F,0.042332,0.046887,0.051742
rougeL-R,0.265609,0.272080,0.277987
rougeL-P,0.298885,0.305543,0.312448
rougeL-F,0.275966,0.282179,0.288476
```

#### S2S-EW2
Considers entities and supporting sentences as a single sequence (`max_len=120`).
```
score_type,low,mid,high
rouge1-R,0.381582,0.388686,0.396485
rouge1-P,0.367045,0.373815,0.381158
rouge1-F,0.367159,0.373685,0.380720
rouge2-R,0.195440,0.202547,0.210077
rouge2-P,0.186044,0.192317,0.199767
rouge2-F,0.186761,0.193084,0.200478
rouge4-R,0.074683,0.080686,0.086981
rouge4-P,0.069943,0.075140,0.081143
rouge4-F,0.070509,0.075928,0.081925
rougeL-R,0.326473,0.333815,0.341458
rougeL-P,0.312558,0.319239,0.326027
rougeL-F,0.313363,0.319964,0.326819
```
