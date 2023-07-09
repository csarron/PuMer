# PuMer (ACL 2023)

This repo is the official implementation for the paper "PuMer: Pruning and Merging Tokens for Efficient Vision Language Models", [paper](https://aclanthology.org/2023.acl-long.721/)

## Usage

### Install

<!-- install [poetry](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions) -->
install [miniforge](https://github.com/conda-forge/miniforge#miniforge3) (same as conda, more portable)
create a python environment: `conda env create -f env.yaml`, activate it: `conda activate pumer`
clone this repo: [git@github.com:csarron/pumer.git](https://github.com/csarron/pumer.git)

test cuda: `python -c "import torch;print(torch.cuda.is_available())"`

get torch env: `python -m torch.utils.collect_env`

install: `pip install -e .`

for local developement purposes: `pip install -e ".[dev]"`

The `env-frozen.yaml` is generated via `conda env export | grep -v "^prefix: | pumer==" > env-frozen.yaml`


### Prepare data and pretrained models

see [notes/data.md](./notes/data.md) for data preprocessing

see `cli/prep/convert_ckpt.py` for converting original pretrained METER and ViLT checkpoints

below is the file layout after preparation:

```text
# tree -h data
├── [4.0K]  ckpt
│   └── [4.0K]  converted
│       ├── [4.0K]  meter_pretrain_384
│       │   ├── [ 674]  config.json
│       │   └── [1.3G]  pytorch_model.bin
│       ├── [4.0K]  meter_pretrain_irtr_384
│       │   ├── [ 729]  config.json
│       │   └── [1.2G]  pytorch_model.bin
│       ├── [4.0K]  meter_pretrain_nlvr2_288
│       │   ├── [ 674]  config.json
│       │   └── [1.3G]  pytorch_model.bin
│       ├── [4.0K]  vilt_pretrain
│       │   ├── [ 619]  config.json
│       │   └── [518M]  pytorch_model.bin
│       ├── [4.0K]  vilt_pretrain_irtr
│       │   ├── [ 718]  config.json
│       │   └── [426M]  pytorch_model.bin
│       └── [4.0K]  vilt_pretrain_nlvr2
│           ├── [ 619]  config.json
│           └── [518M]  pytorch_model.bin
├── [4.0K]  datasets
│   ├── [4.0K]  irtr
│   │   ├── [390K]  flickr30k-test.jsonl
│   │   ├── [ 11M]  flickr30k-train.jsonl
│   │   ├── [397K]  flickr30k-val.jsonl
│   │   ├── [ 10M]  mscoco-restval.jsonl
│   │   ├── [1.7M]  mscoco-test.jsonl
│   │   ├── [ 28M]  mscoco-train.jsonl
│   │   └── [1.7M]  mscoco-val.jsonl
│   ├── [4.0K]  nlvr2
│   │   ├── [3.6M]  dev.json
│   │   ├── [3.6M]  test1.json
│   │   └── [ 39M]  train.json
│   ├── [4.0K]  snli-ve
│   │   ├── [ 16M]  snli_ve_dev.jsonl
│   │   ├── [ 16M]  snli_ve_test.jsonl
│   │   └── [464M]  snli_ve_train.jsonl
│   └── [4.0K]  vqa2
│       ├── [ 57K]  vqa2_ans2label.json
│       ├── [ 39K]  vqa2_label2ans.json
│       ├── [161K]  vqa2-small.jsonl
│       ├── [ 45M]  vqa2-test2015.jsonl
│       ├── [ 71M]  vqa2-train2014.jsonl
│       └── [ 34M]  vqa2-val2014.jsonl
└── [4.0K]  lmdb
    ├── [ 13G]  coco-test2015.lmdb
    ├── [ 19G]  coco-trainval2014.lmdb
    ├── [4.2G]  flickr30k_images.lmdb
    ├── [837M]  nlvr2-dev.lmdb
    ├── [837M]  nlvr2-test1.lmdb
    └── [ 11G]  nlvr2-train.lmdb
```


### Training and Evaluation

see [notes/cmd.md](./notes/cmd.md) for example usage;

checkout [https://huggingface.co/csarron](https://huggingface.co/csarron) for finetuend checkpoints: 
(`-ft` is original finetuned model, `p0.x-r0.x-t0.x-xxx` is our PuMer model)

```text
vilt-vqa2-ft
vilt-vqa2-p0.1-r0.3-t0.2-258
vilt-ve-ft 
vilt-ve-p0.1r0.3t0.2-2468 
vilt-nlvr2-ft 
vilt-nlvr2-p0.1r0.3t0.2-258
meter-vqa2-ft
meter-vqa2-p0.2r0.2t0.2-0246
meter-ve-ft 
meter-ve-p0.3r0.5t0.2-0246 
meter-nlvr2-ft 
meter-nlvr2-p0.3r0.5t0.2-246
```

### Profiling FLOPs

see [notes/profile.md](./notes/profile.md)

## FAQs

- set `TRANSFORMERS_OFFLINE=1` after first use, otherwise sometime it will report 504 error due to always online look up.

## Misc

- ignore the code in `src/pumer/model/pruner.py` (deprecated and unused), needs cleanup
- the current codebase contain many clutters and experimental code that is not related to PuMer implementation, please ignore that.

## Citation

```bibtex
@inproceedings{cao-etal-2023-pumer,
    title = "{P}u{M}er: Pruning and Merging Tokens for Efficient Vision Language Models",
    author = "Cao, Qingqing  and
      Paranjape, Bhargavi  and
      Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.721",
    pages = "12890--12903",
}

```