
### Preprocessing Dataset

#### VQAv2

Download:
- download questions: `python cli/prep/dl_data.py download_vqa2_questions --out_dir data/datasets/vqa2`
- download annotations (answers): `python cli/prep/dl_data.py download_vqa2_annotations --out_dir data/datasets/vqa2`
- download train/val/test images: `python cli/prep/dl_data.py download_vqa2_images --out_dir data/datasets/vqa2`, or only download test images: `python cli/prep/dl_data.py download_vqa2_images --nodownload_train --nodownload_val --download_test --out_dir data/datasets/vqa2`

Convert:
- convert to simplified jsonl format: `python cli/prep/convert_vqa_datasets.py process_vqa --dataset_dir data/datasets/vqa2 --version 2`

- train set (82783 images, ~ 10 minutes): `python cli/prep/vqa2_to_lmdb.py write_db --image_dir data/datasets/vqa2/train2014`
- val set (40504 images, ~ 5 minutes): `python cli/prep/vqa2_to_lmdb.py write_db --image_dir data/datasets/vqa2/val2014`
- test set (81434 images, ~ 5 minutes): `python cli/prep/vqa2_to_lmdb.py write_db --image_dir data/datasets/vqa2/test2015`

#### NLVR2

Download:

- download nvlr2 images: `python cli/prep/dl_data.py download_nlvr2_images --base_url https://xxx/NLVR2/ --out_dir data/datasets/nlvr2`, see https://github.com/lil-lab/nlvr/blob/master/nlvr2/README.md#direct-image-download to get image url, and replace the `xxx` with the base url

- download nvlr2 dataset: `python cli/prep/dl_data.py download_nlvr2_data --out_dir data/datasets/nlvr2 --download_balance true --download_unbalance true`

Convert to lmdb format:

- train set (103170 images, ~ 4 minutes): `python cli/prep/vqa2_to_lmdb.py write_db --image_dir data/datasets/nlvr2/images/train --output_file data/datasets/nlvr2/train.lmdb --path_pattern "*/*"`
- dev set (8102 images, ~ 15 seconds): `python cli/prep/vqa2_to_lmdb.py write_db --image_dir data/datasets/nlvr2/dev`
- test set (8082 images, ~ 15 seconds): `python cli/prep/vqa2_to_lmdb.py write_db --image_dir data/datasets/nlvr2/test1`


#### SNLI-VE

use the following snippet to download flickr30 images dataset:
(you can follow instructions [here](https://github.com/JovianML/opendatasets/blob/master/README.md#kaggle-credentials) to get your kaggle api key)

```python
# pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset", "./download_dir")
```

and move `flickr30k_images` to `data/datasets/flickr30k `

```bash
python cli/prep/convert_flickr_coco.py write_flickr30k_db data/datasets/flickr30k/flickr30k_images

```

Then follow [instructions](https://github.com/necla-ml/SNLI-VE#snli-ve-creation) from snli-ve repo to get train, dev, test splits, and put them in `data/datasets/snli-ve`

### Retrieval

download [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

- flickr30:

```text
data/datasets/flickr30k
├── flickr30k_images
│   ├── 1000092795.jpg
|   └── ...
└── dataset_flickr30k.json
```

`python cli/prep/convert_flickr_coco.py convert_dataset data/datasets/flickr30k/dataset_flickr30k.json data/datasets/flickr30k/flickr30k.jsonl`

- mscoco:

for images, just use vqa `coco-trainval2014.lmdb`

`python cli/prep/convert_flickr_coco.py convert_dataset data/datasets/mscoco/dataset_coco.json data/datasets/mscoco/mscoco.jsonl`

