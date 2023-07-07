import json
from collections import defaultdict
from pathlib import Path

import fire
import lmdb
from tqdm.auto import tqdm


def convert_dataset(dataset_file, out_file=None):
    dataset_file = Path(dataset_file)
    if out_file is None:
        out_file = dataset_file.with_suffix(".jsonl")
    else:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    iid2captions = defaultdict(list)
    iid2split = defaultdict(set)

    items = json.load(open(dataset_file))["images"]
    for item in tqdm(items):
        filename = item["filename"]
        iid2split[item["split"]].add(filename)
        for c in item["sentences"]:
            iid2captions[filename].append(c["raw"])

    for split, split_examples in iid2split.items():
        with open(out_file.parent / f"{out_file.stem}-{split}{out_file.suffix}", "w") as f:
            for img_id in split_examples:
                captions = iid2captions[img_id]
                img_id = Path(img_id).stem
                try:
                    item_id = int(img_id)
                except:
                    assert "COCO" in img_id
                    item_id = int(img_id.split("_")[-1])
                data = {
                    "item_id": item_id,
                    "captions": captions,
                    "img_id": img_id,
                }
                f.write(json.dumps(data, ensure_ascii=False))
                f.write("\n")


def write_flickr30k_db(image_dir, output_file=None, map_size=20 * 1024 * 1024 * 1024, batch_size=1024):
    image_path = Path(image_dir)
    if output_file is None:
        output_file = image_path.with_suffix(".lmdb")

    db = lmdb.open(str(output_file), map_size=map_size, subdir=False)
    with db.begin(write=True) as txn:
        buf = []
        for img_file in tqdm(image_path.glob("*.jpg")):
            with open(img_file, "rb") as f:
                img_data = f.read()
            buf.append((str(img_file.stem).encode("utf-8"), img_data))
            if len(buf) == batch_size:
                txn.cursor().putmulti(buf)
        if buf:
            txn.cursor().putmulti(buf)


if __name__ == "__main__":
    fire.Fire()
