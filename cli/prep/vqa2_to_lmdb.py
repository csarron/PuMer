from pathlib import Path

import fire
import lmdb
from tqdm.auto import tqdm


def write_db(image_dir, output_file=None, map_size=20 * 1024 * 1024 * 1024, batch_size=1024, path_pattern="*"):
    image_path = Path(image_dir)
    if output_file is None:
        output_file = image_path.with_suffix(".lmdb")
    db = lmdb.open(str(output_file), map_size=map_size, subdir=False)
    with db.begin(write=True) as txn:
        buf = []
        for img_file in tqdm(image_path.glob(path_pattern)):
            with open(img_file, "rb") as f:
                img_data = f.read()
            buf.append((str(img_file.stem).encode("utf-8"), img_data))
            if len(buf) == batch_size:
                txn.cursor().putmulti(buf)
        if buf:
            txn.cursor().putmulti(buf)


def write_vlue_db(
    image_dir, output_file=None, map_size=20 * 1024 * 1024 * 1024, batch_size=1024, path_pattern="**/*.jpg"
):
    image_path = Path(image_dir)
    if output_file is None:
        output_file = image_path.with_suffix(".lmdb")
    db = lmdb.open(str(output_file), map_size=map_size, subdir=False)
    with db.begin(write=True) as txn:
        buf = []
        for img_file in tqdm(image_path.glob(path_pattern)):
            with open(img_file, "rb") as f:
                img_data = f.read()
            buf.append((str(img_file.relative_to(image_path)).encode("utf-8"), img_data))
            if len(buf) == batch_size:
                txn.cursor().putmulti(buf)
        if buf:
            txn.cursor().putmulti(buf)


def main():
    fire.Fire()


if __name__ == "__main__":
    main()
