import math
import shutil
from pathlib import Path


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def remove_path(path, ignore_errors=True):
    path = Path(path)
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=ignore_errors)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=ignore_errors)
    else:
        import warnings

        warnings.warn(f"unknown path type: {path}, not removed!")


def print_ckpt(ckpt_dict):
    params = 0
    for k, v in ckpt_dict.items():
        print(k, v.shape)
        params += math.prod(v.shape)
    print(params / 1e6, params * 4 / 1024 / 1024)
