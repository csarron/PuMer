import glob
import json
import logging
import math
import os
import random
import sys
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Generic, Iterator, List, Mapping, Optional, Sequence, Sized, TypeVar

import torch
from loguru import logger
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
)

from ..utils.image_utils import get_image_transforms, image_bytes_to_pil


class PretrainDataCollator:
    def __init__(self, data_collator) -> None:
        self.data_collator = data_collator

    def __call__(self, features):
        keys = set(features[0].keys())
        dict_batch = {}
        for k in keys:
            v = [dic[k] for dic in features]
            if isinstance(v[0], torch.Tensor):
                dict_batch[k] = torch.stack(v)
            else:
                dict_batch[k] = v
        new_features = self.data_collator(features)
        dict_batch.update(new_features)
        if "__key__" in dict_batch:  # pop webdataset extra __key__
            dict_batch.pop("__key__")
        return dict_batch


class HardNegativeBatchRandomSampler(Sampler):

    r"""
    For a batch, first half of examples are random positives and the other half are corresponding negatives (only for mask contrastive learning, still valid examples for the task)
    batch_size must be multiples of (num_negatives_per_example + 1)
    num_negatives_per_example is default to 0, meaning not using any negatives
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        num_negatives_per_example: int = 0,
        negatives_map: Mapping[int, List[int]] = None,
        drop_last: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        assert (
            batch_size % (num_negatives_per_example + 1) == 0
        ), f"{batch_size=} must be multiples of {(num_negatives_per_example + 1)=}"
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.negatives_map = negatives_map or getattr(data_source, "negatives_map", None)
        self.num_negatives_per_example = num_negatives_per_example

        self.data_source = data_source
        self.generator = generator
        self._num_samples = num_samples
        self.shrink_size = self.num_samples // (self.num_negatives_per_example + 1)  # avoid dataset repeat

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def sample_n(self, sample_list, n):
        n_samples = len(sample_list)
        if n > n_samples:
            full_list = sample_list + random.sample(range(self.num_samples), n - n_samples)
        else:
            full_list = sample_list
        return random.sample(full_list, n)

    def __iter__(self) -> Iterator[List[int]]:

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        candidates = torch.randperm(self.num_samples, generator=generator).tolist()
        batch = []
        for idx in candidates[: self.shrink_size]:
            batch.append(idx)
            # sample one num_negatives_per_example negative indices from idx's negative banks, if not enough, sample random indices from the dataset
            if self.num_negatives_per_example > 0 and self.negatives_map is not None:
                neg_bank = self.negatives_map[idx]
                for neg_idx in self.sample_n(neg_bank, n=self.num_negatives_per_example):
                    batch.append(neg_idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def get_data_collator(cfg):
    if cfg.task.name == "pretrain":
        tokenizer = AutoTokenizer.from_pretrained(cfg.task.tokenizer)
        mlm_probability = cfg.task.mlm_probability
        if cfg.task.whole_word_masking:
            mlm_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=mlm_probability)
        else:
            mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
        data_collator = PretrainDataCollator(mlm_collator)
    else:
        data_collator = None

    return data_collator


class PredictionWriter(object):
    def __init__(self, cfg, dataset_size=None):
        self.pred_file = Path(cfg.task.predictions_output_file)
        self.pred_file.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_size = dataset_size
        self.all_preds = []

    @abstractmethod
    def process(self, ids, predictions, extra=None):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    def finish(self):
        self.save()
        logger.info(f"predictions saved to {self.pred_file}!")


class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_file,
        image_db_path,
        tokenizer,
        max_text_len=40,
        image_size=384,
        use_randaug=False,
        is_clip=False,
    ):
        super().__init__()
        import lmdb

        self.data = [json.loads(line) for line in open(dataset_file)]
        self.db = lmdb.open(image_db_path, subdir=False, readonly=True, lock=False)
        self.init_common(tokenizer, max_text_len, image_size, use_randaug, is_clip)

    def build_negatives_map(self, image_key):
        img2q = defaultdict(list)
        for i, di in enumerate(self.data):
            img2q[di[image_key]].append(i)
        self.negatives_map = defaultdict(list)
        for indices in img2q.values():
            for idx in indices:
                for neg in indices:
                    if idx != neg:
                        self.negatives_map[idx].append(neg)

    def init_common(
        self,
        tokenizer,
        max_text_len=40,
        image_size=384,
        use_randaug=False,
        is_clip=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_text_len = max_text_len
        self.image_transforms = get_image_transforms(image_size, use_randaug, is_clip)

    def _load_image_bytes(self, img_id):
        with self.db.begin(write=False) as txn:
            img_bytes = txn.get(img_id.encode())
        return img_bytes

    def image_bytes_to_tensor(self, img_bytes, transform=True):
        image = image_bytes_to_pil(img_bytes)
        if transform:
            image = self.image_transforms(image)
        return image

    def __getitem__(self, index):
        pass

    def _process_label(self, item):
        pass

    def __len__(self):
        return len(self.data)
