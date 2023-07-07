import csv
from collections import defaultdict

import torch

from . import BaseDataset, PredictionWriter


class Nlvr2Dataset(BaseDataset):
    def __init__(
        self, dataset_file, image_db_path, tokenizer, max_text_len=40, image_size=384, use_randaug=False, is_clip=False
    ):
        super().__init__(dataset_file, image_db_path, tokenizer, max_text_len, image_size, use_randaug, is_clip)
        self.build_negatives_map()

    def build_negatives_map(self):
        text2item = defaultdict(list)
        for i, di in enumerate(self.data):
            item_id = di["identifier"]
            set_id = item_id.split("-")[1]
            text2item[set_id].append(i)
        self.negatives_map = defaultdict(list)
        for indices in text2item.values():
            for idx in indices:
                for neg in indices:
                    if idx != neg:
                        self.negatives_map[idx].append(neg)

    def __getitem__(self, index):
        data_item = self.data[index]
        item_id = data_item["identifier"]
        id_splits = item_id.split("-")
        num_id = int("".join(id_splits[1:]))  # accelerate and pytorch only accepts numeric ids
        # format see: https://github.com/lil-lab/nlvr/blob/master/nlvr2/README.md#json-files
        # identifier: split-set_id-pair_id-sentence_id, dev-850-0-0
        # img1: split-set_id-pair_id-img0.png, dev-850-0-img0
        # img2: split-set_id-pair_id-img1.png, dev-850-0-img1
        img1_id = "-".join(id_splits[:-1] + ["img0"])
        img2_id = "-".join(id_splits[:-1] + ["img1"])
        sentence = data_item["sentence"]
        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        # load processed img features
        img1_bytes = self._load_image_bytes(img1_id)
        img1_feat = self.image_bytes_to_tensor(img1_bytes)
        img2_bytes = self._load_image_bytes(img2_id)
        img2_feat = self.image_bytes_to_tensor(img2_bytes)

        labels = self._process_label(data_item)
        return {
            "item_ids": num_id,
            "text_ids": input_ids,
            "text_masks": attention_mask,
            "pixel_values1": img1_feat,
            "pixel_values2": img2_feat,
            "labels": labels,
            "extra": id_splits[0],
        }

    def _process_label(self, data_item):
        if "label" in data_item:
            label = data_item["label"]
            if label == "True":
                label = 1
            elif label == "False":
                label = 0
            else:
                raise ValueError(f"invalid label: {label}")
        return torch.tensor(label)  # default long tensor


class Nlvr2PredictionWriter(PredictionWriter):
    def __init__(self, cfg, dataset_size=None):
        super().__init__(cfg, dataset_size)

    def process(self, ids, predictions, extra=None):
        idx2label = ["False", "True"]
        # map item_id back to original, see lv.dataset.dataset_nlvr2.Nlvr2Dataset for how the num_id is generated, this is a reverse process: 8500 -> dev-850-0-0
        split = extra[0]  # stored the split info from dataset
        for item_id, prediction in zip(ids, predictions):
            predicted_label = idx2label[prediction]
            part1 = str(item_id // 100)
            id_str = str(item_id)
            part2 = id_str[-2] if len(id_str) > 1 else "0"  # should be between 0 and 3
            part3 = id_str[-1]  # should be 0 or 1
            orig_id = "-".join([split, part1, part2, part3])
            self.all_preds.append((orig_id, predicted_label))

    def save(self):
        # self.csv_handle.close()
        all_preds = self.all_preds[: self.dataset_size]
        with open(self.pred_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            for item_id, predicted_label in all_preds:
                writer.writerow([item_id, predicted_label])


class Nlvr2VlueDataset(BaseDataset):
    def __getitem__(self, index):
        data_item = self.data[index]
        img1_id = data_item["images"][0]
        img2_id = data_item["images"][1]
        sentence = data_item["sentence"]
        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        # load processed img features
        img1_bytes = self._load_image_bytes(img1_id)
        img1_feat = self.image_bytes_to_tensor(img1_bytes)
        img2_bytes = self._load_image_bytes(img2_id)
        img2_feat = self.image_bytes_to_tensor(img2_bytes)

        labels = self._process_label(data_item)
        return {
            "item_ids": index,
            "text_ids": input_ids,
            "text_masks": attention_mask,
            "pixel_values1": img1_feat,
            "pixel_values2": img2_feat,
            "labels": labels,
        }

    def _process_label(self, data_item):
        if "label" in data_item:
            label = data_item["label"]
            if label == "True":
                label = 1
            elif label == "False":
                label = 0
            else:
                raise ValueError(f"invalid label: {label}")
        return torch.tensor(label)  # default long tensor
