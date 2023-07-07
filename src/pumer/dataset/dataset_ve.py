import torch

from . import BaseDataset, PredictionWriter


class VeDataset(BaseDataset):
    def __init__(
        self, dataset_file, image_db_path, tokenizer, max_text_len=40, image_size=384, use_randaug=False, is_clip=False
    ):
        super().__init__(dataset_file, image_db_path, tokenizer, max_text_len, image_size, use_randaug, is_clip)
        self.build_negatives_map("Flickr30K_ID")

    def __getitem__(self, index):
        data_item = self.data[index]
        item_id = data_item["pairID"]
        img_id = data_item["Flickr30K_ID"]
        sentence = data_item["sentence2"]

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
        img_bytes = self._load_image_bytes(img_id)
        img_feat = self.image_bytes_to_tensor(img_bytes)

        labels = self._process_label(data_item)
        return {
            "item_ids": item_id,
            "text_ids": input_ids,
            "text_masks": attention_mask,
            "pixel_values": img_feat,
            "labels": labels,
        }

    def _process_label(self, data_item):
        if "gold_label" in data_item:
            label = data_item["gold_label"]
            if label == "contradiction":
                label = 0
            elif label == "neutral":
                label = 1
            elif label == "entailment":
                label = 2
            else:
                raise ValueError(f"invalid label: {label}")
        return torch.tensor(label)  # default long tensor


class VePredictionWriter(PredictionWriter):
    def __init__(self, cfg, dataset_size=None):
        super().__init__(cfg, dataset_size)

    def process(self, ids, predictions, extra=None):
        # idx2label = ["contradiction", "neutral", "entailment"]
        pass

    def save(self):
        pass
