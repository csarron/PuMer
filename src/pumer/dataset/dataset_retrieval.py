import random

import torch

from . import BaseDataset


class RetrievalTrainDataset(BaseDataset):
    def __init__(
        self,
        dataset_file,
        image_db_path,
        tokenizer,
        max_text_len=40,
        image_size=384,
        use_randaug=False,
        is_clip=False,
        num_negative_samples=15,
    ):
        super().__init__(
            dataset_file,
            image_db_path,
            tokenizer,
            max_text_len,
            image_size,
            use_randaug,
            is_clip,
        )
        self.num_negative_samples = num_negative_samples

    def _get_one(self, captions):
        return random.choice(captions)

    def _sample_negative_texts(self, index):
        total = len(self)
        all_indices = [i for i in range(total) if i != index]
        random.shuffle(all_indices)
        sample_indices = all_indices[: self.num_negative_samples]
        sample_captions = []
        for sample_idx in sample_indices:
            captions = self.data[sample_idx]["captions"]
            caption = self._get_one(captions)
            sample_captions.append(caption)
        return sample_captions

    def __getitem__(self, index):
        item = self.data[index]
        img_id = item["img_id"]
        item_id = item["item_id"]
        captions = item["captions"]
        caption = self._get_one(captions)
        # during training, for each text, sample random self.num_negative_samples negative text,
        negative_captions = self._sample_negative_texts(index)
        captions = [caption] + negative_captions

        inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        # load processed img features
        img_bytes = self._load_image_bytes(img_id)
        img_feat = self.image_bytes_to_tensor(img_bytes)
        c, h, w = img_feat.shape
        # repeat image values for self.num_negative_samples + 1 times
        img_feat = img_feat.unsqueeze(0).expand(len(captions), c, h, w)
        data = {
            "item_ids": item_id,
            "text_ids": input_ids,
            "text_masks": attention_mask,
            "pixel_values": img_feat,
        }
        return data


class RetrievalInferenceDataset(BaseDataset):
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
        super().__init__(
            dataset_file,
            image_db_path,
            tokenizer,
            max_text_len,
            image_size,
            use_randaug,
            is_clip,
        )

        self.texts = []
        self.images = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, item in enumerate(self.data):
            self.images.append(item["img_id"])
            self.img2txt[img_id] = []
            for caption in item["captions"]:
                self.texts.append(caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        text_data_size = len(self.texts)
        image_idx = index // text_data_size
        text_idx = index % text_data_size

        ori_image_idx = self.txt2img[text_idx]  # paired image idx
        # ori_text_idxs = self.img2txt[image_idx] # paired text idx
        caption = self.texts[text_idx]
        img_id = self.images[image_idx]

        inputs = self.tokenizer(
            caption,
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

        if image_idx > 0:
            ori_image_idx = -1

        if text_idx > 0:
            image_idx = -1

        data = {
            "item_ids": torch.tensor(index, dtype=torch.long),
            "text_ids": input_ids,
            "text_masks": attention_mask,
            "pixel_values": img_feat,
            "text_image_idx": torch.tensor(ori_image_idx, dtype=torch.long),
            "image_idx": torch.tensor(image_idx, dtype=torch.long),
        }

        return data

    def __len__(self):
        return len(self.images) * len(self.texts)
