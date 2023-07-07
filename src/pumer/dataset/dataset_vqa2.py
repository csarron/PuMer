import json
from collections import defaultdict
from pathlib import Path

import torch

from ..utils.common import get_neighbor_indices
from . import BaseDataset, PredictionWriter


class Vqa2Dataset(BaseDataset):
    def __init__(
        self,
        dataset_file,
        image_db_path,
        tokenizer,
        ans2label_file,
        max_text_len=40,
        image_size=384,
        use_randaug=False,
        is_clip=False,
        gradcam_file=None,
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
        self.ans2label = json.load(open(ans2label_file))
        self.num_answers = len(self.ans2label)
        self.gradcam_masks = None
        self.build_negatives_map("img_id")
        if gradcam_file and Path(gradcam_file).exists():
            self.gradcam_masks = dict()
            # h = w = 12
            # idx2neigh = dict()
            # for layer in [9, 6, 3]:
            #     r = 4 - layer // 3  # r is 1, 2, 3
            h = w = 24
            idx2neigh = dict()
            for layer in [6, 4, 2]:
                r = 4 - layer // 2  # r is 1, 2, 3
                idx2neighbors = dict()
                for idx, neighbor_indices in get_neighbor_indices(h, w, r):
                    idx2neighbors[idx] = neighbor_indices
                idx2neigh[layer] = idx2neighbors

            self.all_gradcam_masks = defaultdict(dict)
            for line in open(gradcam_file):
                item = json.loads(line)
                self.gradcam_masks[item["qid"]] = item["gradcam_masks"]
                self.gradcam_total = item["total_gradcams"]
                # h = w = int(math.sqrt(self.gradcam_total))

                # THIS will take long time ~ 2 minutes
                for layer, idx2neighbors in idx2neigh.items():
                    gc_indices = set(item["gradcam_masks"])
                    for idx in item["gradcam_masks"]:
                        neighbors = idx2neighbors.get(idx)
                        for ni in neighbors:
                            gc_indices.add(ni)

                    self.all_gradcam_masks[layer][item["qid"]] = list(gc_indices)

    def __getitem__(self, index):
        qa_item = self.data[index]
        img_id = qa_item["img_id"]
        q_id = qa_item["qid"]
        question = qa_item["question"]
        inputs = self.tokenizer(
            question,
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
        labels = self._process_label(qa_item)
        data = {
            "item_ids": q_id,
            "text_ids": input_ids,
            "text_masks": attention_mask,
            "pixel_values": img_feat,
            "labels": labels,
        }
        if self.gradcam_masks is not None:
            gradcam_masks = self.gradcam_masks[q_id]
            gradcam_tensor = torch.zeros(self.gradcam_total)
            gradcam_tensor[gradcam_masks] = 1.0
            data["gradcam_masks"] = gradcam_tensor

            for layer, gradcams in self.all_gradcam_masks.items():
                gd_mask = gradcams[q_id]
                gradcam_tensor = torch.zeros(self.gradcam_total)
                gradcam_tensor[gd_mask] = 1.0
                data[f"gradcam_masks_{layer}"] = gradcam_tensor
        return data

    def _process_label(self, qa_item):
        target = torch.zeros(self.num_answers)
        if "label" in qa_item:
            label = qa_item["label"]
            for ans, score in label.items():
                if ans not in self.ans2label:
                    continue
                target[self.ans2label[ans]] = score
        return target


class Vqa2PredictionWriter(PredictionWriter):
    def __init__(self, cfg, dataset_size=None):
        super().__init__(cfg, dataset_size)
        self.vqa_label2ans = json.load(open(cfg.task.label2ans_file))

    def process(self, ids, predictions, extra=None):
        answers = [self.vqa_label2ans[i] for i in predictions]
        for item_id, answer in zip(ids, answers):
            item = {
                "question_id": item_id,
                "answer": answer,
            }
            self.all_preds.append(item)

    def save(self):
        all_preds = self.all_preds[: self.dataset_size]
        json.dump(all_preds, open(self.pred_file, "w"))
