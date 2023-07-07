# %load_ext autoreload
# %autoreload 2

import json
import math

import fire
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer

from pumer.model import get_model, get_model_config
from pumer.model.meter.vit_meter import adapt_position_encoding
from pumer.utils.image_utils import get_image_transforms


# os.getcwd()
# !pip install pytorch-lightning
def print_ckpt(orig_ckpt):
    ckpt_dict = torch.load(orig_ckpt, map_location="cpu")
    params = 0
    for k, v in ckpt_dict.items():
        print(k, v.shape)
        params += math.prod(v.shape)
    print(params / 1e6, params * 4 / 1024 / 1024)


def convert_meter_ckpt(meter_model, ckpt_weights):
    new_ckpt = {}
    for k, v in meter_model.state_dict().items():
        old_key = k.replace("meter.", "")
        # if old_key not in meter_ckpt["state_dict"]:
        if old_key.startswith("vit_model"):
            old_key = k.replace("meter.vit_model", "vit_model.visual")
            # print(f"visual key: {old_key}")
        new_ckpt[k] = ckpt_weights[old_key]
    return new_ckpt


def convert_meter(
    task,
    meter_orig_weights,
    out_dir="data/ckpt/converted",
    image_size=288,
    token_types=2,
):
    allowd_tasks = ("pretrain", "vqa2", "nlvr2", "snli_ve", "irtr_flickr30k", "irtr_mscoco")
    assert task in allowd_tasks, f"{task} must be in {allowd_tasks}"
    if task.startswith("irtr"):
        task = "irtr"
    meter_cfg_class = get_model_config("meter")
    meter_class = get_model("meter", task)
    meter_cfg = meter_cfg_class(image_size=image_size, token_types=token_types)
    meter_model = meter_class(meter_cfg).eval()
    meter_model_weights = convert_meter_ckpt(meter_model, meter_orig_weights)
    meter_model.load_state_dict(meter_model_weights)
    meter_model.save_pretrained(out_dir)


def convert_meter_pretrain_nlvr2(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_pretrain.ckpt",
    out_dir="data/ckpt/converted/meter_pretrain_nlvr2",
    image_size=288,
):
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 288:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_orig_weights = meter_new_weights
    else:
        meter_orig_weights = meter_orig_ckpt["state_dict"]

    tte = meter_orig_weights["token_type_embeddings.weight"]
    new_tte = torch.concat([tte, torch.unsqueeze(tte[1, :], 0)], 0)
    meter_orig_weights["token_type_embeddings.weight"] = new_tte
    convert_meter("pretrain", meter_orig_weights, out_dir, image_size, token_types=3)


def convert_meter_pretrain(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_pretrain.ckpt",
    out_dir="data/ckpt/converted/meter_pretrain",
    image_size=288,
):
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 288:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_orig_weights = meter_new_weights
    else:
        meter_orig_weights = meter_orig_ckpt["state_dict"]
    convert_meter("pretrain", meter_orig_weights, out_dir, image_size)


def convert_meter_vqa2(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_vqa.ckpt",
    out_dir="data/ckpt/converted/meter_vqa2",
    image_size=576,
):
    allowed_sizes = (576, 288, 384)
    assert image_size in allowed_sizes, f"{image_size} must be in {allowed_sizes}"
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 576:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_orig_weights = meter_new_weights
    else:
        meter_orig_weights = meter_orig_ckpt["state_dict"]
    convert_meter("vqa2", meter_orig_weights, out_dir, image_size)


def convert_meter_nlvr2(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_nlvr2.ckpt",
    out_dir="data/ckpt/converted/meter_nlvr2",
    image_size=288,
):
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 288:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_orig_weights = meter_new_weights
    else:
        meter_orig_weights = meter_orig_ckpt["state_dict"]
    convert_meter("nlvr2", meter_orig_weights, out_dir, image_size, 3)


def convert_meter_irtr_flickr30k(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_flickr.ckpt",
    out_dir="data/ckpt/converted/meter-irtr-flickr30k",
    image_size=384,
):
    allowed_sizes = (576, 288, 384)
    assert image_size in allowed_sizes, f"{image_size} must be in {allowed_sizes}"
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 576:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_orig_weights = meter_new_weights
    else:
        meter_orig_weights = meter_orig_ckpt["state_dict"]
    convert_meter("irtr_flickr30k", meter_orig_weights, out_dir, image_size)


def convert_meter_irtr_mscoco(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_coco.ckpt",
    out_dir="data/ckpt/converted/meter-irtr-mscoco",
    image_size=384,
):
    allowed_sizes = (576, 288, 384)
    assert image_size in allowed_sizes, f"{image_size} must be in {allowed_sizes}"
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 576:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_orig_weights = meter_new_weights
    else:
        meter_orig_weights = meter_orig_ckpt["state_dict"]
    convert_meter("irtr_mscoco", meter_orig_weights, out_dir, image_size)


def convert_meter_irtr(
    orig_ckpt="data/ckpt/original/meter_clip16_288_roberta_pretrain.ckpt",
    out_dir="data/ckpt/converted/meter-irtr-pretrained",
    image_size=384,
):
    allowed_sizes = (576, 288, 384)
    assert image_size in allowed_sizes, f"{image_size} must be in {allowed_sizes}"
    meter_orig_ckpt = torch.load(orig_ckpt, map_location="cpu")
    if image_size != 576:
        # interpolate embeddings
        meter_new_weights = adapt_position_encoding(
            meter_orig_ckpt["state_dict"], 16, image_size
        )  # change 16 to 32 for CLIP-ViT-B32
        out_dir += f"-{image_size}"
        meter_weights = meter_new_weights
    else:
        meter_weights = meter_orig_ckpt["state_dict"]

    # adapt rank_output, use itm_score.fc weight
    meter_weights["rank_output.weight"] = meter_weights["itm_score.fc.weight"][
        1:,
    ]
    meter_weights["rank_output.bias"] = meter_weights["itm_score.fc.bias"][1:]
    convert_meter("irtr_mscoco", meter_weights, out_dir, image_size)


def convert_vilt_ckpt(vilt_model, ckpt_weights, task):
    # for pretrain
    new_ckpt = {}
    if task == "pretrain":
        for k, v in vilt_model.state_dict().items():
            new_ckpt[k] = ckpt_weights[k.replace("vilt.", "")]

    # for vqa2
    elif task == "vqa2":
        for k, v in ckpt_weights.items():
            # print(k, v.shape)
            new_k = k if k.startswith("vqa_classifier") else "vilt." + k
            new_ckpt[new_k] = v

    # for nlvr2
    elif task == "nlvr2":
        for k, v in ckpt_weights.items():
            # print(k, v.shape)
            new_k = k if k.startswith("nlvr2_classifier") else "vilt." + k
            new_ckpt[new_k] = v
    elif task == "irtr":
        for k, v in ckpt_weights.items():
            # print(k, v.shape)
            if "mlm_score" in k:
                continue
            new_k = k if k.startswith("rank_output") or k.startswith("itm_score") else "vilt." + k
            new_ckpt[new_k] = v
    else:
        raise ValueError(f"{task} cktp conversion not supported for vilt")
    return new_ckpt


def convert_vilt(
    task,
    vilt_orig_weights,
    out_dir="data/ckpt/converted",
    token_types=2,
):
    allowd_tasks = ("pretrain", "vqa2", "nlvr2", "snli_ve", "irtr_flickr30k", "irtr_mscoco")
    assert task in allowd_tasks, f"{task} must be in {allowd_tasks}"
    vilt_cfg_class = get_model_config("vilt")
    if task.startswith("irtr"):
        task = "irtr"
    vilt_class = get_model("vilt", task)
    cfg = vilt_cfg_class(token_types=token_types)
    vilt_model = vilt_class(cfg).eval()
    vilt_model_weights = convert_vilt_ckpt(vilt_model, vilt_orig_weights["state_dict"], task)
    vilt_model.load_state_dict(vilt_model_weights)
    vilt_model.save_pretrained(out_dir)


def convert_vilt_pretrain_nlvr2(
    orig_ckpt="data/ckpt/original/vilt_200k_mlm_itm.ckpt", out_dir="data/ckpt/converted/vilt_pretrain_nlvr2"
):
    vilt_orig_weights = torch.load(orig_ckpt, map_location="cpu")
    tte = vilt_orig_weights["state_dict"]["token_type_embeddings.weight"]
    new_tte = torch.concat([tte, torch.unsqueeze(tte[1, :], 0)], 0)
    vilt_orig_weights["state_dict"]["token_type_embeddings.weight"] = new_tte
    convert_vilt("pretrain", vilt_orig_weights, out_dir, token_types=3)


def convert_vilt_pretrain(
    orig_ckpt="data/ckpt/original/vilt_200k_mlm_itm.ckpt", out_dir="data/ckpt/converted/vilt_pretrain"
):
    vilt_orig_weights = torch.load(orig_ckpt, map_location="cpu")
    convert_vilt("pretrain", vilt_orig_weights, out_dir)


def convert_vilt_vqa2(orig_ckpt="data/ckpt/original/vilt_vqa.ckpt", out_dir="data/ckpt/converted/vilt_vqa2"):
    vilt_orig_weights = torch.load(orig_ckpt, map_location="cpu")
    convert_vilt("vqa2", vilt_orig_weights, out_dir)


def convert_vilt_nlvr2(orig_ckpt="data/ckpt/original/vilt_nlvr2.ckpt", out_dir="data/ckpt/converted/vilt_nlvr2"):
    vilt_orig_weights = torch.load(orig_ckpt, map_location="cpu")
    convert_vilt("nlvr2", vilt_orig_weights, out_dir, 3)


def convert_vilt_irtr(
    orig_ckpt="data/ckpt/original/vilt_200k_mlm_itm.ckpt", out_dir="data/ckpt/converted/vilt-irtr-pretrained"
):
    vilt_weights = torch.load(orig_ckpt, map_location="cpu")
    # adapt rank_output, use itm_score.fc weight
    vilt_weights["state_dict"]["rank_output.weight"] = vilt_weights["state_dict"]["itm_score.fc.weight"][
        1:,
    ]
    vilt_weights["state_dict"]["rank_output.bias"] = vilt_weights["state_dict"]["itm_score.fc.bias"][1:]
    convert_vilt("irtr_mscoco", vilt_weights, out_dir)


def convert_vilt_irtr_mscoco(
    orig_ckpt="data/ckpt/original/vilt_irtr_coco.ckpt", out_dir="data/ckpt/converted/vilt-irtr-mscoco"
):
    vilt_orig_weights = torch.load(orig_ckpt, map_location="cpu")
    convert_vilt("irtr_mscoco", vilt_orig_weights, out_dir)


def convert_vilt_irtr_flickr30k(
    orig_ckpt="data/ckpt/original/vilt_irtr_f30k.ckpt", out_dir="data/ckpt/converted/vilt-irtr-flickr30k"
):
    vilt_orig_weights = torch.load(orig_ckpt, map_location="cpu")
    convert_vilt("irtr_flickr30k", vilt_orig_weights, out_dir)


def get_image(image_url, image_size=384):
    im = Image.open(requests.get(image_url, stream=True).raw)
    img = get_image_transforms(image_size)(im)
    pixel_values = img.unsqueeze(0)
    return pixel_values


def get_text_inputs(text, tokenizer="roberta-base"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    text_outputs = tokenizer(
        text,
        padding="max_length",
        max_length=40,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_outputs.input_ids
    attn_masks = text_outputs.attention_mask
    return input_ids, attn_masks


def infer_vqa2(vqa_model, image_size, tokenizer, label2ans_file):
    img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    pixel_values = get_image(img_url, image_size)

    text = "How many cats are there?"
    input_ids, attn_masks = get_text_inputs(text, tokenizer)

    outputs = vqa_model(text_ids=input_ids, text_masks=attn_masks, pixel_values=pixel_values, return_dict=True)

    pred_idx = int(outputs["logits"].argmax(-1))
    vqa_label2ans = json.load(open(label2ans_file))  # label index
    print(pred_idx, vqa_label2ans[pred_idx])
    # should be 17, 2


def infer_nlvr2(nvlr2_model, image_size, tokenizer):
    ## examples from https://lil.nlp.cornell.edu/nlvr/

    # 'https://lil.nlp.cornell.edu/nlvr/exs/acorns_1.jpg'
    # https://lil.nlp.cornell.edu/nlvr/exs/acorns_6.jpg
    # One image shows exactly two brown acorns in back-to-back caps on green foliage.
    # False

    nlvr2_img0_url = "https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg"
    nlvr2_img1_url = "https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg"
    sentence = "The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing."

    pixel_values1 = get_image(nlvr2_img0_url, image_size)
    pixel_values2 = get_image(nlvr2_img1_url, image_size)
    input_ids, attn_masks = get_text_inputs(sentence, tokenizer)
    outputs = nvlr2_model(
        text_ids=input_ids,
        text_masks=attn_masks,
        pixel_values1=pixel_values1,
        pixel_values2=pixel_values2,
        return_dict=True,
    )
    pred_idx = int(outputs["logits"].argmax(-1))
    print(pred_idx, ["False", "True"][pred_idx])
    # should be 1, True


def test_infer_vilt_vqa2(
    model_path="data/ckpt/converted/vilt_vqa2", label2ans_file="data/datasets/vqa2/vilt_label2ans.json"
):
    cfg_class = get_model_config("vilt")
    cfg = cfg_class()
    model_class = get_model("vilt", "vqa2")
    ## modify cfg before model loads checkpoint
    # cfg.update({"a": 1, "b": 2, "max_text_len": 25})
    # model = model_class.from_pretrained(model_path, config=cfg)
    model = model_class.from_pretrained(model_path)
    # modify cfg after model loads checkpoint
    # model.config.update({"a": 1, "b": 2, "max_text_len": 25})
    model.eval()
    print("initialized vilt vqa2")
    infer_vqa2(model, 384, cfg.tokenizer, label2ans_file)


def test_infer_meter_vqa2(
    model_path="data/ckpt/converted/meter_vqa2", label2ans_file="data/datasets/vqa2/vilt_label2ans.json"
):
    cfg_class = get_model_config("meter")
    cfg = cfg_class()
    model_class = get_model("meter", "vqa2")
    model = model_class.from_pretrained(model_path)
    model.eval()
    print("initialized meter vqa2")
    infer_vqa2(model, 576, cfg.tokenizer, label2ans_file)


def test_infer_vilt_nlvr2(model_path="data/ckpt/converted/vilt_nlvr2"):
    cfg_class = get_model_config("vilt")
    cfg = cfg_class()
    model_class = get_model("vilt", "nlvr2")
    model = model_class.from_pretrained(model_path, return_dict=True)
    model.eval()
    print("initialized vilt nlvr2")
    infer_nlvr2(model, 384, cfg.tokenizer)


def test_infer_meter_nlvr2(model_path="data/ckpt/converted/meter_nlvr2"):
    cfg_class = get_model_config("meter")
    cfg = cfg_class()
    model_class = get_model("meter", "nlvr2")
    model = model_class.from_pretrained(model_path, return_dict=True)
    model.eval()
    print("initialized meter nlvr2")
    infer_nlvr2(model, 288, cfg.tokenizer)


if __name__ == "__main__":
    fire.Fire()
