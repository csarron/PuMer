# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.8.12 64-bit (''mm'': conda)'
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import warnings

warnings.filterwarnings("ignore")
# %%
import os
import sys
import time

import fire
import torch
from fvcore.nn import (
    ActivationCountAnalysis,
    FlopCountAnalysis,
    flop_count_str,
    flop_count_table,
)

# %%
from pumer.model import get_model, get_model_config

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d_%H:%M:%S", stream=sys.stdout, level=logging.INFO
)


# %%
def get_inputs(batch_size=1, img_size=384, device="cuda:0", task="vqa2"):
    pixel_values = torch.randn(batch_size, 3, img_size, img_size, device=device)
    input_ids = torch.randint(1000, [batch_size, 32], device=device)
    attn_masks = torch.randint(1, [batch_size, 32], device=device)
    if task == "vqa2" or task == "ve" or task == "irtr":
        inputs = dict(text_ids=input_ids, text_masks=attn_masks, pixel_values=pixel_values)
    elif task == "nlvr2":
        pixel_values2 = torch.randn(batch_size, 3, img_size, img_size, device=device)
        inputs = dict(
            text_ids=input_ids, text_masks=attn_masks, pixel_values1=pixel_values, pixel_values2=pixel_values2
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return inputs


# %%
def get_bench_model(name, task="vqa2", img_size=384, device="cuda:0", cfg_path=None):
    cfg_class = get_model_config(name)
    model_class = get_model(name, task)
    prune_layers = os.environ.get("prune_layers", None)
    reduce_layers = os.environ.get("reduce_layers", None)
    prune_method = os.environ.get("prune_method", "mlp_states")
    keep_ratio = os.environ.get("keep_ratio", 1)
    merge_ratio = os.environ.get("merge_ratio", 0)
    sim_method = os.environ.get("sim_method", "mean_head")
    merge_style = os.environ.get("merge_style", "tome")
    merge_r = os.environ.get("merge_r", 0)
    prune_r = os.environ.get("prune_r", 0)

    cfg = cfg_class()
    cfg.merge_ratio = float(merge_ratio)
    cfg.merge_r = float(merge_r)
    cfg.prune_r = float(prune_r)
    cfg.image_size = img_size
    if hasattr(cfg, "vision_config"):
        cfg.vision_config.image_size = img_size
        # logger.info(cfg.vision_config)
    cfg.prune_method = prune_method  # linear_states, mlp_states, first_head, mean_head
    cfg.sim_method = sim_method
    cfg.merge_style = merge_style
    cfg.merge_text = float(os.environ.get("merge_text", 0))
    cfg.keep_ratio = float(keep_ratio)
    if prune_layers:
        cfg.prune_layers = list(map(int, prune_layers.split(",")))
        logger.info(f"{cfg.prune_method}, {cfg.prune_layers=}, {cfg.keep_ratio=}")
    if reduce_layers:
        cfg.reduce_layers = list(map(int, reduce_layers.split(",")))
        # logger.info(f"{cfg.reduce_layers=}, {cfg.merge_ratio=}, {cfg.keep_ratio=}")
        logger.info(f"{cfg.reduce_layers=}, {cfg.merge_r=}, {cfg.prune_r=}, {cfg.merge_text=}, {cfg.merge_style=}")
    if task == "nlvr2":
        cfg.token_types = 3
    model = model_class(cfg)
    model.eval()

    #
    model = model.to(device)
    return model


# %%
def bench_latency(model, inputs):
    warm_steps = 3
    for _ in range(warm_steps):
        model(**inputs, return_dict=True)

    timings = []
    start_mems = []
    end_mems = []
    diff_mems = []
    MB = 1024.0 * 1024.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    for _ in range(10):
        torch.cuda.synchronize()
        start_mem = torch.cuda.max_memory_allocated() / MB
        start = time.perf_counter()
        model(**inputs, return_dict=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        end_mem = torch.cuda.max_memory_allocated() / MB
        diff_mem = end_mem - start_mem
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        timings.append(end - start)
        start_mems.append(start_mem)
        end_mems.append(end_mem)
        diff_mems.append(diff_mem)

    timings = torch.as_tensor(timings, dtype=torch.float32)
    start_mems = torch.as_tensor(start_mems, dtype=torch.float32)
    end_mems = torch.as_tensor(end_mems, dtype=torch.float32)
    diff_mems = torch.as_tensor(diff_mems, dtype=torch.float32)
    t_mean = timings.mean().item()
    t_std = timings.std().item()
    sm_mean = start_mems.mean().item()
    em_mean = end_mems.mean().item()
    dm_mean = diff_mems.mean().item()
    sm_std = start_mems.std().item()
    em_std = end_mems.std().item()
    dm_std = diff_mems.std().item()

    # logger.info(t_mean, t_std, sm_mean, sm_std, em_mean, em_std, dm_mean, dm_std)
    return t_mean, t_std, sm_mean, sm_std, em_mean, em_std, dm_mean, dm_std


# %%
def bench_model_throughput(model, inputs, batch_size=1, warmup=3, duration=30):
    T0 = warmup  # warmup
    T1 = duration  # run 60 seconds

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(**inputs, return_dict=True)
    timing = []
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(**inputs, return_dict=True)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    std = timing.std().item()
    return batch_size / timing.mean().item(), std


# %%
def bench(
    batch_size=1,
    img_size=384,
    model="vilt",
    task="vqa2",
    device="cuda:0",
    warmup=3,
    duration=30,
    cfg_path=None,
    bench_cross=False,
):
    torch.cuda.empty_cache()
    logger.info(f"init {model=}, {task=}, {batch_size=}, {img_size=}, {device=}...")
    vl_model = get_bench_model(model, task, img_size, device, cfg_path)

    # logger.info(f"get {model} inputs...")
    inputs = get_inputs(batch_size, img_size, device, task)
    inputs["bench_cross"] = bench_cross
    # logger.info(f"start {model} benchmarking latency...")
    latency, lat_std, start_mem, sm_std, end_mem, em_std, diff_mem, dm_std = bench_latency(vl_model, inputs)
    logger.info(
        f"{latency=:.5f} s, {lat_std=:4f}, {start_mem=:.1f} MB, {sm_std:.4f}, {end_mem=:.1f} MB, {em_std:.4f}, {diff_mem=:.1f} MB, {dm_std:.4f}"
    )

    if duration > 5:
        # logger.info(f"start {model} benchmarking for {duration} seconds...")
        throughput, std = bench_model_throughput(vl_model, inputs, batch_size, warmup, duration)
        logger.info(f"{model=}, {img_size=}, {throughput=:.2f} examples/s, {batch_size=}, {std=:.4f}, done.")


# %%
# def logger.info_act(activation_counter):
#     max_len = max(len(k) for k in activation_counter.keys())
#     for k, v in activation_counter.items():
#         if v:
#             logger.info(f"{k.ljust(max_len)}| {v/1e6:.3f} M")


# %%
def profile_flops(batch_size=1, img_size=384, model="vilt", task="vqa2", cfg_path=None):
    logger.info(f"init {model=}, {batch_size=}, {img_size=}...")
    vl_model = get_bench_model(model, task, img_size, "cpu", cfg_path)

    logger.info(f"get {model} inputs...")
    inputs = get_inputs(batch_size, img_size, "cpu", task)
    flops = FlopCountAnalysis(vl_model, tuple(inputs.values()))
    acts = ActivationCountAnalysis(vl_model, tuple(inputs.values()))
    # logger.info_act(acts.by_module())
    logger.info(f"\n{flop_count_table(flops, max_depth=10, activations=acts)}".replace("|", ""))
    logger.info(f"\n{flop_count_str(flops, acts)}".replace("|", ""))
    # logger.info(flops.total(), acts.total())


# %% [markdown]
#
# NVIDIA GeForce GTX 1080 Ti, 12 GiB Memory
#
# vilt:
# 94.0003568320111 examples/s @ batch size 1
# 128.90132188036617 examples/s @ batch size 8
# 154.82472467664638 examples/s @ batch size 16
# 164.70111470716608 examples/s @ batch size 32
# 166.14938672807935 examples/s @ batch size 64
# 165.35095128232382 examples/s @ batch size 72
#
# meter:
# 19.49256573751063 examples/s @ batch size 1
# 24.75214526372998 examples/s @ batch size 8
# OOM for batch size 16

# %%

if __name__ == "__main__":
    fire.Fire()
