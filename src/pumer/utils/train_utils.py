import os
import re
from pathlib import Path
from typing import MutableMapping, Optional

import torch
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.tracking import TensorBoardTracker, WandBTracker
from loguru import logger

from .file_utils import remove_path

home_str = str(Path().home())


def get_params_group(cfg, model):
    learning_rate = cfg.train.optimizer.learning_rate
    weight_decay = cfg.train.optimizer.weight_decay
    lr_mult = cfg.train.optimizer.lr_mult
    lr_mult_cross_modal = cfg.train.optimizer.lr_mult_cross_modal
    lr_mult_pruner = cfg.train.optimizer.lr_mult_pruner

    model_vars = {n: p for n, p in model.named_parameters() if p.requires_grad}

    no_decay_patterns = "(.*)bias|(.*)[N,n]orm(.*)"

    module_patterns = {
        "(.*)token_predictors(.*)": lr_mult_pruner,
        "(.*)cross_modal(.*)": lr_mult_cross_modal,
        "(.*)classifier(.*)": lr_mult,
    }

    module_names = set()
    all_module_vars = []
    for module_pattern, lr_multiplier in module_patterns.items():
        module_vars = {n: p for n, p in model_vars.items() if re.match(module_pattern, n)}
        if module_vars:
            all_module_vars.append(
                {
                    "name": module_pattern,
                    "vars": module_vars,
                    "lr": lr_multiplier * learning_rate,
                }
            )
            module_names |= set(module_vars.keys())

    ## rest names are backbone modules
    rest_names = set(model_vars.keys()) - module_names
    rest_vars = {n: p for n, p in model_vars.items() if n in rest_names}
    if rest_vars:
        all_module_vars.append(
            {
                "name": "backbone",
                "vars": rest_vars,
                "lr": learning_rate,
            }
        )

    params_groups = []
    for module in all_module_vars:
        decay_vars = [p for n, p in module["vars"].items() if not re.match(no_decay_patterns, n)]
        if decay_vars:
            param_group = {
                "name": module["name"],
                "params": decay_vars,
                "weight_decay": weight_decay,
                "lr": module["lr"],
            }
            params_groups.append(param_group)

        nodecay_vars = [p for n, p in module["vars"].items() if re.match(no_decay_patterns, n)]
        if nodecay_vars:
            param_group = {
                "name": module["name"] + "-nodecay",
                "params": nodecay_vars,
                "weight_decay": 0.0,
                "lr": module["lr"],
            }
            params_groups.append(param_group)

    return params_groups


def init_deepspeed(cfg):
    if cfg.use_deepspeed:
        grad_accum_steps = cfg.gradient_accumulation_steps
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=grad_accum_steps)
        logger.info(f"use deepspeed, {grad_accum_steps=}")
    else:
        deepspeed_plugin = None
    return deepspeed_plugin


def find_resume_checkpoint(resume_from_checkpoint):
    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        return None
    # find checkpoint.txt
    checkpoint_file = checkpoint_path / "checkpoint.txt"
    if not checkpoint_file.exists():
        return None
    ckpt_dir = checkpoint_file.read_text().strip()
    resume_ckpt_dir = checkpoint_path / ckpt_dir
    if not resume_ckpt_dir.exists():
        return None
    return resume_ckpt_dir


def handle_ckpt_files(ckpt_dir, pattern, num_keep):
    glob_checkpoints = [str(x) for x in Path(ckpt_dir).glob(pattern)]
    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    num_ckpts = len(checkpoints_sorted)
    if num_ckpts <= num_keep:
        return
    checkpoints_to_be_deleted = checkpoints_sorted[: num_ckpts - num_keep]
    for checkpoint in checkpoints_to_be_deleted:
        remove_path(checkpoint)


def save_states(accelerator, step, ckpt_save_dir, num_kept_ckpt, wandb_id=None, model=None, save_dict=None):
    accelerator.wait_for_everyone()
    ckpt_save_dir = Path(ckpt_save_dir)
    ckpt_file = ckpt_save_dir / "checkpoint.txt"
    # ckpt_str = "best-ckpt" if is_best else f"step-{step}"
    ckpt_str = f"step-{step}"
    wandb_id_file = ckpt_save_dir / "wandb-id.txt"
    current_save_dir = ckpt_save_dir / ckpt_str
    save_dict_file = current_save_dir / "save_dict.pt"
    with accelerator.main_process_first():
        current_save_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        ckpt_file.write_text(ckpt_str)
        if wandb_id:
            wandb_id_file.write_text(wandb_id)
        if save_dict:
            torch.save(save_dict, save_dict_file)
    accelerator.save_state(current_save_dir)
    if model is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.config.save_pretrained(current_save_dir)  # save config
        is_best = save_dict.get("save_best", False) if save_dict else False
        if is_best:
            unwrapped_model.save_pretrained(
                ckpt_save_dir / "best-model",
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
    # delete old ckpt files
    if accelerator.is_main_process:
        handle_ckpt_files(ckpt_save_dir, "step-*", num_kept_ckpt)


def init_wandb_id(wandbid_file):
    return wandbid_file.read_text().strip() if wandbid_file.exists() else wandb.util.generate_id()


def m_print(self, *args, **kwargs):
    """
    Use in replacement of `print()` to only print once per server.
    """
    if self.is_local_main_process:
        if isinstance(args[0], str):
            args = (args[0].replace(home_str, "~"),) + args[1:]
        logger.info(*args)


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class MyWandBTracker(WandBTracker):
    """just a thin wrapper that removes logger for the log() from WandBTracker, and support full init kwargs"""

    def __init__(self, run_name: str, **kwargs):
        self.run_name = run_name  # wandb.name
        self.run = wandb.init(**kwargs)

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.
        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        self.run.log(values, step=step, **kwargs)


class MyTensorBoardTracker(TensorBoardTracker):
    """just a thin wrapper that removes logger for the log() from TensorBoardTracker"""

    def store_init_configuration(self, values: dict):
        # import pprint
        # pp = pprint.PrettyPrinter(depth=4)
        # self.writer.add_text("config", pp.pformat(values).replace("\n", "<br />"))
        import yaml

        self.writer.add_text("config", f'<h3><pre>{yaml.dump(values).replace(home_str, "~")}</pre></h3>')
        self.writer.flush()
        # super().store_init_configuration(flatten(values, sep="/"))

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.
        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `SummaryWriter.add_scaler`,
                `SummaryWriter.add_text`, or `SummaryWriter.add_scalers` method based on the contents of `values`.
        """
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step, **kwargs)
            elif isinstance(v, dict):
                self.writer.add_scalars(k, v, global_step=step, **kwargs)
        self.writer.flush()


def wandb_log_builder(accelerator: Accelerator, wandb_run):
    def wandb_log(values, step=None):
        if accelerator.is_local_main_process and wandb_run is not None:
            wandb_run.log(values, step=step)

    return wandb_log
