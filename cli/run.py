import json
import re
import types
from pathlib import Path

import hydra
import torch
import transformers.optimization as opt
from accelerate import Accelerator, DistributedDataParallelKwargs, tracking
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pumer.metric import get_prediction_writer
from pumer.model import get_model
from pumer.task import get_task
from pumer.utils.train_utils import (
    MyTensorBoardTracker,
    MyWandBTracker,
    find_resume_checkpoint,
    get_params_group,
    init_deepspeed,
    init_wandb_id,
    m_print,
    save_states,
)

tracking.logger = logger


def make_infinite_dataloader(dataloader):
    while True:
        for step, batch in enumerate(dataloader):
            yield step, batch


def predict(accelerator, model, task, dataloader, prediction_writer=None):
    model.eval()
    should_write_predict = prediction_writer is not None and accelerator.is_main_process
    # cfg = task.config
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        # dataset specific extra info
        extra = batch.get("extra", None)
        # dataset example always have numeric id
        item_ids = batch.get("item_ids", None)
        if item_ids is not None:
            if isinstance(item_ids, torch.Tensor):
                item_ids = accelerator.gather(item_ids)
                item_ids = item_ids.tolist()

        with torch.no_grad():
            outputs = model(**batch)
        preds, _, _ = task(outputs, batch, accelerator.gather_for_metrics, is_training=False)
        del outputs

        preds = preds["prediction"].tolist()
        if should_write_predict:
            prediction_writer.process(item_ids, preds, extra)

    if should_write_predict:
        prediction_writer.finish()
    return task.reset_metric()


def train(
    cfg,
    accelerator: Accelerator,
    model,
    train_task,
    train_dataloader,
    dev_task=None,
    dev_dataloader=None,
    teacher_model=None,
):
    # configure optimizer
    params_group = get_params_group(cfg, model)
    optimizer = torch.optim.AdamW(
        params_group,
        lr=cfg.train.optimizer.learning_rate,
        betas=(0.9, 0.98),
    )

    # config training and warmup stesp
    num_iters_per_epoch = len(train_dataloader)

    epochs = cfg.train.epochs
    num_training_steps_total = num_iters_per_epoch * epochs
    ratio_steps = int(num_training_steps_total * cfg.train.scheduler.warmup_ratio)
    num_warmup_steps = cfg.train.scheduler.num_warmup_steps or ratio_steps

    # use transformers predefined scheduler
    sched_fn = getattr(opt, cfg.train.scheduler.name)
    sched_cfg = cfg.train.scheduler.cfg if cfg.train.scheduler.cfg else {}
    scheduler = sched_fn(optimizer, num_warmup_steps, num_training_steps_total, **sched_cfg)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

    if accelerator.is_main_process and cfg.train.log_freq > 0:
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.watch(model, "all", log_freq=cfg.train.log_freq)
            accelerator.print("will monitor model parameters and gradients using wandb ")
        except Exception as e:
            accelerator.print(f"wandb watch failed: {e=}")

    # we need to recompute total training steps after dataloader preparation
    num_iters_per_epoch = len(train_dataloader)
    num_training_steps = num_iters_per_epoch * epochs
    total_batch_size = cfg.per_device_train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    global_step = 0
    epoch = 0
    resume_checkpoint = find_resume_checkpoint(cfg.train.resume_from_checkpoint)
    if resume_checkpoint:  # resume checkpoint and find remaining steps to train
        accelerator.print(f"Resumed from checkpoint: {resume_checkpoint}")
        accelerator.load_state(resume_checkpoint)
        path = Path(resume_checkpoint)
        global_step = int(path.name.replace("step-", ""))
        epoch = global_step // num_iters_per_epoch
        log_str = f"resume from {global_step=}, "
        save_dict_file = resume_checkpoint / "save_dict.pt"
        save_dict = torch.load(save_dict_file) if save_dict_file.exists() else {}
    else:
        log_str = f"training from scratch, "
        save_dict = {}

    accelerator.print(
        f"{log_str}{num_training_steps_total=}, {num_warmup_steps=}, {num_training_steps=}, {total_batch_size=}"
    )

    g_bar = tqdm(
        range(num_training_steps),
        desc=f"{epoch=}, {global_step=}",
        position=0,
        disable=not accelerator.is_local_main_process,
    )
    g_bar.set_description(f"{epoch=}, {global_step=}")
    g_bar.update(global_step)

    d_bar = tqdm(range(num_iters_per_epoch), position=1, leave=False, disable=not accelerator.is_local_main_process)

    infinite_dataloader = make_infinite_dataloader(train_dataloader)
    model.train()

    dev_metric = 0
    best_score = float(save_dict.get("best_score", -1e6))
    patience = save_dict.get("patience", 0)
    accelerator.log({"train/epoch": epoch}, step=global_step)
    while True:
        # get one batch, train one step
        step, batch = next(infinite_dataloader)
        batch.pop("item_ids", None)
        batch.pop("extra", None)

        teacher_outputs = None
        if teacher_model is not None and epoch >= cfg.train.distill_after_epochs:
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)

        with accelerator.accumulate(model):

            batch["global_step"] = global_step

            outputs = model(**batch)
            _, losses, metrics = train_task(
                outputs,
                batch,
                accelerator.gather_for_metrics,
                is_training=True,
                global_step=global_step,
                teacher_outputs=teacher_outputs,
            )
            train_loss = losses["loss"]
            train_metric = metrics["metric"]

            accelerator.backward(train_loss)
            if cfg.train.clip_gradient_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_gradient_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # log train stats
        logs = {f"train/{k}": v for k, v in losses.items()}
        for k, v in metrics.items():
            logs[f"train/{k}"] = v
        for g in optimizer.param_groups:
            logs[f"lr/{g['name']}"] = g["lr"]
        accelerator.log(logs, step=global_step)

        # if more than eval_steps or the last step, then do eval
        if (global_step + 1) % cfg.train.save_every_steps == 0 or (global_step + 1) >= num_training_steps:
            save_best = False
            if cfg.do_step_eval:
                dev_metrics = predict(accelerator, model, dev_task, dev_dataloader)
                dev_metric = dev_metrics.pop("metric")
                dev_logs = {f"dev/metric": dev_metric, "dev/epoch": epoch}
                for dk, dv in dev_metrics.items():
                    dev_logs[f"dev/{dk}"] = dv
                accelerator.log(dev_logs, step=global_step)
                g_bar.set_description(f"{epoch=}, {global_step=}, {dev_metric=:.4f}")
                if dev_metric > best_score:
                    best_score = dev_metric
                    save_best = True
                    patience = 0
                    accelerator.log({"dev/best": best_score}, step=global_step)
                    g_bar.set_description(f"{epoch=}, {global_step=}, {dev_metric=:.4f}, best={best_score:.4f}")
                else:
                    patience += 1
                if patience >= cfg.train.stop_patience:
                    accelerator.print(f"early stopped training for {patience=}, {best_score=}")
                    break
                model.train()  # put model back into train mode
            save_dict = {
                "best_score": best_score,
                "save_best": save_best,
                "patience": patience,
            }
            save_states(
                accelerator,
                global_step,
                cfg.train.ckpt_save_dir,
                cfg.train.num_kept_ckpt,
                cfg.wandb.id,
                model,
                save_dict,
            )

        global_step += 1
        if global_step >= num_training_steps:
            break

        if step == num_iters_per_epoch - 1:
            epoch += 1
            d_bar.reset()
            train_task.reset_metric()  # reset train metrics after an epoch
            accelerator.log({"train/epoch": epoch}, step=global_step)

        g_bar.set_description(f"{epoch=}, {global_step=}")
        g_bar.update()

        d_bar.set_description(f"{step=}, {train_loss=:.6f}, {train_metric=:.4f}")
        d_bar.update()

    accelerator.print(f"training finished!")


@hydra.main(config_path="../conf", config_name="run", version_base="1.2")
def run(cfg):
    resume_ckpt_path = Path(cfg.train.resume_from_checkpoint)

    #### ===== init accelerator
    deepspeed_plugin = init_deepspeed(cfg)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.train.find_unused_parameters)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        cpu=cfg.cpu,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[kwargs],
    )
    accelerator.print = types.MethodType(m_print, accelerator)
    accelerator.print(f"initialized {accelerator.num_processes} gpus")

    # initialze tracker on main process
    #### ===== init wandb and tensorboard trackers on the main process
    if accelerator.is_main_process:
        trackers = []
        if cfg.wandb.project and cfg.wandb.name:
            cfg.wandb.id = init_wandb_id(resume_ckpt_path / "wandb-id.txt")
            wandb_tracker = MyWandBTracker(cfg.wandb.name, **cfg.wandb)
            trackers.append(wandb_tracker)
        if cfg.tensorboard.name:
            tensorboard_tracker = MyTensorBoardTracker(cfg.tensorboard.name, cfg.tensorboard.logging_dir)
            trackers.append(tensorboard_tracker)
        accelerator.log_with.extend(trackers)
    accelerator.init_trackers(cfg.log_name, config=OmegaConf.to_container(cfg, resolve=True))

    accelerator.print("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    #### ===== init model
    model_ckpt_path = cfg.model.ckpt_path
    resume_model_path = resume_ckpt_path / "best-model"
    if cfg.do_train and resume_model_path.exists():
        model_ckpt_path = resume_model_path
        accelerator.print(f"init model from {model_ckpt_path}")

    # use model config class to init config
    model_class = get_model(cfg.model.name, cfg.task.name)
    if model_ckpt_path:
        model_cfg = model_class.config_class.from_pretrained(model_ckpt_path, from_scratch=False)
    else:
        model_cfg = model_class.config_class(from_scratch=True)
    model_conf = cfg.model.get("config", {})
    if model_conf and cfg.do_train:  # only use customized config for training
        model_cfg.update(OmegaConf.to_container(model_conf))

    if model_ckpt_path:  # init model weights
        model, loading_info = model_class.from_pretrained(model_ckpt_path, config=model_cfg, output_loading_info=True)
        accelerator.print(f"loaded model from {model_ckpt_path}, {loading_info=}")
    else:
        model = model_class(model_cfg)
        accelerator.print(f"initialized model")

    ### init teacher model
    teacher_ckpt_path = cfg.model.teacher_ckpt_path
    if teacher_ckpt_path and Path(teacher_ckpt_path).exists():
        # add support for different model_class
        model_class = get_model(cfg.model.teacher_name, cfg.task.name)
        teacher_model = model_class.from_pretrained(
            teacher_ckpt_path, output_hidden_states=True, output_attentions=True
        )
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model = teacher_model.to(accelerator.device)
        teacher_model.eval()
        # teacher_model = accelerator.prepare_model(teacher_model)
        accelerator.print(f"use teacher from {teacher_ckpt_path} for distillation!")
    else:
        teacher_model = None

    ### freeze model parameters
    freeze_patterns = cfg.train.freeze_patterns
    if freeze_patterns:  # freeze parameters
        for param_name, param in model.named_parameters():
            if re.match(freeze_patterns, param_name):
                param.requires_grad = False
                accelerator.print(f"freeze {param_name}")
    model = model.to(accelerator.device)

    train_task = get_task(cfg.task.name)(cfg, accelerator.device)
    dev_task = get_task(cfg.task.name)(cfg, accelerator.device)
    test_task = get_task(cfg.task.name)(cfg, accelerator.device)

    if cfg.do_eval:
        # prepare val dataloader
        val_dataset = hydra.utils.instantiate(cfg.task.eval_dataset)
        dev_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=cfg.per_device_dev_batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        dev_dataloader = accelerator.prepare_data_loader(dev_dataloader)
        accelerator.print(f"dev dataset size={len(dev_dataloader.dataset)}, dev loader size={len(dev_dataloader)}...")

    else:
        dev_dataloader = None

    if cfg.do_train:
        accelerator.print(f"init train dataset...")

        # prepare train dataloader and train metric
        train_dataset = hydra.utils.instantiate(cfg.task.train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=cfg.per_device_train_batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        accelerator.print(
            f"train dataset size={len(train_dataloader.dataset)}, train loader size={len(train_dataloader)}..."
        )

        total_num_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        accelerator.print(f"{total_num_params=:.1f} millon")
        accelerator.print(f"{trainable_num_params=:.1f} millon")

        #### begin training
        train(cfg, accelerator, model, train_task, train_dataloader, dev_task, dev_dataloader, teacher_model)

        accelerator.wait_for_everyone()
        accelerator.end_training()

    accelerator.print("\n")
    accelerator.print("prepare accelerator for inference")
    ckpt_save_dir = Path(cfg.train.ckpt_save_dir)
    model_ckpt_path = ckpt_save_dir / "best-model"
    if cfg.do_train and model_ckpt_path.exists():
        # load best model ckpt if trained
        accelerator.clear()
        model, loading_info = model_class.from_pretrained(model_ckpt_path, output_loading_info=True)
        accelerator.print(f"load model from {model_ckpt_path}, {loading_info=}")
        model = model.to(accelerator.device)
    model = accelerator.prepare_model(model)

    eval_results = {}
    if cfg.do_eval:
        accelerator.print("do evaluation")
        dev_metrics = predict(accelerator, model, dev_task, dev_dataloader)
        dev_metric = dev_metrics.pop("metric")
        eval_results["dev_metric"] = round(float(dev_metric), 4)
        accelerator.print(f"evaluation finished, {dev_metric=:.4f}")
        accelerator.print(f"evaluation metrics, {dev_metrics=}")

    if cfg.do_predict:
        accelerator.print("do prediction for test")
        accelerator.print("init test dataset...")
        predict_dataset = hydra.utils.instantiate(cfg.task.predict_dataset)
        test_dataloader = DataLoader(
            predict_dataset,
            shuffle=False,
            batch_size=cfg.per_device_dev_batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        test_dataset_size = len(test_dataloader.dataset)
        accelerator.print(
            f"test dataset size={len(test_dataloader.dataset)}, test loader size={len(test_dataloader)}..."
        )

        test_dataloader = accelerator.prepare_data_loader(test_dataloader)

        prediction_writer = None
        if accelerator.is_main_process:
            prediction_writer_class = get_prediction_writer(cfg.task.name)
            if prediction_writer_class is not None:
                prediction_writer = prediction_writer_class(cfg, test_dataset_size)

        predict_metrics = predict(accelerator, model, test_task, test_dataloader, prediction_writer)
        if cfg.do_eval_predict:
            predict_metric = predict_metrics.pop("metric")
            eval_results["predict_metric"] = round(float(predict_metric), 4)
            accelerator.print(f"{predict_metric=:.4f}!")
            accelerator.print(f"{predict_metrics=}!")
        accelerator.print("prediction finished for test")
    if accelerator.is_main_process:
        with open(cfg.results_metric_file, "w") as f:
            json.dump(eval_results, f)
    accelerator.wait_for_everyone()
    accelerator.print("all done!")


if __name__ == "__main__":
    run()
