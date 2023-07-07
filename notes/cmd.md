# Common commands

Below are some example commands for training and evaluation, please change dir related path to your actual path

If you notice (slightly) different training/evaluation results, watch out the different configurations for ViLT and METER

for vilt:
- `task.is_clip=False`
- `task.tokenizer=bert-base-uncased`

for meter:
- `task.is_clip=True`
- `task.tokenizer=roberta-base`

## Training

### vilt-vqa2-ft

```bash

task=vqa2
n_gpus=8
exp=vilt-vqa2
ep=10
ckpt=ft-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=vilt \
    model.ckpt=converted/vilt_pretrain \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=1e-4 \
    train.optimizer.lr_mult=10 \
    train.save_every_steps=800 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=64 \
    do_eval=True \
    per_device_dev_batch_size=64 \
    do_step_eval=True \
    do_predict=True \
    task.image_size=384 \
    task.is_clip=False \
    log_name=${ckpt//\//-} \
    wandb.project=tip-${exp}

```

### vilt-vqa2-pumer

```bash
l=2,5,8
pr=0.1
mr=0.3
mt=0.2
s=tome
sm=mh
ep=20


ckpt=p${pr}r${mr}t${mt}-${l//,}-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=vilt \
    model.config.prune_r=${pr} \
    model.config.merge_r=${mr} \
    model.config.merge_text=${mt} \
    model.config.merge_style=${s} \
    model.config.reduce_layers="[${l}]" \
    model.ckpt=converted/vilt_pretrain \
    model.teacher_ckpt_path=/vilt-vqa2-ft/ckpt/ft-e10/best-model \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=1e-4 \
    train.optimizer.lr_mult=10 \
    train.save_every_steps=800 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=64 \
    do_eval=True \
    per_device_dev_batch_size=64 \
    do_step_eval=True \
    do_predict=True \
    task.image_size=384 \
    task.is_clip=False \
    log_name=${ckpt//\//-} \
    wandb.project=tip-${exp}

```

### vilt-ve-ft

```bash


```

### vilt-ve-pumer

```bash
task=ve
lr=1e-4
n_gpus=8
exp=vilt-ve

l=2,5,8
pr=0.1
mr=0.3
mt=0.2
s=tome
sm=mh
ep=20

ckpt=p${pr}-r${mr}-t${mt}-${sm}-${s}-${l//,}-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=vilt \
    model.config.prune_r=${pr} \
    model.config.merge_r=${mr} \
    model.config.merge_text=${mt} \
    model.config.merge_style=${s} \
    model.config.sim_method=${sm} \
    model.config.reduce_layers="[${l}]" \
    model.ckpt=converted/vilt_pretrain \
    model.teacher_ckpt_path=/vilt-ve-ft/ckpt/ft/best-model \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=${lr} \
    train.optimizer.lr_mult=10 \
    train.save_every_steps=800 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=64 \
    do_eval=True \
    per_device_dev_batch_size=64 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.is_clip=False \
    log_name=ai2-${ckpt//\//-} \
    wandb.project=tip-${exp}

```


### vilt-nlvr2-ft

```bash
task=nlvr2
exp=vilt-nlvr2
ep=10
ckpt=ft-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=vilt \
    model.ckpt=converted/vilt_pretrain_nlvr2 \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=${lr} \
    train.optimizer.lr_mult=10 \
    train.save_every_steps=300 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=32 \
    do_eval=True \
    per_device_dev_batch_size=32 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.is_clip=False \
    log_name=ai2-${ckpt//\//-} \
    wandb.project=tip-${exp}


```

### vilt-nlvr2-pumer

```bash

ckpt=p${pr}-r${mr}-t${mt}-${sm}-${s}-${l//,}-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=vilt \
    model.config.prune_r=${pr} \
    model.config.merge_r=${mr} \
    model.config.merge_text=${mt} \
    model.config.merge_style=${s} \
    model.config.sim_method=${sm} \
    model.config.reduce_layers="[${l}]" \
    model.ckpt=converted/vilt_pretrain_nlvr2 \
    model.teacher_ckpt_path=/vilt-nlvr2-ft/ckpt/ft-e10/best-model \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=${lr} \
    train.optimizer.lr_mult=10 \
    train.save_every_steps=300 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=32 \
    do_eval=True \
    per_device_dev_batch_size=32 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.is_clip=False \
    log_name=ai2-${ckpt//\//-} \
    wandb.project=tip-${exp}

```

### meter-vqa2-ft

```bash
task=vqa2
n_gpus=8

exp=meter-vqa2
ep=10
ckpt=ft-e${ep}

# e.g. using 8 A100-40GB GPUs
torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=meter \
    model.ckpt=converted/meter_pretrain_384 \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=5e-6 \
    train.optimizer.lr_mult=50 \
    train.optimizer.lr_mult_cross_modal=5 \
    train.save_every_steps=2000 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=24 \
    do_eval=True \
    per_device_dev_batch_size=24 \
    do_step_eval=True \
    do_predict=True \
    task.image_size=384 \
    task.tokenizer=roberta-base \
    log_name=${ckpt//\//-} \
    wandb.project=tip-${exp} \
    2>&1 | tee data/wip/train-${ckpt//\//_}.log

```

### meter-vqa2-pumer

```bash


l=0,2,4,6
pr=0.2
mr=0.2
mt=0.2
ep=20

ckpt=p${pr}r${mr}t${mt}-${l//,}-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    data_dir=${data_dir} \
    task.predictions_output_file=${data_dir}/preds/meter-vqa2-p0.2r0.2t0.2-0246-e20.json \
    task=vqa2 \
    model=meter \
    ++model.config.prune_r=0.2 \
    ++model.config.merge_r=0.2 \
    ++model.config.merge_text=0.2 \
    ++model.config.reduce_layers=[0,2,4,6] \
    model.ckpt=converted/meter_pretrain_384 \
    model.teacher_ckpt_path=ft/meter-vqa2/best-model \
    train.ckpt=p0.2r0.2t0.2-0246-e20 \
    train.epochs=20 \
    train.optimizer.learning_rate=5e-6 \
    train.optimizer.lr_mult=50 \
    train.optimizer.lr_mult_cross_modal=5 \
    train.save_every_steps=3000 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=32 \
    do_eval=True \
    per_device_dev_batch_size=32 \
    do_step_eval=True \
    do_predict=True \
    task.image_size=384 \
    task.tokenizer=roberta-base \
    log_name=p0.2r0.2t0.2-0246-e20 \
    wandb.project=pumer-meter-vqa2

```


### meter-ve-ft

```bash
# meter ve

task=ve
n_gpus=8

exp=meter-ve
# ft meter ve
ep=10
ckpt=ft-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=meter \
    model.ckpt=converted/meter_pretrain_384 \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=2e-6 \
    train.optimizer.lr_mult=10 \
    train.optimizer.lr_mult_cross_modal=5 \
    train.save_every_steps=2500 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=24 \
    do_eval=True \
    per_device_dev_batch_size=24 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.tokenizer=roberta-base \
    log_name=${ckpt//\//-} \
    wandb.project=tip-${exp} \
    2>&1 | tee data/wip/train-${ckpt//\//_}.log
```

### meter-ve-pumer

```bash

l=0,2,4,6
pr=0.3
mr=0.5
mt=0.2
ep=20

ckpt=p${pr}r${mr}t${mt}-${l//,}-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/meter-ve-p0.3r0.5t0.2-0246-e20.json \
    task=ve \
    data_dir=\data \
    results_dir=\results \
    model=meter \
    ++model.config.prune_r=0.3 \
    ++model.config.merge_r=0.5 \
    ++model.config.merge_text=0.2 \
    ++model.config.reduce_layers=[0,2,4,6] \
    model.ckpt_path=/ft/ckpt/ft-e10/best-model \
    model.teacher_ckpt_path=/ft/ckpt/ft-e10\best-model \
    train.ckpt=p0.3r0.5t0.2-0246-e20 \
    train.epochs=20 \
    train.optimizer.learning_rate=2e-6 \
    train.optimizer.lr_mult=10 \
    train.optimizer.lr_mult_cross_modal=5 \
    train.save_every_steps=2500 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=24 \
    do_eval=True \
    per_device_dev_batch_size=24 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.tokenizer=roberta-base \
    log_name=p0.3r0.5t0.2-0246-e20 \
    wandb.project=tip-meter-ve
```

### meter-nlvr2-ft

```bash

ep=10
task=nlvr2
exp=meter-nlvr2
ckpt=ft-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=meter \
    model.ckpt=converted/meter_pretrain_nlvr2_288 \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=5e-6 \
    train.optimizer.lr_mult=50 \
    train.optimizer.lr_mult_cross_modal=5 \
    train.save_every_steps=800 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=12 \
    do_eval=True \
    per_device_dev_batch_size=12 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=288 \
    task.tokenizer=roberta-base \
    log_name=ai2-${ckpt//\//-} \
    wandb.project=tip-${exp}
```

### meter-nlvr2-pumer

```bash


ckpt=p${pr}r${mr}t${mt}-${l//,}-e${ep}

torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task.image_db_dir=/data/lmdb \
    task.predictions_output_file=/results/${exp}-${ckpt}.json \
    task=${task} \
    data_dir=/data \
    results_dir=/results \
    model=meter \
    model.config.prune_r=${pr} \
    model.config.merge_r=${mr} \
    model.config.merge_text=${mt} \
    model.config.merge_style=${s} \
    model.config.sim_method=${sm} \
    model.config.reduce_layers="[${l}]" \
    model.ckpt=converted/meter_pretrain_nlvr2_288 \
    model.teacher_ckpt_path=/meter-nlvr2-ft/ckpt/ft-e10/best-model \
    train.ckpt=${ckpt} \
    train.epochs=${ep} \
    train.optimizer.learning_rate=5e-6 \
    train.optimizer.lr_mult=50 \
    train.optimizer.lr_mult_cross_modal=5 \
    train.save_every_steps=800 \
    gradient_accumulation_steps=1 \
    do_train=True \
    per_device_train_batch_size=12 \
    do_eval=True \
    per_device_dev_batch_size=12 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=288 \
    task.tokenizer=roberta-base \
    log_name=ai2-${ckpt//\//-} \
    wandb.project=tip-${exp}

```

## Evaluation

remove training related args in above scripts for evaluation

e.g.

```bash
# predict vilt-ve-ft

data_dir=`pwd`/data

torchrun --nproc_per_node 4 cli/run.py \
    task=ve \
    data_dir=${data_dir} \
    task.image_db_dir=${data_dir}/lmdb \
    model=vilt \
    model.ckpt_path=csarron/vilt-ve-ft \
    gradient_accumulation_steps=1 \
    per_device_dev_batch_size=32 \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.is_clip=False

# eval pumer vilt-ve
torchrun --nproc_per_node 4 cli/run.py \
    task=ve \
    data_dir=${data_dir} \
    task.image_db_dir=${data_dir}/lmdb \
    model=vilt \
    model.ckpt_path=csarron/vilt-ve-p0.1r0.3t0.2-2468 \
    gradient_accumulation_steps=1 \
    per_device_dev_batch_size=32 \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.is_clip=False


# predict vilt-vqa2, need to submit predictions to vqa server at
# https://eval.ai/web/challenges/challenge-page/830/submission

torchrun --nproc_per_node 4 cli/run.py \
    task=vqa2 \
    data_dir=${data_dir} \
    task.image_db_dir=${data_dir}/lmdb \
    model=vilt \
    model.ckpt_path=csarron/vilt-vqa2-ft \
    gradient_accumulation_steps=1 \
    per_device_dev_batch_size=32 \
    do_predict=True \
    do_eval_predict=False \
    task.image_size=384 \
    task.is_clip=False


# eval meter-ve-pumer
torchrun --nproc_per_node ${n_gpus} cli/run.py \
    task=ve \
    data_dir=${data_dir} \
    task.image_db_dir=${data_dir}/lmdb \
    model=meter \
    model.ckpt_path=csarron/meter-ve-p0.3r0.5t0.2-0246 \
    do_eval=True \
    per_device_dev_batch_size=4 \
    do_step_eval=True \
    do_predict=True \
    do_eval_predict=True \
    task.image_size=384 \
    task.tokenizer=roberta-base

```
