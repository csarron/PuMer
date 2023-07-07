import json
import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x


def load_labels_into_config(config):
    if config.get("label2id_file", None):
        label2id_file = Path(config.label2id_file)
        if label2id_file.exists():
            config.label2id = json.load(open(config.label2id_file))

    if config.get("label2id_file", None):
        id2label_file = Path(config.id2label_file)
        if id2label_file.exists():
            config.id2label = {i: k for i, k in enumerate(json.load(open(config.id2label_file)))}
    return config


def compute_cosine_sim(t, v):
    """Compute cosine similarity across every pairs of t, v (batched)
    [B, L_t, D] [B, L_v, D] -> [B, L_t, L_v]"""
    t_norm = F.normalize(t, p=2, dim=-1)
    v_norm = F.normalize(v, p=2, dim=-1)
    cosine_sim = t_norm.matmul(v_norm.transpose(1, 2))
    # cosine_dist = 1 - cosine_sim
    return cosine_sim


def differentiable_masked_softmax(attn, mask, eps=1e-6):
    B, N = mask.size()
    B, H, N1, N = attn.size()
    attn_mask = mask.reshape(B, 1, 1, N)
    eye = torch.eye(N1, N, dtype=attn_mask.dtype, device=attn_mask.device).view(1, 1, N1, N)
    attn_mask = attn_mask + (1.0 - attn_mask) * eye
    # breakpoint()
    max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    attn = attn - max_att
    # for stable training
    attn = attn.to(torch.float32).exp_() * attn_mask.to(torch.float32)
    attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
    return attn.type_as(max_att)


def get_negative_idx(j, bs):
    return random.choice([i for i in range(bs) if i != j])


def get_itm_labels_and_index(batch_size, device):
    half_size = batch_size // 2
    itm_labels = [0.0] * half_size + [1.0] * (batch_size - half_size)
    random.shuffle(itm_labels)

    new_idx = [i if v == 1 else get_negative_idx(i, batch_size) for i, v in enumerate(itm_labels)]
    itm_labels = torch.tensor(itm_labels, device=device)
    itm_idx = torch.tensor(new_idx, device=device)
    return itm_labels, itm_idx


# modified from https://github.com/facebookresearch/mae/blob/6a2ba402291005b003a70e99f7c87d1a2c376b0d/models_mae.py#L123
def random_mask(x, mask_ratio):
    N, L, D = x.shape  # batch, length, dim
    len_keep = math.ceil(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_masked


def preprocess_ckpt(state_dict):
    # ignore_position_id_in_ckpt
    pattern = "image_encoder.embeddings.position_ids"
    keys = [k for k in state_dict.keys() if pattern in k]
    assert len(keys) == 1, f"{keys=}"
    key = keys[0]
    state_dict.pop(key)
    # print(f'pop {key=} from ckpt....')


def interpolate_position_embedding(
    image_size=224, patch_size=16, pattern="image_encoder.embeddings.position_embedding"
):
    def state_dict_pre_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        preprocess_ckpt(state_dict)

        keys = [k for k in state_dict.keys() if pattern in k]
        assert len(keys) == 1, f"{keys=}"
        key = keys[0]
        old_pos_embed = state_dict[key]  # shape: [197, 768]
        old_num_grids = int(math.sqrt(old_pos_embed.shape[0] - 1))  # 14
        old_img_size = int(old_num_grids * patch_size)  # 224

        if old_img_size == image_size:  # no need to modify
            # print('no need to interpolate pos embd....')
            return state_dict
        # print(f'interpolate pos embd, {key=}, {old_pos_embed.shape}')
        assert (old_img_size % patch_size) == 0
        num_grids = image_size // patch_size
        assert (image_size % patch_size) == 0
        embed_dim = old_pos_embed.shape[-1]  # 768
        pos_embed = old_pos_embed[1:, :].reshape((old_num_grids, old_num_grids, embed_dim))  # [14, 14, 768]
        new_size = (num_grids, num_grids)  # for image_size=384, this is [24, 24]
        new_pos_embed = torch.nn.functional.interpolate(
            pos_embed.permute((2, 0, 1)).unsqueeze(0),
            size=new_size,
            mode="bicubic",
            align_corners=False,
        )  # [1, 768, 24, 24]
        new_pos_embed = (
            new_pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
        )  # [24, 24, 768] => [576, 768]
        new_pos_embed = torch.cat((old_pos_embed[:1, :], new_pos_embed), dim=0)  # [577, 768]
        print(f"interpolate position embeddings, {key=}, {old_pos_embed.shape}, {new_pos_embed.shape}")
        state_dict[key] = new_pos_embed

    return state_dict_pre_hook


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        if is_dist_avail_and_initialized():
            output = [torch.empty_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(output, tensor)
        else:
            output = [tensor]
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )


all_gather = AllGather.apply
