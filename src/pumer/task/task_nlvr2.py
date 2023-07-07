import math

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from ..utils.common import _kl_bernoulli_bernoulli, get_neighbor_indices
from .task_base import Task


class Nlvr2Task(Task):
    def compute_prune_loss(self, outputs, gradcam_masks=None):
        prune_loss = {}
        cnt_loss = {}
        grdcm_loss = {}
        keep_masks1, keep_masks2 = outputs.previous_keep_masks
        keep_ratios, layer_keep_info1, layer_keep_info2 = outputs.pruner_info
        for prune_layer, keep_ratio in keep_ratios.items():
            keep_mask1 = keep_masks1[prune_layer].float()
            keep_mask2 = keep_masks2[prune_layer].float()
            if self.config.task.ib_kl:
                (keep_logits1, q_z1), _ = layer_keep_info1[prune_layer]
                (keep_logits2, q_z2), _ = layer_keep_info2[prune_layer]
                prior = torch.ones_like(keep_logits1) * keep_ratio
                # prune_loss += F.kl_div(keep_mask, prior, reduction="batchmean")
                # q_z = Bernoulli(logits=keep_logits)
                p_prior = Bernoulli(probs=prior)
                # prune_loss[prune_layer] = torch.distributions.kl.kl_divergence(q_z, p_prior).mean()
                prune_loss[prune_layer] = (
                    _kl_bernoulli_bernoulli(q_z1, p_prior).mean() + _kl_bernoulli_bernoulli(q_z2, p_prior).mean()
                ) / 2
            else:
                keep_logits1, _ = layer_keep_info1[prune_layer]
                keep_logits2, _ = layer_keep_info2[prune_layer]
                prune_loss1 = ((keep_mask1.mean(1) - keep_ratio) ** 2).mean()
                prune_loss2 = ((keep_mask2.mean(1) - keep_ratio) ** 2).mean()
                prune_loss[prune_layer] = (prune_loss1 + prune_loss2) / 2
                # prune_loss[prune_layer] = prune_loss1

            if self.config.task.cnt_loss_scale > 0:
                size = keep_mask1.shape[-1]
                h = w = int(math.sqrt(size))  # assume h=w, otherwise need to get height and width info
                total_diff = 0
                for idx, neighbor_indices in get_neighbor_indices(h, w):
                    diff1 = (keep_mask1[:, [idx]] - keep_mask1[:, neighbor_indices]) ** 2
                    diff2 = (keep_mask2[:, [idx]] - keep_mask2[:, neighbor_indices]) ** 2
                    diff = diff1 + diff2
                    total_diff += diff.float().mean(1)
                cnt_loss[prune_layer] = (total_diff / size).mean()

            if gradcam_masks is not None and self.config.task.grdcm_loss_scale > 0:
                # FIXME: should have two gradcam_masks
                true_mask = gradcam_masks > 0
                keep_mask_logits1 = keep_logits1[true_mask].sigmoid()
                keep_mask_logits2 = keep_logits2[true_mask].sigmoid()
                if self.config.task.grdcm_bce:
                    grdcm_loss1 = F.binary_cross_entropy(keep_mask_logits1, gradcam_masks[true_mask])
                    grdcm_loss2 = F.binary_cross_entropy(keep_mask_logits2, gradcam_masks[true_mask])
                else:
                    grdcm_loss1 = ((keep_mask_logits1 - gradcam_masks[true_mask]) ** 2).mean()
                    grdcm_loss2 = ((keep_mask_logits2 - gradcam_masks[true_mask]) ** 2).mean()
                grdcm_loss[prune_layer] = (grdcm_loss1 + grdcm_loss2) / 2

        return prune_loss, cnt_loss, grdcm_loss

    def compute_text_loss(self, outputs, gradcam_masks=None):
        text_loss = {}
        keep_ratios, layer_keep_info1, layer_keep_info2 = outputs.pruner_info
        for i, layer in enumerate(self.config.model.config.prune_layers, 1):
            keep_logits1, tv_sim1 = layer_keep_info1[layer]
            keep_logits2, tv_sim2 = layer_keep_info2[layer]

            batch_size_all = tv_sim1.shape[0]
            itc_labels = torch.arange(batch_size_all, device=tv_sim1.device)

            loss_i2t1 = F.cross_entropy(tv_sim1, itc_labels)
            loss_t2i1 = F.cross_entropy(tv_sim1.t(), itc_labels)
            loss1 = (loss_i2t1 + loss_t2i1) / 2

            loss_i2t2 = F.cross_entropy(tv_sim2, itc_labels)
            loss_t2i2 = F.cross_entropy(tv_sim2.t(), itc_labels)
            loss2 = (loss_i2t2 + loss_t2i2) / 2

            text_loss[layer] = (loss1 + loss2) / 2
        return text_loss

    def compute_teacher_loss(self, outputs, teacher_outputs):
        # kl div of logits,
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        logits_loss = self.compute_logit_loss(logits, teacher_logits)

        states_loss = {}
        if self.config.task.states_loss_scale > 0:
            for layer in self.config.model.config.prune_layers:
                states1, states2 = outputs.hidden_states
                # use token_masks to only supervise unmasked tokens
                token_masks1, token_masks2 = outputs.previous_keep_masks
                teacher_states1, teacher_states2 = teacher_outputs.hidden_states
                states_loss1 = self.compute_states_loss(states1, token_masks1, teacher_states1)
                states_loss2 = self.compute_states_loss(states2, token_masks2, teacher_states2)
                states_loss = states_loss1 + states_loss2
                states_loss[layer] = states_loss / 2

        return logits_loss, states_loss
