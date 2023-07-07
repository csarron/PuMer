import abc
import math

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from ..metric import get_metric
from ..utils.common import _kl_bernoulli_bernoulli, get_neighbor_indices


class Task(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config, device):
        self.config = config
        self.metric_class = get_metric(config.task.name)
        self.init_metrics(device)

    def __call__(self, outputs, batch, gather_fn, *args, **kwargs):
        is_training = kwargs.pop("is_training", True)
        gradcam_masks = kwargs.pop("gradcam_masks", None)
        # global_step = kwargs.pop("global_step", 0)
        preds = outputs.logits.argmax(dim=-1)
        preds = gather_fn(preds)
        score = None
        labels = batch.get("labels", None)
        if labels is not None:
            labels = gather_fn(labels)
            score = self.metric(preds, labels)

        task_loss = outputs.loss
        loss = None
        losses = {}
        if task_loss is not None:
            loss = self.config.task.task_loss_scale * task_loss
            losses["loss_task"] = task_loss

        if is_training and self.config.task.prune_loss_scale > 0:
            mask_loss = self.compute_prune_loss(outputs, gradcam_masks)
            prune_loss, cnt_loss, grdcm_loss = mask_loss

            prune_loss_total = 0
            for layer, p_loss in prune_loss.items():
                losses[f"loss_prune/{layer}"] = p_loss
                prune_loss_total += p_loss
            prune_loss_total /= len(prune_loss)
            prune_scaled_loss = self.config.task.prune_loss_scale * prune_loss_total
            losses["loss_prune"] = prune_loss_total
            loss = loss + prune_scaled_loss

            if self.config.task.grdcm_loss_scale > 0:
                grdcm_loss_total = 0
                for layer, g_loss in grdcm_loss.items():
                    losses[f"loss_grdcm/{layer}"] = g_loss
                    grdcm_loss_total += g_loss
                grdcm_loss_total /= len(grdcm_loss)
                grdcm_scaled_loss = self.config.task.grdcm_loss_scale * grdcm_loss_total
                losses["loss_grdcm"] = grdcm_loss_total
                loss = loss + grdcm_scaled_loss

            if self.config.task.cnt_loss_scale > 0:
                cnt_loss_total = 0
                for layer, c_loss in cnt_loss.items():
                    losses[f"loss_cnt/{layer}"] = c_loss
                    cnt_loss_total += c_loss
                cnt_loss_total /= len(cnt_loss)
                cnt_scaled_loss = self.config.task.cnt_loss_scale * cnt_loss_total
                losses["loss_cnt"] = cnt_loss_total
                # assert loss is not None
                loss = loss + cnt_scaled_loss

        if is_training and self.config.task.text_loss_scale > 0:
            text_loss = self.compute_text_loss(outputs, gradcam_masks)
            text_loss_total = 0
            for layer, t_loss in text_loss.items():
                losses[f"loss_text/{layer}"] = t_loss
                text_loss_total += t_loss
            text_loss_total /= len(text_loss)
            scaled_text_loss = self.config.task.text_loss_scale * text_loss_total
            losses["loss_text"] = text_loss_total
            loss = loss + scaled_text_loss

        if is_training and self.config.task.attn_loss_scale > 0:
            attn_loss = self.compute_attn_loss(outputs, gradcam_masks)
            attn_loss_total = 0
            for layer, a_loss in attn_loss.items():
                losses[f"loss_attn/{layer}"] = a_loss
                attn_loss_total += a_loss
            attn_loss_total /= len(attn_loss)
            scaled_attn_loss = self.config.task.attn_loss_scale * attn_loss_total
            losses["loss_attn"] = attn_loss_total
            loss = loss + scaled_attn_loss

        teacher_outputs = kwargs.pop("teacher_outputs", None)
        if teacher_outputs is not None:
            # kl div of logits,
            logits_loss, states_loss = self.compute_teacher_loss(outputs, teacher_outputs)
            losses["loss_logits"] = logits_loss
            scaled_logits_loss = self.config.task.logits_loss_scale * logits_loss
            loss = loss + scaled_logits_loss

            if self.config.task.states_loss_scale > 0:
                states_loss_total = 0
                for layer, s_loss in states_loss.items():
                    losses[f"loss_states/{layer}"] = s_loss
                    states_loss_total += s_loss
                states_loss_total /= len(states_loss)
                losses["loss_states"] = states_loss_total
                scaled_states_loss = self.config.task.states_loss_scale * states_loss_total
                loss = loss + scaled_states_loss

        losses["loss"] = loss
        predictions = {
            "prediction": preds,
        }

        scores = {
            "metric": score,
        }

        return predictions, losses, scores

    def init_metrics(self, device):
        self.metric = self.metric_class().to(device)

    def compute_prune_loss(self, outputs, gradcam_masks=None):
        prune_loss = {}
        cnt_loss = {}
        grdcm_loss = {}
        keep_masks = outputs.previous_keep_masks
        keep_ratios, layer_keep_info = outputs.pruner_info
        for prune_layer, keep_ratio in keep_ratios.items():
            keep_mask = keep_masks[prune_layer].float()
            if self.config.task.ib_kl:
                (keep_logits, q_z), _ = layer_keep_info[prune_layer]
                prior = torch.ones_like(keep_logits) * keep_ratio
                # prune_loss += F.kl_div(keep_mask, prior, reduction="batchmean")
                # q_z = Bernoulli(logits=keep_logits)
                p_prior = Bernoulli(probs=prior)
                # prune_loss[prune_layer] = torch.distributions.kl.kl_divergence(q_z, p_prior).mean()
                prune_loss[prune_layer] = _kl_bernoulli_bernoulli(q_z, p_prior).mean()
            else:
                keep_logits, _ = layer_keep_info[prune_layer]
                if self.config.task.prune_no_sparsity:
                    prune_loss[prune_layer] = keep_mask.mean()
                else:
                    prune_loss[prune_layer] = ((keep_mask.mean(1) - keep_ratio) ** 2).mean()

            if self.config.task.cnt_loss_scale > 0:
                size = keep_mask.shape[-1]
                h = w = int(math.sqrt(size))  # assume h=w, otherwise need to get height and width info
                total_diff = 0
                for idx, neighbor_indices in get_neighbor_indices(h, w):
                    diff = (keep_mask[:, [idx]] - keep_mask[:, neighbor_indices]) ** 2
                    total_diff += diff.float().mean(1)
                cnt_loss[prune_layer] = (total_diff / size).mean()

            if gradcam_masks is not None and self.config.task.grdcm_loss_scale > 0:
                true_mask = gradcam_masks > 0
                keep_mask_logits = keep_logits[true_mask].sigmoid()
                if self.config.task.grdcm_bce:
                    grdcm_loss[prune_layer] = F.binary_cross_entropy(keep_mask_logits, gradcam_masks[true_mask])
                else:
                    grdcm_loss[prune_layer] = ((keep_mask_logits - gradcam_masks[true_mask]) ** 2).mean()
                # if self.config.task.gradcam_high_recall:
                #     # supervise only gradcam masks, encourage high recall
                #     true_mask = gradcam_masks > 0
                #     grdcm_loss[prune_layer] = ((keep_mask[true_mask] - gradcam_masks[true_mask]) ** 2).mean()
                # else:
                #     # supervise all tokens
                #     grdcm_loss[prune_layer] = ((keep_mask - gradcam_masks) ** 2).mean()

        return prune_loss, cnt_loss, grdcm_loss

    def compute_logit_loss(self, logits, teacher_logits):
        logits_loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(teacher_logits, dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        return logits_loss

    def compute_text_loss(self, outputs, gradcam_masks=None):
        text_loss = {}
        _, layer_keep_info = outputs.pruner_info
        for i, layer in enumerate(self.config.model.config.prune_layers, 1):
            keep_logits, tv_sim = layer_keep_info[layer]
            # cosine_sim = compute_cosine_sim(text_states, image_states)  # [B, T, V]
            # max, mean of text tokens. for mean, we may handle text masks? pad tokens should have low sim, so it should not affect max
            # v_sim = cosine_sim.max(1)[0]  # [B, V]
            # v_sim = tv_sim.max(1)[0]  # [B, V]

            # if gradcam_masks is not None and self.config.task.grdcm_ctr:
            #     # use gradcam to decide positive and negative examples
            #     # pos_mask = gradcam_masks > 0  # == 1, [B, V]
            #     # neg_mask = gradcam_masks < 1  # == 0, [B, V]
            #     loss = F.cross_entropy(v_sim, gradcam_masks)
            # else:
            #     # pos_mask = keep_masks > 0
            #     # neg_mask = keep_masks < 1
            #     loss = F.cross_entropy(v_sim, keep_masks)

            batch_size_all = tv_sim.shape[0]
            itc_labels = torch.arange(batch_size_all, device=tv_sim.device)
            loss_i2t = F.cross_entropy(tv_sim, itc_labels)
            loss_t2i = F.cross_entropy(tv_sim.t(), itc_labels)
            loss = (loss_i2t + loss_t2i) / 2
            text_loss[layer] = loss
        return text_loss

    def compute_attn_loss(self, outputs, gradcam_masks=None):
        attentions = outputs.attentions
        previous_keep_masks = outputs.previous_keep_masks
        # all are tuples, only handle pruner_layer states
        attn_loss = {}
        for i, layer in enumerate(self.config.model.config.prune_layers, 1):
            cross_attn = attentions[layer]  # [B, H, T, V]
            keep_masks = previous_keep_masks[layer]  # [B, V]
            txt2img_attnm = cross_attn.mean(1).sum(1)
            # encourage large attn for keep_masks tokens, smaller attn for removed (1-keep_masks) tokens

            keep_attn = txt2img_attnm[keep_masks > 0.0].mean()
            removed_attn = txt2img_attnm[keep_masks < 1.0].mean()
            # txt2img_loss = F.cross_entropy(txt2img_attnm, keep_masks)
            # txt2img_loss = 1.0 - (txt2img_attnm * keep_masks).mean()
            attn_loss[layer] = 1.0 - keep_attn + removed_attn
        return attn_loss

    def compute_states_loss(self, states, token_masks, teacher_states):
        batch_size, seq_len, hidden_size = states.shape
        # use token_masks to only supervise unmasked tokens
        img_seq_len = token_masks.shape[1]
        text_len = seq_len - img_seq_len

        bool_mask = token_masks.reshape(-1) > 0.5  #

        text_states = states[:, :text_len]
        image_states = states[:, text_len:]
        # print(f"{seq_len=}, {text_len+1=},")
        img_states = image_states.reshape(-1, hidden_size)[bool_mask]

        t_text_states = teacher_states[:, :text_len]
        teacher_image_states = teacher_states[:, text_len:]
        t_img_states = teacher_image_states.reshape(-1, hidden_size)[bool_mask]

        # mse loss
        img_states_loss = F.mse_loss(t_img_states, img_states)
        txt_states_loss = F.mse_loss(t_text_states, text_states)
        states_loss = (img_states_loss + txt_states_loss) / 2
        return states_loss

    def compute_teacher_loss(self, outputs, teacher_outputs):
        # kl div of logits,
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        logits_loss = self.compute_logit_loss(logits, teacher_logits)

        states_loss = {}
        if self.config.task.states_loss_scale > 0:
            for layer in self.config.model.config.prune_layers:
                # mse loss of token states
                states = outputs.hidden_states[layer]
                # use token_masks to only supervise unmasked tokens
                token_masks = outputs.previous_keep_masks[layer]
                teacher_states = teacher_outputs.hidden_states[layer]

                states_loss[layer] = self.compute_states_loss(states, token_masks, teacher_states)
        return logits_loss, states_loss

    def reset_metric(self):
        score = self.metric.compute()
        self.metric.reset()
        return {"metric": score}
