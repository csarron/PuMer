import torch

from .task_base import Task


class RetrievalTask(Task):
    def __init__(self, config=None, device=None):
        self.config = config
        self.rank_scores = []
        self.image_indices = []
        self.text_image_indices = []
        self.is_training = False

    def __call__(self, outputs, batch, gather_fn, *args, **kwargs):
        is_training = kwargs.pop("is_training", True)
        gradcam_masks = kwargs.pop("gradcam_masks", None)
        # global_step = kwargs.pop("global_step", 0)
        self.is_training = is_training
        preds = outputs.logits[:, 0]
        preds = gather_fn(preds)
        if not is_training:
            image_idx = batch["image_idx"]
            image_idx = gather_fn(image_idx)
            image_idx = image_idx[image_idx >= 0]
            self.image_indices.extend(image_idx.cpu().tolist())

            text_image_idx = batch["text_image_idx"]
            text_image_idx = gather_fn(text_image_idx)
            text_image_idx = text_image_idx[text_image_idx >= 0]
            self.text_image_indices.extend(text_image_idx.cpu().tolist())

            self.rank_scores.extend(preds.cpu().tolist())

        task_loss, itm_loss = outputs.loss
        loss = None
        losses = {}
        if task_loss is not None:
            loss = self.config.task.task_loss_scale * task_loss
            losses["loss_rank"] = task_loss

        if itm_loss is not None:
            scaled_itm_loss = self.config.task.itm_loss_scale * itm_loss
            losses["loss_itm"] = itm_loss
            loss = loss + scaled_itm_loss

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
            "metric": 0,
        }

        return predictions, losses, scores

    def reset_metric(self):
        if self.is_training:
            return 0
        else:
            all_scores = torch.tensor(self.rank_scores)
            iids = torch.tensor(self.image_indices)
            iids = iids.view(-1)
            scores = all_scores.view(len(iids), -1)

            tiids = torch.tensor(self.text_image_indices)
            irtr_metrics = calc_ir_tr_metrics(iids, tiids, scores)
            self.rank_scores = []
            self.image_indices = []
            self.text_image_indices = []
            return irtr_metrics  # ir tr recall metrics


def calc_ir_tr_metrics(iids, tiids, scores):

    topk1 = scores.topk(1, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk10 = scores.topk(10, dim=1)
    topk1_iids = tiids[topk1.indices]
    topk5_iids = tiids[topk5.indices]
    topk10_iids = tiids[topk10.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk1 = scores.topk(1, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk10 = scores.topk(10, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
    return {
        "metric": ir_r1,
        "ir_r1": ir_r1,
        "ir_r5": ir_r5,
        "ir_r10": ir_r10,
        "tr_r1": tr_r1,
        "tr_r5": tr_r5,
        "tr_r10": tr_r10,
    }
