from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

from ..common import ITMHead, MLMHead, MPPHead, Pooler
from ..pruner import TokenPruner
from .configuration_vilt import ViltConfig
from .vit_vilt import _create_vision_transformer


@dataclass
class ViltModelOutput(ModelOutput):
    last_hidden_states: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    text_feats: Optional[Tuple[torch.FloatTensor]] = None
    image_feats: Optional[Tuple[torch.FloatTensor]] = None
    cls_feats: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    previous_keep_masks: Optional[Tuple[torch.FloatTensor]] = None
    pruner_info: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


class ViltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViltConfig
    base_model_prefix = "Vilt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, VisionTransformer):
            module.gradient_checkpointing = value


class ViltModel(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )
        self.text_embeddings = BertEmbeddings(self.bert_config)
        self.token_type_embeddings = nn.Embedding(config.token_types, config.hidden_size)

        # self.transformer = timm.create_model(config.vit, num_classes=0)  # pop vit head
        model_kwargs = dict(
            patch_size=32, embed_dim=config.hidden_size, depth=config.num_layers, num_heads=config.num_heads
        )
        self.transformer = _create_vision_transformer("vit_base_patch32_384", pretrained=False, **model_kwargs)

        self.pooler = Pooler(config.hidden_size)

        self.token_pruner = TokenPruner(config)

        self.merge_ratio = config.merge_ratio
        self.keep_ratio = config.keep_ratio
        self.merge_r = config.merge_r
        self.prune_r = config.prune_r
        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        text_ids=None,
        text_masks=None,
        pixel_values=None,
        image_token_type_idx=1,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_embeds = self.text_embeddings(text_ids)
        image_embeds, image_masks, _, _ = self.transformer.visual_embed(
            pixel_values,
            max_image_len=-1,
            mask_it=False,
        )

        text_len = text_embeds.shape[1]
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        attentions = ()
        text_feats = ()
        image_feats = ()
        previous_keep_masks = ()
        all_hidden_states = ()
        all_keep_info = ()

        batch_size = image_embeds.shape[0]
        image_len = image_embeds.shape[1] - 1  # -1 for img cls token
        previous_keep_mask = torch.ones(
            batch_size, image_len, dtype=image_embeds.dtype, device=image_embeds.device
        )  # to propagate the masks to subsequent layers

        kwargs["sim_method"] = self.config.sim_method
        kwargs["merge_style"] = self.config.merge_style
        kwargs["merge_r"] = self.config.merge_r
        for layer_idx, blk in enumerate(self.transformer.blocks):
            # print(f"{layer_idx=}, {x.shape=}, {text_len=}")
            kwargs["text_len"] = text_len
            kwargs["text_masks"] = text_masks
            if self.config.reduce_layers and layer_idx in self.config.reduce_layers:
                kwargs["merge_text"] = self.config.merge_text
                kwargs["merge_r"] = self.config.merge_r
                kwargs["prune_r"] = self.config.prune_r
                x, attn, co_masks, text_len = blk(x, mask=co_masks, **kwargs)
                text_masks = co_masks[:, :text_len]
            else:
                if self.config.reduce_layers:  # reset merge_r for reduce case
                    kwargs["merge_text"] = 0
                    kwargs["merge_r"] = 0
                    kwargs["prune_r"] = 0
                x, attn, co_masks, text_len = blk(x, mask=co_masks, **kwargs)
                # text_masks = co_masks[:, :text_len]
            if self.config.prune_layers and layer_idx in self.config.prune_layers:
                # print(f"prune {layer_idx=}")
                # adapt below to dynamicvit version
                # breakpoint()
                cross_attn = attn[:, :, :text_len, text_len + 1 :]
                text_states = x[:, :text_len]
                image_states = x[:, text_len:]
                image_states, image_masks, previous_keep_mask, layer_keep_info = self.token_pruner(
                    layer_idx,
                    text_states,
                    text_masks,
                    image_states,
                    image_masks,
                    cross_attn,
                    previous_keep_mask,
                    **kwargs,
                )
                x = torch.cat([text_states, image_states], dim=1)
                co_masks = torch.cat([text_masks, image_masks], dim=1)

                text_feats = text_feats + (text_states,)
                image_feats = image_feats + (image_states,)
                attentions = attentions + (cross_attn,)
                previous_keep_masks = previous_keep_masks + (previous_keep_mask,)
                all_hidden_states = all_hidden_states + (x,)
                all_keep_info = all_keep_info + (layer_keep_info,)

        x = self.transformer.norm(x)
        text_feats = text_feats + (x[:, :text_len],)
        image_feats = image_feats + (x[:, text_len:],)

        cls_feats = self.pooler(x)

        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    all_hidden_states,
                    text_feats,
                    image_feats,
                    cls_feats,
                    attentions,
                    previous_keep_masks,
                ]
                if v is not None
            )

        return ViltModelOutput(
            last_hidden_states=x,
            hidden_states=all_hidden_states,
            text_feats=text_feats,
            image_feats=image_feats,
            cls_feats=cls_feats,
            attentions=attentions,
            previous_keep_masks=previous_keep_masks,
            pruner_info=(self.token_pruner.keep_ratios, all_keep_info),
        )


class ViltForPreTraining(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        if config.mlm_loss > 0:
            self.mlm_score = MLMHead(self.vilt.bert_config)

        if config.itm_loss > 0:
            self.itm_score = ITMHead(config.hidden_size)

        if config.mpp_loss > 0:
            self.mpp_score = MPPHead(self.vilt.bert_config)
        # Initialize weights and apply final processing
        self.init_weights()


class ViltForQuestionAnswering(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        hidden_size = config.hidden_size
        self.num_labels = config.vqa_label_size
        self.vqa_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, self.num_labels),
        )
        self.layer_keep_info = None
        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        text_ids=None,
        text_masks=None,
        pixel_values=None,
        image_token_type_idx=1,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            text_ids,
            text_masks,
            pixel_values,
            image_token_type_idx,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )
        pool_feats = outputs.cls_feats
        vqa_logits = self.vqa_classifier(pool_feats)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(vqa_logits.view(-1, self.num_labels), labels)

        return ViltModelOutput(
            logits=vqa_logits,
            loss=loss,
            last_hidden_states=outputs.last_hidden_states,
            hidden_states=outputs.hidden_states,
            text_feats=outputs.text_feats,
            image_feats=outputs.image_feats,
            cls_feats=outputs.cls_feats,
            attentions=outputs.attentions,
            previous_keep_masks=outputs.previous_keep_masks,
            pruner_info=outputs.pruner_info,
        )


class ViltForVisualReasoning(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vilt = ViltModel(config)
        hidden_size = config.hidden_size
        self.nlvr2_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, 2),
        )

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        text_ids=None,
        text_masks=None,
        pixel_values1=None,
        pixel_values2=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.vilt(
            text_ids,
            text_masks,
            pixel_values1,
            1,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )
        keep_ratio, keep_info1 = outputs1.pruner_info
        outputs2 = self.vilt(
            text_ids,
            text_masks,
            pixel_values2,
            2,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )

        keep_ratio, keep_info2 = outputs2.pruner_info

        pool_feats1 = outputs1.cls_feats
        pool_feats2 = outputs2.cls_feats
        cls_feats = torch.cat([pool_feats1, pool_feats2], dim=-1)

        nlvr_logits = self.nlvr2_classifier(cls_feats)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(nlvr_logits, labels)

        return ViltModelOutput(
            logits=nlvr_logits,
            loss=loss,
            last_hidden_states=(outputs1.last_hidden_states, outputs2.last_hidden_states),
            hidden_states=(outputs1.hidden_states, outputs2.hidden_states),
            cls_feats=cls_feats,
            attentions=(outputs1.attentions, outputs2.attentions),
            previous_keep_masks=(outputs1.previous_keep_masks, outputs2.previous_keep_masks),
            pruner_info=(keep_ratio, keep_info1, keep_info2),
        )


class ViltForVisualEntailment(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vilt = ViltModel(config)
        hidden_size = config.hidden_size
        self.ve_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, 3),
        )
        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        text_ids=None,
        text_masks=None,
        pixel_values=None,
        image_token_type_idx=1,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            text_ids,
            text_masks,
            pixel_values,
            image_token_type_idx,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )
        pool_feats = outputs.cls_feats
        ve_logits = self.ve_classifier(pool_feats)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(ve_logits, labels)

        return ViltModelOutput(
            logits=ve_logits,
            loss=loss,
            last_hidden_states=outputs.last_hidden_states,
            hidden_states=outputs.hidden_states,
            text_feats=outputs.text_feats,
            image_feats=outputs.image_feats,
            cls_feats=outputs.cls_feats,
            attentions=outputs.attentions,
            previous_keep_masks=outputs.previous_keep_masks,
            pruner_info=outputs.pruner_info,
        )


class ViltForImageAndTextRetrieval(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vilt = ViltModel(config)
        self.itm_score = ITMHead(config.hidden_size)
        # Classifier head
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        text_ids=None,
        text_masks=None,
        pixel_values=None,
        image_token_type_idx=1,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_shape = pixel_values.shape
        batch_size = image_shape[0]
        if len(image_shape) == 5:
            # for training
            _, num_pairs, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_pairs, c, h, w)
            text_ids = text_ids.reshape(batch_size * num_pairs, -1)
            text_masks = text_masks.reshape(batch_size * num_pairs, -1)

            # itm labels
            itm_labels = torch.tensor([1] + [0] * (num_pairs - 1)).repeat(batch_size).to(text_ids).long()

        else:
            num_pairs = None
            itm_labels = None
            assert len(image_shape) == 4  # for inference

        outputs = self.vilt(
            text_ids,
            text_masks,
            pixel_values,
            image_token_type_idx,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )
        # pooler_output = outputs.cls_feats if return_dict else outputs[1]
        logits = self.rank_output(outputs.cls_feats)

        loss = None
        if num_pairs is not None:
            score = logits[:, 0].reshape(batch_size, -1)
            answer = torch.zeros(batch_size).to(score).long()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(score, answer)

        itm_loss = None
        if itm_labels is not None:
            itm_logits = self.itm_score(outputs.cls_feats)
            loss_fn = nn.CrossEntropyLoss()
            itm_loss = loss_fn(itm_logits, itm_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ViltModelOutput(
            logits=logits,
            loss=(loss, itm_loss),
            last_hidden_states=outputs.last_hidden_states,
            hidden_states=outputs.hidden_states,
            text_feats=outputs.text_feats,
            image_feats=outputs.image_feats,
            cls_feats=outputs.cls_feats,
            attentions=outputs.attentions,
            previous_keep_masks=outputs.previous_keep_masks,
            pruner_info=outputs.pruner_info,
        )
