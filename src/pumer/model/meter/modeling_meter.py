from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoConfig, AutoModel, RobertaConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from ..common import ITMHead, MLMHead, MPPHead, Pooler
from ..pruner import TokenPruner
from .configuration_meter import MeterConfig
from .modules_meter import BertCrossLayer
from .vit_meter import build_clip


@dataclass
class MeterModelOutput(ModelOutput):
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


class MeterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MeterConfig
    base_model_prefix = "meter"
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

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, MeterEncoder):
    #         module.gradient_checkpointing = value


class MeterModel(MeterPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        vit_patch_size = config.vit_patch_size
        image_size = config.image_size
        self.vit_model = build_clip(vit_patch_size, image_size)

        self.bert_config = RobertaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )

        self.cross_modal_text_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.cross_modal_image_transform = nn.Linear(config.hidden_size, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(config.token_types, config.hidden_size)
        auto_config = AutoConfig.from_pretrained(config.tokenizer)
        self.text_transformer = AutoModel.from_config(
            auto_config, add_pooling_layer=False
        )  #  tokenizer = "bert-base-uncased", roberta-base

        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(self.bert_config) for _ in range(config.num_top_layer)]
        )
        self.cross_modal_text_layers = nn.ModuleList(
            [BertCrossLayer(self.bert_config) for _ in range(config.num_top_layer)]
        )
        self.cross_modal_image_pooler = Pooler(config.hidden_size)
        self.cross_modal_text_pooler = Pooler(config.hidden_size)
        self.token_pruner = TokenPruner(config)

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

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        text_len = input_shape[1]
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        image_embeds = self.vit_model(pixel_values)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones(
            (image_embeds.size(0), image_embeds.size(1)),
            dtype=torch.long,
            device=device,
        )
        # extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        batch_size = image_embeds.shape[0]
        image_len = image_embeds.shape[1] - 1  # -1 for img cls token
        previous_keep_mask = torch.ones(
            batch_size, image_len, dtype=image_embeds.dtype, device=image_embeds.device
        )  # to propagate the masks to subsequent layers

        attentions = ()
        previous_keep_masks = ()
        all_hidden_states = ()
        all_keep_info = (None,)

        text_feats = ()
        image_feats = ()
        x, y = text_embeds, image_embeds
        layer_idx = 0
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            # TODO: add pruner_r and merger_r args
            if self.config.reduce_layers and layer_idx in self.config.reduce_layers:
                kwargs["merge_text"] = self.config.merge_text
                kwargs["merge_r"] = self.config.merge_r
                kwargs["prune_r"] = self.config.prune_r
            else:  # reset merge_r for reduce case
                kwargs["merge_text"] = 0
                kwargs["merge_r"] = 0
                kwargs["prune_r"] = 0

            # print(f"{layer_idx=}, {y.shape=}, {x.shape=}")
            x1 = text_layer(x, y, text_masks, image_masks, output_attentions=True, **kwargs)
            x, text_masks, y, image_masks, attn = x1
            kwargs["merge_text"] = 0
            kwargs["merge_r"] = 0
            kwargs["prune_r"] = 0
            y1 = image_layer(y, x, image_masks, text_masks, output_attentions=True, **kwargs)

            layer_idx += 1
            layer_idx += 1
            # attentions = attentions + ((x1[1], x1[2], y1[1], y1[2]),)
            # print(f"be {y.shape=}")
            y = y1[0]
            # print(f"af {y.shape=}")
            text_feats = text_feats + (x,)
            image_feats = image_feats + (y,)
            previous_keep_masks = previous_keep_masks + (previous_keep_mask,)
            # attn = x1[-1][:, :, :, 1:]  # text2image cross attention, [B, 12, text_len, image_len]
            attentions = attentions + (attn,)
            # all_hidden_states = all_hidden_states + (hidden_states,)
            all_keep_info = all_keep_info + (all_keep_info[-1],)
            if self.config.prune_layers and layer_idx in self.config.prune_layers:
                y, image_masks, previous_keep_mask, layer_keep_info = self.token_pruner(
                    layer_idx,
                    x,
                    text_masks,
                    y,
                    image_masks,
                    attn,
                    previous_keep_mask,
                    **kwargs,
                )
                previous_keep_masks = previous_keep_masks + (previous_keep_mask,)
                text_feats = text_feats + (x,)
                image_feats = image_feats + (y,)
                attentions = attentions + (attn,)
                all_keep_info = all_keep_info + (layer_keep_info,)

        # text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        cls_feats_image = self.cross_modal_image_pooler(y)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        if not return_dict:
            return tuple(
                v
                for v in [
                    (x, y),
                    all_hidden_states,
                    text_feats,
                    image_feats,
                    cls_feats,
                    attentions,
                    previous_keep_masks,
                ]
                if v is not None
            )

        return MeterModelOutput(
            last_hidden_states=(x, y),
            hidden_states=text_feats,
            text_feats=text_feats,
            image_feats=image_feats,
            cls_feats=cls_feats,
            attentions=attentions,
            previous_keep_masks=previous_keep_masks,
            pruner_info=(self.token_pruner.keep_ratios, all_keep_info),
        )


class MeterForPreTraining(MeterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.meter = MeterModel(config)
        if config.mlm_loss > 0:
            self.mlm_score = MLMHead(self.meter.bert_config)

        if config.itm_loss > 0:
            self.itm_score = ITMHead(config.hidden_size * 2)

        if config.mpp_loss > 0:
            self.mpp_score = MPPHead(self.meter.bert_config)
        # Initialize weights and apply final processing
        self.init_weights()


class MeterForQuestionAnswering(MeterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.meter = MeterModel(config)
        hidden_size = config.hidden_size
        self.num_labels = config.vqa_label_size
        self.vqa_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, self.num_labels),
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

        outputs = self.meter(
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

        return MeterModelOutput(
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


class MeterForVisualReasoning(MeterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.meter = MeterModel(config)
        hidden_size = config.hidden_size
        self.nlvr2_classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
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
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = self.meter(
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

        outputs2 = self.meter(
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

        return MeterModelOutput(
            logits=nlvr_logits,
            loss=loss,
            last_hidden_states=(outputs1.last_hidden_states, outputs2.last_hidden_states),
            hidden_states=(outputs1.hidden_states, outputs2.hidden_states),
            cls_feats=cls_feats,
            attentions=(outputs1.attentions, outputs2.attentions),
            previous_keep_masks=(outputs1.previous_keep_masks, outputs2.previous_keep_masks),
            pruner_info=(keep_ratio, keep_info1, keep_info2),
        )


class MeterForVisualEntailment(MeterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.meter = MeterModel(config)
        hidden_size = config.hidden_size
        self.ve_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
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

        outputs = self.meter(
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

        return MeterModelOutput(
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


class MeterForImageAndTextRetrieval(MeterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.meter = MeterModel(config)
        self.itm_score = ITMHead(config.hidden_size * 2)
        self.rank_output = nn.Linear(config.hidden_size * 2, 1)
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

        outputs = self.meter(
            text_ids,
            text_masks,
            pixel_values,
            image_token_type_idx,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )
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

        return MeterModelOutput(
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
