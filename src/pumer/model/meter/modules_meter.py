import math

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from ..common import differentiable_masked_softmax
from ..reducer import bipartite_soft_matching, merge_wavg

# from transformers.models.bert.modeling_bert import (
#     BertAttention,
#     BertIntermediate,
#     BertOutput,
# )


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # print(f"{attention_scores.shape=}, {attention_mask.shape=}")
            attention_probs = differentiable_masked_softmax(attention_scores, attention_mask)
            # # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # attention_scores = attention_scores + attention_mask
        else:
            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # if True:
        # if False:
        #     self.save_attention_map(attention_probs)
        #     attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        outputs += (key_layer.mean(1),)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# modified from from transformers.models.bert.modeling_bert import BertLayer, "use self.crossattention without decoder logic"
class BertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None  # past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        txt_metric = self_attention_outputs[-1]

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # extract t2img attn, add prune and merge
        txt2img_attn = cross_attention_outputs[1]  # size: [bs, num_heads, text_len, image_len]
        img_metric = cross_attention_outputs[-1]

        batch_size, image_length, hidden_size = encoder_hidden_states.shape
        text_length = hidden_states.shape[1]

        prune_ratio = kwargs.pop("prune_r", 0)
        merge_ratio = kwargs.pop("merge_r", 0)
        text_states = attention_output
        image_states = encoder_hidden_states
        text_masks = attention_mask
        image_masks = encoder_attention_mask
        if prune_ratio > 0:
            prune_r = round(prune_ratio * int(image_length))
            merge_r = round(merge_ratio * int(image_length))
            keep_r = max(image_length - prune_r, 1)
            t_len = text_masks.sum(1, keepdim=True)

            # cross attention pruning mh
            attn_scores = txt2img_attn.mean(1).sum(1) / t_len
            top_val, top_idx = torch.topk(attn_scores, keep_r, dim=-1)
            t_idx = top_idx.unsqueeze(2).expand(batch_size, keep_r, hidden_size)
            new_image_states = image_states.gather(1, t_idx)

            # merge tome
            if merge_r > 0:
                tm_idx = top_idx.unsqueeze(2).expand(batch_size, keep_r, img_metric.shape[-1])
                new_img_metric = img_metric.gather(1, tm_idx)
                merge_fn = bipartite_soft_matching(new_img_metric, merge_r)
                image_states, _ = merge_wavg(merge_fn, new_image_states)

            image_masks = torch.ones(
                (batch_size, image_states.shape[1]),
                dtype=torch.long,
                device=image_states.device,
            )
            # merge text
            text_mr = kwargs.get("merge_text", 0)
            text_r = round(int(text_length) * text_mr)
            if text_r > 0:
                merge_fn = bipartite_soft_matching(txt_metric, text_r)
                text_states, _ = merge_wavg(merge_fn, text_states)
                mask_m = text_masks[..., None]
                mask_m, _ = merge_wavg(merge_fn, mask_m)
                text_masks = mask_m[..., 0]

                text_length = text_states.shape[1]

        # else:
        #     merge_r = int(merge_ratio)
        #     # tome merging
        #     if merge_r > 0:
        #         # print(f"{merge_r=}")
        #         merge_fn = bipartite_soft_matching(metric, merge_r)

        attention_output = text_states

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        # outputs = (layer_output,) + outputs

        # return outputs
        return layer_output, text_masks, image_states, image_masks, txt2img_attn

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
