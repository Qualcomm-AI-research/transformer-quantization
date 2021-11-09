# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertSelfAttention,
    BertSelfOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import ModuleUtilsMixin, apply_chunking_to_forward

from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel
from quantization.range_estimators import RangeEstimators, OptMethod
from utils import _tb_advance_global_step, _tb_advance_token_counters, _tb_hist


class QuantizedBertEmbeddings(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = quant_params['quant_dict']

        super().__init__()

        quant_params_ = quant_params.copy()
        if 'Et' in self.quant_dict:
            quant_params_['weight_range_method'] = RangeEstimators.MSE
            quant_params_['weight_range_options'] = dict(opt_method=OptMethod.golden_section)
        self.word_embeddings = quantize_model(org_model.word_embeddings, **quant_params_)

        self.position_embeddings = quantize_model(org_model.position_embeddings, **quant_params)
        self.token_type_embeddings = quantize_model(org_model.token_type_embeddings, **quant_params)

        self.dropout = org_model.dropout

        position_ids = org_model.position_ids
        if position_ids is not None:
            self.register_buffer('position_ids', position_ids)
        else:
            self.position_ids = position_ids

        self.position_embedding_type = getattr(org_model, 'position_embedding_type', 'absolute')

        # Activation quantizers
        self.sum_input_token_type_embd_act_quantizer = QuantizedActivation(**quant_params)
        self.sum_pos_embd_act_quantizer = QuantizedActivation(**quant_params)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.sum_input_token_type_embd_act_quantizer(embeddings)

        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            embeddings = self.sum_pos_embd_act_quantizer(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QuantizedBertSelfAttention(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = quant_params['quant_dict']

        super().__init__()

        # copy attributes
        self.num_attention_heads = org_model.num_attention_heads
        self.attention_head_size = org_model.attention_head_size
        self.all_head_size = org_model.all_head_size

        self.position_embedding_type = getattr(org_model, 'position_embedding_type', None)
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            raise NotImplementedError('current branch of computation is not yet supported')

            self.max_position_embeddings = org_model.max_position_embeddings
            self.distance_embedding = org_model.distance_embedding

        # quantized modules
        self.query = quantize_model(org_model.query, **quant_params)
        self.key = quantize_model(org_model.key, **quant_params)
        self.value = quantize_model(org_model.value, **quant_params)
        self.dropout = org_model.dropout

        # Activation quantizers
        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.context_act_quantizer = QuantizedActivation(**quant_params)

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
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.attn_scores_act_quantizer(attention_scores)

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            raise NotImplementedError('current branch of computation is not yet supported')

            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            # fp16 compatibility:
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores + relative_position_scores_query + relative_position_scores_key
                )

        # NOTE: factor 1/d^0.5 can be absorbed into the previous act. quant. delta
        attention_scores /= math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() fn)
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_probs_act_quantizer(attention_probs)

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
        context_layer = self.context_act_quantizer(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        _tb_advance_global_step(self)
        return outputs


class QuantizedBertSelfOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = quant_params['quant_dict']

        # Exact same structure as for BertOutput.
        # Kept in order to be able to disable activation quantizer.
        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        self.dropout = org_model.dropout

        # Activation quantizer
        self.res_act_quantizer = QuantizedActivation(**quant_params)

        # LN
        self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor
        hidden_states = self.res_act_quantizer(hidden_states)

        hidden_states = self.LayerNorm(hidden_states)

        _tb_advance_global_step(self)
        return hidden_states


class QuantizedBertOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = quant_params['quant_dict']

        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        self.dropout = org_model.dropout
        self.res_act_quantizer = QuantizedActivation(**quant_params)

        # LN
        self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        _tb_advance_token_counters(self, input_tensor)
        _tb_hist(self, input_tensor, 'res_output_x')
        _tb_hist(self, hidden_states, 'res_output_h')

        hidden_states = hidden_states + input_tensor

        _tb_hist(self, hidden_states, 'res_output_x_h')

        hidden_states = self.res_act_quantizer(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        _tb_advance_global_step(self)
        return hidden_states


def quantize_intermediate(org_module, **quant_params):
    m_dense = org_module.dense
    m_act = org_module.intermediate_act_fn
    if not isinstance(m_act, nn.Module):
        if m_act == F.gelu:
            m_act = nn.GELU()
        else:
            raise NotImplementedError()
    return quantize_model(nn.Sequential(m_dense, m_act), **quant_params)


class QuantizedBertLayer(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = quant_params['quant_dict']

        super().__init__()

        # copy attributes
        self.chunk_size_feed_forward = org_model.chunk_size_feed_forward
        self.seq_len_dim = org_model.seq_len_dim
        self.is_decoder = org_model.is_decoder
        self.add_cross_attention = org_model.add_cross_attention

        # quantized components
        attention_specials = {
            BertSelfAttention: QuantizedBertSelfAttention,
            BertSelfOutput: QuantizedBertSelfOutput,
        }
        self.attention = quantize_model(
            org_model.attention, specials=attention_specials, **quant_params
        )
        if self.add_cross_attention:
            self.crossattention = quantize_model(
                org_model.crossattention, specials=attention_specials, **quant_params
            )
        self.intermediate = quantize_intermediate(org_model.intermediate, **quant_params)
        self.output = QuantizedBertOutput(org_model.output, **quant_params)

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
        attn_args = (hidden_states, attention_mask, head_mask)
        attn_kw = dict(output_attentions=output_attentions)

        self_attention_outputs = self.attention(*attn_args, **attn_kw)

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            raise NotImplementedError('current branch of computation is not yet supported')

            assert hasattr(self, "crossattention"), (
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                f"cross-attention layers by setting `config.add_cross_attention=True`"
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights:
            outputs = outputs + cross_attention_outputs[1:]

        assert self.chunk_size_feed_forward == 0  # below call is a no-op in that case
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class QuantizedBertPooler(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.dense_act = quantize_model(
            nn.Sequential(org_model.dense, org_model.activation), **quant_params
        )

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)

        _tb_advance_global_step(self)
        return pooled_output


class QuantizedBertModel(QuantizedModel, ModuleUtilsMixin):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.config = org_model.config

        self.embeddings = QuantizedBertEmbeddings(org_model.embeddings, **quant_params)
        self.encoder = quantize_model(
            org_model.encoder, specials={BertLayer: QuantizedBertLayer}, **quant_params
        )
        self.pooler = (
            QuantizedBertPooler(org_model.pooler, **quant_params)
            if org_model.pooler is not None
            else None
        )

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Parameters
        ----------
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            raise NotImplementedError('current branch of computation is not yet supported')

            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class QuantizedBertForSequenceClassification(QuantizedModel):
    def __init__(self, org_model, quant_setup=None, **quant_params):
        super().__init__()

        self.num_labels = org_model.num_labels
        self.config = org_model.config

        if hasattr(org_model, 'bert'):
            self.bert = QuantizedBertModel(org_model=org_model.bert, **quant_params)
        if hasattr(org_model, 'dropout'):
            self.dropout = org_model.dropout

        quant_params_ = quant_params.copy()

        if quant_setup == 'MSE_logits':
            quant_params_['act_range_method'] = RangeEstimators.MSE
            quant_params_['act_range_options'] = dict(opt_method=OptMethod.golden_section)
            self.classifier = quantize_model(org_model.classifier, **quant_params_)

        elif quant_setup == 'FP_logits':
            print('Do not quantize output of FC layer')

            self.classifier = quantize_model(org_model.classifier, **quant_params_)
            # no activation quantization of logits:
            self.classifier.activation_quantizer = FP32Acts()

        elif quant_setup == 'all':
            self.classifier = quantize_model(org_model.classifier, **quant_params_)

        else:
            raise ValueError("Quantization setup '{}' not supported.".format(quant_setup))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if isinstance(input_ids, tuple):
            if len(input_ids) == 2:
                input_ids, attention_mask = input_ids
            elif len(input_ids) == 3:
                input_ids, attention_mask, token_type_ids = input_ids
            elif len(input_ids) == 4:
                input_ids, attention_mask, token_type_ids, labels = input_ids
            else:
                raise ValueError('cannot interpret input tuple, use dict instead')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if self.num_labels == 1:
            logits = torch.clamp(logits, 0.0, 5.0)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        _tb_advance_global_step(self)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
