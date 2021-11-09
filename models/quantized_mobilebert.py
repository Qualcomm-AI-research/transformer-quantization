# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.models.mobilebert.modeling_mobilebert import (
    BaseModelOutputWithPooling,
    BottleneckLayer,
    FFNLayer,
    MobileBertLayer,
    MobileBertSelfAttention,
    MobileBertSelfOutput,
    NoNorm,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import ModuleUtilsMixin

from quantization.autoquant_utils import quantize_model, quantize_module_list
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel
from quantization.hijacker import QuantizationHijacker
from quantization.range_estimators import RangeEstimators, OptMethod
from utils import DotDict, _tb_advance_global_step, _tb_advance_token_counters, _tb_hist


DEFAULT_QUANT_DICT = {
    # Embeddings
    'sum_input_pos_embd': True,
    'sum_token_type_embd': True,

    # Attention
    'attn_scores': True,
    'attn_probs': True,
    'attn_probs_n_bits_act': None,
    'attn_probs_act_range_method': None,
    'attn_probs_act_range_options': None,
    'attn_output': True,

    # Residual connections
    'res_self_output': True,
    'res_output': True,
    'res_output_bottleneck': True,
    'res_ffn_output': True,
}


def _make_quant_dict(partial_dict):
    quant_dict = DEFAULT_QUANT_DICT.copy()
    quant_dict.update(partial_dict)
    return DotDict(quant_dict)


class QuantNoNorm(QuantizationHijacker):
    def __init__(self, org_model, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        self.weight = org_model.weight
        self.bias = org_model.bias

    def forward(self, x, offsets=None):
        weight, bias = self.weight, self.bias
        if self._quant_w:
            weight = self.weight_quantizer(weight)
            bias = self.weight_quantizer(bias)

        res = x * weight + bias
        res = self.quantize_activations(res)
        return res


class QuantizedMobileBertEmbeddings(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        # copy attributes
        self.trigram_input = org_model.trigram_input
        self.embedding_size = org_model.embedding_size
        self.hidden_size = org_model.hidden_size

        # quantized modules
        self.word_embeddings = quantize_model(org_model.word_embeddings, **quant_params)
        self.position_embeddings = quantize_model(org_model.position_embeddings, **quant_params)
        self.token_type_embeddings = quantize_model(org_model.token_type_embeddings, **quant_params)

        self.embedding_transformation = quantize_model(
            org_model.embedding_transformation, **quant_params
        )

        assert isinstance(org_model.LayerNorm, NoNorm)
        self.LayerNorm = QuantNoNorm(org_model.LayerNorm, **quant_params)

        self.dropout = org_model.dropout

        position_ids = org_model.position_ids
        if position_ids is not None:
            self.register_buffer('position_ids', position_ids)
        else:
            self.position_ids = position_ids

        # activation quantizers
        self.quant_dict = _make_quant_dict(quant_params['quant_dict'])
        self.sum_input_pos_embd_act_quantizer = (
            QuantizedActivation(**quant_params)
            if self.quant_dict.sum_input_pos_embd
            else FP32Acts()
        )
        self.sum_token_type_embd_act_quantizer = (
            QuantizedActivation(**quant_params)
            if self.quant_dict.sum_token_type_embd
            else FP32Acts()
        )

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
            inputs_embeds = self.word_embeddings(input_ids)  # (B, T, 128)

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = torch.cat(
                [
                    F.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0),
                    inputs_embeds,
                    F.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0),
                ],
                dim=2,
            )  # (B, T, 384)

        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)  # (B, T, 512)

        # Add positional embeddings and token type embeddings, then layer  # normalize and
        # perform dropout.
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = self.sum_input_pos_embd_act_quantizer(inputs_embeds + position_embeddings)
        embeddings = self.sum_token_type_embd_act_quantizer(embeddings + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QuantizedMobileBertSelfAttention(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        # copy attributes
        self.num_attention_heads = org_model.num_attention_heads
        self.attention_head_size = org_model.attention_head_size
        self.all_head_size = org_model.all_head_size

        # quantized modules
        self.query = quantize_model(org_model.query, **quant_params)
        self.key = quantize_model(org_model.key, **quant_params)
        self.value = quantize_model(org_model.value, **quant_params)
        self.dropout = org_model.dropout

        # activation quantizers
        self.quant_dict = _make_quant_dict(quant_params['quant_dict'])
        self.attn_scores_act_quantizer = (
            QuantizedActivation(**quant_params) if self.quant_dict.attn_scores else FP32Acts()
        )

        quant_params_ = quant_params.copy()
        if self.quant_dict.attn_probs_n_bits_act is not None:
            quant_params_['n_bits_act'] = self.quant_dict.attn_probs_n_bits_act
        if self.quant_dict.attn_probs_act_range_method is not None:
            quant_params_['act_range_method'] = RangeEstimators[
                self.quant_dict.attn_probs_act_range_method
            ]
        if self.quant_dict.attn_probs_act_range_options is not None:
            act_range_options = self.quant_dict.attn_probs_act_range_options
            if 'opt_method' in act_range_options and not isinstance(act_range_options['opt_method'],
                                                                    OptMethod):
                act_range_options['opt_method'] = OptMethod[act_range_options['opt_method']]
            quant_params_['act_range_options'] = act_range_options
        self.attn_probs_act_quantizer = (
            QuantizedActivation(**quant_params_) if self.quant_dict.attn_probs else FP32Acts()
        )

        self.attn_output_act_quantizer = (
            QuantizedActivation(**quant_params) if self.quant_dict.attn_output else FP32Acts()
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.attn_scores_act_quantizer(attention_scores)

        # NOTE: factor 1/d^0.5 can be absorbed into the previous act. quant. delta
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

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
        context_layer = self.attn_output_act_quantizer(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class QuantizedMobileBertSelfOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        # copy attributes
        self.use_bottleneck = org_model.use_bottleneck

        # quantized modules
        self.dense = quantize_model(org_model.dense, **quant_params)

        assert isinstance(org_model.LayerNorm, NoNorm)
        self.LayerNorm = QuantNoNorm(org_model.LayerNorm, **quant_params)

        if not self.use_bottleneck:
            self.dropout = org_model.dropout

        # activation quantizers
        self.quant_dict = _make_quant_dict(quant_params['quant_dict'])
        self.res_act_quantizer = (
            QuantizedActivation(**quant_params) if self.quant_dict.res_self_output else FP32Acts()
        )

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)

        _tb_advance_token_counters(self, layer_outputs)
        _tb_hist(self, layer_outputs, 'res_self_output_h')
        _tb_hist(self, residual_tensor, 'res_self_output_x')

        layer_outputs = layer_outputs + residual_tensor

        _tb_hist(self, residual_tensor, 'res_self_output_x_h')

        layer_outputs = self.res_act_quantizer(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs)

        _tb_advance_global_step(self)
        return layer_outputs


def quantize_intermediate(org_module, **quant_params):
    m_dense = org_module.dense
    m_act = org_module.intermediate_act_fn
    if not isinstance(m_act, nn.Module):
        if m_act == F.gelu:
            m_act = nn.GELU()
        elif m_act == F.relu:
            m_act = nn.ReLU()
        else:
            raise NotImplementedError()
    return quantize_model(nn.Sequential(m_dense, m_act), **quant_params)


class QuantizedOutputBottleneck(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        assert isinstance(org_model.LayerNorm, NoNorm)
        self.LayerNorm = QuantNoNorm(org_model.LayerNorm, **quant_params)
        self.dropout = org_model.dropout

        # activation quantizers
        self.quant_dict = _make_quant_dict(quant_params['quant_dict'])
        self.res_act_quantizer = (
            QuantizedActivation(**quant_params)
            if self.quant_dict.res_output_bottleneck
            else FP32Acts()
        )

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs)

        _tb_advance_token_counters(self, layer_outputs)
        _tb_hist(self, layer_outputs, 'res_layer_h')
        _tb_hist(self, residual_tensor, 'res_layer_x')

        layer_outputs = layer_outputs + residual_tensor

        _tb_hist(self, layer_outputs, 'res_layer_x_h')

        layer_outputs = self.res_act_quantizer(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs)

        _tb_advance_global_step(self)
        return layer_outputs


class QuantizedMobileBertOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        # copy attributes
        self.use_bottleneck = org_model.use_bottleneck

        # quantized modules
        self.dense = quantize_model(org_model.dense, **quant_params)
        assert isinstance(org_model.LayerNorm, NoNorm)
        self.LayerNorm = QuantNoNorm(org_model.LayerNorm, **quant_params)

        if not self.use_bottleneck:
            self.dropout = org_model.dropout
        else:
            self.bottleneck = QuantizedOutputBottleneck(
                org_model=org_model.bottleneck, **quant_params
            )

        # activation quantizers
        self.quant_dict = _make_quant_dict(quant_params['quant_dict'])
        self.res_act_quantizer = (
            QuantizedActivation(**quant_params) if self.quant_dict.res_output else FP32Acts()
        )

    def forward(self, intermediate_states, residual_tensor_1, residual_tensor_2):
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = layer_output + residual_tensor_1
            layer_output = self.res_act_quantizer(layer_output)
            layer_output = self.LayerNorm(layer_output)
        else:
            _tb_advance_token_counters(self, layer_output)
            _tb_hist(self, layer_output, 'res_interm_h')
            _tb_hist(self, residual_tensor_1, 'res_interm_x')

            layer_output = layer_output + residual_tensor_1

            _tb_hist(self, layer_output, 'res_interm_x_h')

            layer_output = self.res_act_quantizer(layer_output)
            layer_output = self.LayerNorm(layer_output)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)

        _tb_advance_global_step(self)
        return layer_output


class QuantizedBottleneckLayer(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        assert isinstance(org_model.LayerNorm, NoNorm)
        self.LayerNorm = QuantNoNorm(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states):
        layer_input = self.dense(hidden_states)
        layer_input = self.LayerNorm(layer_input)
        return layer_input


class QuantizedFFNOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        assert isinstance(org_model.LayerNorm, NoNorm)
        self.LayerNorm = QuantNoNorm(org_model.LayerNorm, **quant_params)

        # activation quantizers
        self.quant_dict = _make_quant_dict(quant_params['quant_dict'])
        self.res_act_quantizer = (
            QuantizedActivation(**quant_params) if self.quant_dict.res_ffn_output else FP32Acts()
        )

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)

        _tb_advance_token_counters(self, layer_outputs)
        num_ffn = self.ffn_idx + 1
        _tb_hist(self, layer_outputs, f'res_ffn{num_ffn}_h')
        _tb_hist(self, residual_tensor, f'res_ffn{num_ffn}_x')

        layer_outputs = layer_outputs + residual_tensor

        _tb_hist(self, layer_outputs, f'res_ffn{num_ffn}_x_h')

        layer_outputs = self.res_act_quantizer(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs)

        _tb_advance_global_step(self)
        return layer_outputs


class QuantizedFFNLayer(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.intermediate = quantize_intermediate(org_model.intermediate, **quant_params)
        self.output = QuantizedFFNOutput(org_model.output, **quant_params)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


class QuantizedMobileBertLayer(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        # copy
        self.use_bottleneck = org_model.use_bottleneck
        self.num_feedforward_networks = org_model.num_feedforward_networks

        # quantized modules
        attention_specials = {
            MobileBertSelfAttention: QuantizedMobileBertSelfAttention,
            MobileBertSelfOutput: QuantizedMobileBertSelfOutput,
        }
        self.attention = quantize_model(
            org_model.attention, specials=attention_specials, **quant_params
        )
        self.intermediate = quantize_intermediate(org_model.intermediate, **quant_params)
        self.output = QuantizedMobileBertOutput(org_model.output, **quant_params)

        if self.use_bottleneck:
            self.bottleneck = quantize_model(
                org_model.bottleneck,
                specials={BottleneckLayer: QuantizedBottleneckLayer},
                **quant_params,
            )
        if getattr(org_model, 'ffn', None) is not None:
            self.ffn = quantize_module_list(
                org_model.ffn, specials={FFNLayer: QuantizedFFNLayer}, **quant_params
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                # attach index for TB vis
                for m in ffn_module.modules():
                    m.ffn_idx = i

                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        outputs = (
            (layer_output,)
            + outputs
            + (
                torch.tensor(1000),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        return outputs


class QuantizedMobileBertPooler(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.do_activate = org_model.do_activate
        if self.do_activate:
            self.dense_act = quantize_model(
                nn.Sequential(org_model.dense, nn.Tanh()), **quant_params
            )

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            return first_token_tensor
        else:
            pooled_output = self.dense_act(first_token_tensor)
            return pooled_output


class QuantizedMobileBertModel(QuantizedModel, ModuleUtilsMixin):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.config = org_model.config

        self.embeddings = QuantizedMobileBertEmbeddings(org_model.embeddings, **quant_params)
        self.encoder = quantize_model(
            org_model.encoder, specials={MobileBertLayer: QuantizedMobileBertLayer}, **quant_params
        )
        self.pooler = (
            QuantizedMobileBertPooler(org_model.pooler, **quant_params)
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
        output_hidden_states=None,
        output_attentions=None,
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
            attention_mask, input_shape, self.device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class QuantizedMobileBertForSequenceClassification(QuantizedModel):
    def __init__(self, org_model, quant_setup=None, **quant_params):
        super().__init__()

        self.num_labels = org_model.num_labels
        self.config = org_model.config

        self.mobilebert = QuantizedMobileBertModel(org_model=org_model.mobilebert, **quant_params)
        self.dropout = org_model.dropout
        self.classifier = quantize_model(org_model.classifier, **quant_params)

        if quant_setup == 'FP_logits':
            print('Do not quantize output of FC layer')
            # no activation quantization of logits:
            self.classifier.activation_quantizer = FP32Acts()
        elif quant_setup is not None and quant_setup != 'all':
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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

        # NB: optionally can keep final logits un-quantized, if only used for prediction
        # (can be enabled via --quant-setup FP_logits)
        logits = self.classifier(pooled_output)

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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
