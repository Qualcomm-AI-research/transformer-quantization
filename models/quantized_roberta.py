# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaSelfAttention,
    RobertaSelfOutput,
    RobertaLayer,
)

from models.quantized_bert import (
    QuantizedBertEmbeddings,
    QuantizedBertSelfAttention,
    QuantizedBertSelfOutput,
    QuantizedBertOutput,
    QuantizedBertLayer,
    QuantizedBertPooler,
    QuantizedBertModel,
    QuantizedBertForSequenceClassification,
)
from quantization.autoquant_utils import quantize_model


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX
    # export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class QuantizedRobertaEmbeddings(QuantizedBertEmbeddings):
    def __init__(self, org_model, **quant_params):
        super().__init__(org_model, **quant_params)

        self.padding_idx = org_model.padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.sum_input_token_type_embd_act_quantizer(embeddings)

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.sum_pos_embd_act_quantizer(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QuantizedRobertaSelfAttention(QuantizedBertSelfAttention):
    pass


class QuantizedRobertaSelfOutput(QuantizedBertSelfOutput):
    pass


class QuantizedRobertaOutput(QuantizedBertOutput):
    pass


class QuantizedRobertaLayer(QuantizedBertLayer):
    def __init__(self, org_model, **quant_params):
        super().__init__(org_model, **quant_params)

        # update quantized components
        attention_specials = {
            RobertaSelfAttention: QuantizedRobertaSelfAttention,
            RobertaSelfOutput: QuantizedRobertaSelfOutput,
        }
        self.attention = quantize_model(
            org_model.attention, specials=attention_specials, **quant_params
        )
        if self.add_cross_attention:
            self.crossattention = quantize_model(
                org_model.crossattention, specials=attention_specials, **quant_params
            )
        self.output = QuantizedRobertaOutput(org_model.output, **quant_params)


class QuantizedRobertaPooler(QuantizedBertPooler):
    pass


class QuantizedRobertaModel(QuantizedBertModel):
    def __init__(self, org_model, **quant_params):
        super().__init__(org_model, **quant_params)

        # update quantized components
        self.embeddings = QuantizedRobertaEmbeddings(org_model.embeddings, **quant_params)
        self.encoder = quantize_model(
            org_model.encoder, specials={RobertaLayer: QuantizedRobertaLayer}, **quant_params
        )
        self.pooler = (
            QuantizedRobertaPooler(org_model.pooler, **quant_params)
            if org_model.pooler is not None
            else None
        )


class QuantizedRobertaForSequenceClassification(QuantizedBertForSequenceClassification):
    def __init__(self, org_model, quant_setup=None, **quant_params):
        super().__init__(org_model, quant_setup=quant_setup, **quant_params)

        # update quantization components
        self.roberta = QuantizedRobertaModel(org_model=org_model.roberta, **quant_params)
        self.classifier = quantize_model(org_model.classifier, **quant_params)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        sequence_output = outputs[0]

        # NOTE: optionally can keep final logits un-quantized, if only used for prediction
        # (can be enabled via --quant-setup FP_logits)
        logits = self.classifier(sequence_output)

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
