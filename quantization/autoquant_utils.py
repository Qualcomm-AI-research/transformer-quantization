# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import warnings

from torch.nn import functional as F
from torch import nn
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AvgPoolNd

from quantization.base_quantized_classes import FP32Acts, QuantizedActivation, QuantizedModule
from quantization.hijacker import QuantizationHijacker, activations_list
from quantization.quantization_manager import QuantizationManager


class QuantLinear(QuantizationHijacker, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantizedActivationWrapper(QuantizedActivation):
    """
    Wraps over a layer and quantized the activation.
    It also allow for tying the input and output quantizer which is helpful
    for layers such Average Pooling.
    """
    def __init__(self, layer, tie_activation_quantizers=False,
                 input_quantizer: QuantizationManager = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tie_activation_quantizers = tie_activation_quantizers
        if input_quantizer:
            assert isinstance(input_quantizer, QuantizationManager)
            self.activation_quantizer = input_quantizer
        self.layer = layer

    def quantize_activations_no_range_update(self, x):
        if self._quant_a:
            return self.activation_quantizer.quantizer(x)
        else:
            return x

    def forward(self, x):
        x = self.layer(x)
        if self.tie_activation_quantizers:
            # The input activation quantizer is used to quantize the activation
            # but without updating the quantization range
            return self.quantize_activations_no_range_update(x)
        else:
            return self.quantize_activations(x)


class QuantLayerNorm(QuantizationHijacker, nn.LayerNorm):
    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )


class QuantEmbedding(QuantizationHijacker, nn.Embedding):
    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        # NB: Embedding should not quantize activations, as it is simply a lookup table,
        # which is already quantized.
        self.activation_quantizer = FP32Acts()

    def run_forward(self, x, weight, bias, offsets=None):
        return F.embedding(
            input=x.contiguous(),
            weight=weight.contiguous(),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


module_map = {
    nn.Linear: QuantLinear,
    nn.LayerNorm: QuantLayerNorm,
    nn.Embedding: QuantEmbedding
}


non_param_modules = (_AdaptiveAvgPoolNd, _AvgPoolNd)


def get_act(module, i):
    result, act_idx = None, None
    for i in range(i + 1, len(module)):
        if isinstance(module[i], tuple(activations_list)):
            result = module[i]
            act_idx = i
            break
    return result, act_idx


def get_linear_args(module):
    args = dict(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    )
    return args


def get_layernorm_args(module):
    args = dict(normalized_shape=module.normalized_shape, eps=module.eps)
    return args


def get_embedding_args(module):
    args = dict(
        num_embeddings=module.num_embeddings,
        embedding_dim=module.embedding_dim,
        padding_idx=module.padding_idx,
        max_norm=module.max_norm,
        norm_type=module.norm_type,
        scale_grad_by_freq=module.scale_grad_by_freq,
        sparse=module.sparse,
    )
    return args


def get_module_args(mod, act):
    if isinstance(mod, nn.Linear):
        kwargs = get_linear_args(mod)
    elif isinstance(mod, nn.LayerNorm):
        kwargs = get_layernorm_args(mod)
    elif isinstance(mod, nn.Embedding):
        kwargs = get_embedding_args(mod)
    else:
        raise ValueError

    kwargs['activation'] = act
    return kwargs


def quant_module(module, i, **quant_params):
    act, act_idx = get_act(module, i)
    modtype = module_map[type(module[i])]

    kwargs = get_module_args(module[i], act)
    new_module = modtype(**kwargs, **quant_params)
    new_module.weight.data = module[i].weight.data.clone()

    if module[i].bias is not None:
        new_module.bias.data = module[i].bias.data.clone()

    return new_module, i + int(bool(act)) + 1


def quantize_sequence(model, specials=None, tie_activation_quantizers=False, **quant_params):
    specials = specials or dict()

    i = 0
    quant_modules = []
    while i < len(model):
        if isinstance(model[i], QuantizedModule):
            quant_modules.append(model[i])
        elif type(model[i]) in module_map:
            new_module, new_i = quant_module(model, i, **quant_params)
            quant_modules.append(new_module)
            i = new_i
            continue

        elif type(model[i]) in specials:
            quant_modules.append(specials[type(model[i])](model[i], **quant_params))

        elif isinstance(model[i], non_param_modules):
            # check for last quantizer
            input_quantizer = None
            if (
                quant_modules
                and isinstance(quant_modules[-1], QuantizedModule)
                and tie_activation_quantizers
            ):
                input_quantizer = quant_modules[-1].activation_quantizer
                warnings.warn(
                    f'Tying input quantizer {i}^th layer of type '
                    f'{type(quant_modules[-1])} to the quantized {type(model[i])} '
                    f'following it'
                )
            quant_modules.append(
                QuantizedActivationWrapper(
                    model[i],
                    tie_activation_quantizers=tie_activation_quantizers,
                    input_quantizer=input_quantizer,
                    **quant_params,
                )
            )

        else:
            quant_modules.append(quantize_model(model[i], specials=specials, **quant_params))
        i += 1
    return quant_modules


def quantize_sequential(model, specials=None, tie_activation_quantizers=False, **quant_params):
    quant_modules = quantize_sequence(model, specials, tie_activation_quantizers, **quant_params)
    return nn.Sequential(*quant_modules)


def quantize_module_list(model, specials=None, tie_activation_quantizers=False, **quant_params):
    quant_modules = quantize_sequence(model, specials, tie_activation_quantizers, **quant_params)
    return nn.ModuleList(quant_modules)


def quantize_model(model, specials=None, tie_activation_quantizers=False, **quant_params):
    specials = specials or dict()

    if isinstance(model, nn.Sequential):
        quant_model = quantize_sequential(
            model, specials, tie_activation_quantizers, **quant_params
        )

    elif type(model) in specials:
        quant_model = specials[type(model)](model, **quant_params)

    elif isinstance(model, non_param_modules):
        quant_model = QuantizedActivationWrapper(model, **quant_params)

    elif type(model) in module_map:
        # if we do isinstance() then we might run into issues with modules that inherit from
        # one of these classes, for whatever reason
        modtype = module_map[type(model)]
        kwargs = get_module_args(model, None)
        quant_model = modtype(**kwargs, **quant_params)

        quant_model.weight.data = model.weight.data
        if getattr(model, 'bias', None) is not None:
            quant_model.bias.data = model.bias.data

    else:
        # unknown type, try to quantize all child modules
        quant_model = copy.deepcopy(model)
        for name, module in quant_model._modules.items():
            new_model = quantize_model(module, specials=specials, **quant_params)
            if new_model is not None:
                setattr(quant_model, name, new_model)

    return quant_model
