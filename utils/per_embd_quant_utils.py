# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from quantization.base_quantized_classes import FP32Acts


def _hijack_act_quant(module, value):
    if value is None:
        return

    if isinstance(value, int):
        module.activation_quantizer.quantizer.n_bits = value
    elif value == 'fp32':
        module.activation_quantizer = FP32Acts()
    elif value == 'per_embd':
        set_act_quant_axis_and_groups(module, axis=2, n_groups=None)
    elif value.startswith('ngp'):
        set_act_quant_axis_and_groups(module, axis=2, n_groups=int(value[3:]), permute=True)
    elif value.startswith('ng'):
        set_act_quant_axis_and_groups(module, axis=2, n_groups=int(value[2:]), permute=False)
    else:
        raise NotImplementedError(f'Unknown value "{value}" in quant_dict')


def _hijack_weight_quant(module, value):
    if value is None:
        return

    if isinstance(value, int):
        module.weight_quantizer.quantizer.n_bits = value
    elif value == 'fp32':
        module.weight_quantizer = FP32Acts()
    else:
        raise NotImplementedError(f'Unknown value "{value}" in quant_dict')


def hijack_act_quant(quant_dict, name, m):
    value = quant_dict.get(name, None)
    _hijack_act_quant(m, value)


def hijack_weight_quant(quant_dict, name, m):
    value = quant_dict.get(name, None)
    _hijack_weight_quant(m, value)


def hijack_act_quant_modules(quant_dict, name, m):
    value = quant_dict.get(name, None)
    for m_ in m.modules():
        if hasattr(m_, 'activation_quantizer'):
            _hijack_act_quant(m_, value)


def set_act_quant_axis_and_groups(module, axis, n_groups, permute=False):
    if hasattr(module, 'activation_quantizer'):
        module = module.activation_quantizer

    module.axis = axis
    module.quantizer.axis = axis
    module.range_estimator.axis = axis

    module.n_groups = n_groups
    module.range_estimator.n_groups = n_groups

    if permute:
        module.range_estimator.per_group_range_estimation = True

    return module
