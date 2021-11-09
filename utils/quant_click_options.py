# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from functools import wraps

import click

from quantization.adaround.config import AdaRoundConfig, DEFAULT_ADAROUND_CONFIG as C
from quantization.adaround.utils import (
    AdaRoundActQuantMode,
    AdaRoundInitMode,
    AdaRoundMode,
    AdaRoundTempDecayType,
)
from quantization.quantizers import QMethods
from quantization.range_estimators import RangeEstimators, OptMethod
from utils.utils import DotDict


class StrTuple(click.ParamType):
    name = 'str_sequence'

    def convert(self, value, param, ctx):
        values = value.split(',')
        values = tuple(map(lambda s: s.strip(), values))
        return values


def split_dict(src: dict, include=()):
    """
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.

    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    """
    result = DotDict()

    for arg in include:
        result[arg] = src[arg]
    remainder = {key: val for key, val in src.items() if key not in include}
    return result, remainder


def quantization_options(func):
    @click.option(
        '--qmethod',
        type=click.Choice(QMethods.list()),
        required=True,
        help='Quantization scheme to use.',
    )
    @click.option(
        '--qmethod-act',
        type=click.Choice(QMethods.list()),
        default=None,
        help='Quantization scheme for activation to use. If not specified `--qmethod` is used.',
    )
    @click.option(
        '--weight-quant-method',
        default=RangeEstimators.current_minmax.name,
        type=click.Choice(RangeEstimators.list()),
        help='Method to determine weight quantization clipping thresholds.',
    )
    @click.option(
        '--weight-opt-method',
        default=OptMethod.grid.name,
        type=click.Choice(OptMethod.list()),
        help='Optimization procedure for activation quantization clipping thresholds',
    )
    @click.option(
        '--num-candidates',
        type=int,
        default=None,
        help='Number of grid points for grid search in MSE range method.',
    )
    @click.option('--n-bits', default=8, type=int, help='Default number of quantization bits.')
    @click.option(
        '--n-bits-act', default=None, type=int, help='Number of quantization bits for activations.'
    )
    @click.option('--per-channel', is_flag=True, help='If given, quantize each channel separately.')
    @click.option(
        '--percentile',
        type=float,
        default=None,
        help='Percentile clipping parameter (weights and activations)',
    )
    @click.option(
        '--act-quant/--no-act-quant',
        is_flag=True,
        default=True,
        help='Run evaluation with activation quantization or use FP32 activations',
    )
    @click.option(
        '--weight-quant/--no-weight-quant',
        is_flag=True,
        default=True,
        help='Run evaluation weight quantization or use FP32 weights',
    )
    @click.option(
        '--quant-setup',
        default='all',
        type=click.Choice(['all', 'FP_logits', 'MSE_logits']),
        help='Method to quantize the network.',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        config.quant, remainder_kwargs = split_dict(kwargs, [
            'qmethod',
            'qmethod_act',
            'weight_quant_method',
            'weight_opt_method',
            'num_candidates',
            'n_bits',
            'n_bits_act',
            'per_channel',
            'percentile',
            'act_quant',
            'weight_quant',
            'quant_setup',
        ])

        config.quant.qmethod_act = config.quant.qmethod_act or config.quant.qmethod

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def activation_quantization_options(func):
    @click.option(
        '--act-quant-method',
        default=RangeEstimators.running_minmax.name,
        type=click.Choice(RangeEstimators.list()),
        help='Method to determine activation quantization clipping thresholds',
    )
    @click.option(
        '--act-opt-method',
        default=OptMethod.grid.name,
        type=click.Choice(OptMethod.list()),
        help='Optimization procedure for activation quantization clipping thresholds',
    )
    @click.option(
        '--act-num-candidates',
        type=int,
        default=None,
        help='Number of grid points for grid search in MSE/Cross-entropy',
    )
    @click.option(
        '--act-momentum',
        type=float,
        default=None,
        help='Exponential averaging factor for running_minmax',
    )
    @click.option(
        '--cross-entropy-layer',
        default=None,
        type=str,
        help='Cross-entropy for activation range setting (often valuable for last layer)',
    )
    @click.option(
        '--num-est-batches',
        type=int,
        default=1,
        help='Number of training batches to be used for activation range estimation',
    )
    @wraps(func)
    def func_wrapper(config, act_quant_method, act_opt_method, act_num_candidates, act_momentum,
                     cross_entropy_layer, num_est_batches, *args, **kwargs):
        config.act_quant = DotDict()
        config.act_quant.quant_method = act_quant_method
        config.act_quant.cross_entropy_layer = cross_entropy_layer
        config.act_quant.num_batches = num_est_batches

        config.act_quant.options = {}

        if act_num_candidates is not None:
            if act_quant_method != 'MSE':
                raise ValueError('Wrong option num_candidates passed')
            else:
                config.act_quant.options['num_candidates'] = act_num_candidates

        if act_momentum is not None:
            if act_quant_method != 'running_minmax':
                raise ValueError('Wrong option momentum passed')
            else:
                config.act_quant.options['momentum'] = act_momentum

        if act_opt_method != 'grid':
            config.act_quant.options['opt_method'] = OptMethod[act_opt_method]
        return func(config, *args, **kwargs)

    return func_wrapper


def qat_options(func):
    @click.option(
        '--learn-ranges',
        is_flag=True,
        default=False,
        help='Learn quantization ranges, in that case fix ranges will be ignored.',
    )
    @click.option(
        '--fix-act-ranges/--no-fix-act-ranges',
        is_flag=True,
        default=False,
        help='Fix all activation quantization ranges for stable training',
    )
    @click.option(
        '--fix-weight-ranges/--no-fix-weight-ranges',
        is_flag=True,
        default=False,
        help='Fix all weight quantization ranges for stable training',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        config.qat, remainder_kwargs = split_dict(
            kwargs, ['learn_ranges', 'fix_act_ranges', 'fix_weight_ranges']
        )

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def adaround_options(func):
    # Base options
    @click.option(
        '--adaround',
        default=None,
        type=StrTuple(),
        help="Apply AdaRound: for full model ('all'), or any number of layers, "
        "specified by comma-separated names.",
    )
    @click.option(
        '--adaround-num-samples',
        default=C.num_samples,
        type=int,
        help='Number of samples to use for learning the rounding.',
    )
    @click.option(
        '--adaround-init',
        default=C.init.name,
        type=click.Choice(AdaRoundInitMode.list_names(), case_sensitive=False),
        help='Method to initialize the quantization grid for weights.',
    )

    # Method and continuous relaxation options
    @click.option(
        '--adaround-mode',
        default=C.round_mode.name,
        type=click.Choice(AdaRoundMode.list_names(), case_sensitive=False),
        help='Method to learn the rounding.',
    )
    @click.option(
        '--adaround-asym/--no-adaround-asym',
        is_flag=True,
        default=C.asym,
        help='Whether to use asymmetric reconstruction for AdaRound.',
    )
    @click.option(
        '--adaround-include-act-func/--adaround-no-act-func',
        is_flag=True,
        default=C.include_act_func,
        help='Include activation function into AdaRound.',
    )
    @click.option(
        '--adaround-lr',
        default=C.lr,
        type=float,
        help='Learning rate for continuous relaxation in AdaRound.',
    )
    @click.option(
        '--adaround-iters',
        default=C.iters,
        type=int,
        help='Number of iterations to train each layer.',
    )
    @click.option(
        '--adaround-weight',
        default=C.weight,
        type=float,
        help='Weight of rounding cost vs the reconstruction loss.',
    )
    @click.option(
        '--adaround-annealing',
        default=C.annealing,
        nargs=2,
        type=float,
        help='Annealing of regularization function temperature (tuple: start, end).',
    )
    @click.option(
        '--adaround-decay-type',
        default=C.decay_type.name,
        type=click.Choice(AdaRoundTempDecayType.list_names(), case_sensitive=False),
        help='Type of temperature annealing schedule.',
    )
    @click.option(
        '--adaround-decay-shape',
        default=C.decay_shape,
        type=float,
        help="Positive "
        "scalar value that controls the shape of decay schedules 'sigmoid', 'power', "
        "'exp', 'log'. Sensible values to try: sigmoid{10}, power{4,6,8}, exp{4,6,8}, "
        "log{1,2,3}.",
    )
    @click.option(
        '--adaround-decay-start',
        default=C.decay_start,
        type=float,
        help='Start of annealing (relative to --ltr-iters).',
    )
    @click.option(
        '--adaround-warmup',
        default=C.warmup,
        type=float,
        help='In the warmup period no regularization is applied (relative to --ltr-iters).',
    )

    # Activation quantization
    @click.option(
        '--adaround-act-quant',
        default=C.act_quant_mode.name,
        type=click.Choice(AdaRoundActQuantMode.list_names(), case_sensitive=False),
        help='Method to deal with activation quantization during AdaRound.',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        config.adaround = AdaRoundConfig(**C)

        config.adaround.layers = kwargs.pop('adaround')
        config.adaround.num_samples = kwargs.pop('adaround_num_samples')
        config.adaround.init = AdaRoundInitMode[kwargs.pop('adaround_init')]

        config.adaround.round_mode = AdaRoundMode[kwargs.pop('adaround_mode')]
        config.adaround.asym = kwargs.pop('adaround_asym')
        config.adaround.include_act_func = kwargs.pop('adaround_include_act_func')
        config.adaround.lr = kwargs.pop('adaround_lr')
        config.adaround.iters = kwargs.pop('adaround_iters')
        config.adaround.weight = kwargs.pop('adaround_weight')
        config.adaround.annealing = kwargs.pop('adaround_annealing')
        config.adaround.decay_type = AdaRoundTempDecayType[kwargs.pop('adaround_decay_type')]
        config.adaround.decay_shape = kwargs.pop('adaround_decay_shape')
        config.adaround.decay_start = kwargs.pop('adaround_decay_start')
        config.adaround.warmup = kwargs.pop('adaround_warmup')

        config.adaround.act_quant_mode = AdaRoundActQuantMode[kwargs.pop('adaround_act_quant')]
        return func(config, *args, **kwargs)

    return func_wrapper


def make_qparams(config):
    weight_range_options = {}
    if config.quant.weight_quant_method in ['MSE', 'cross_entropy']:
        weight_range_options = dict(opt_method=OptMethod[config.quant.weight_opt_method])
    if config.quant.num_candidates is not None:
        weight_range_options['num_candidates'] = config.quant.num_candidates

    act_range_options = config.act_quant.options
    if config.quant.percentile is not None:
        act_range_options['percentile'] = config.quant.percentile

    params = {
        'method': QMethods[config.quant.qmethod],
        'act_method': QMethods[config.quant.qmethod_act],
        'n_bits': config.quant.n_bits,
        'n_bits_act': config.quant.n_bits_act,
        'per_channel_weights': config.quant.per_channel,
        'percentile': config.quant.percentile,
        'quant_setup': config.quant.quant_setup,
        'weight_range_method': RangeEstimators[config.quant.weight_quant_method],
        'weight_range_options': weight_range_options,
        'act_range_method': RangeEstimators[config.act_quant.quant_method],
        'act_range_options': config.act_quant.options,
    }
    return params
