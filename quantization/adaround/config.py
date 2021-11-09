# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from quantization.adaround.utils import (
    AdaRoundActQuantMode,
    AdaRoundInitMode,
    AdaRoundMode,
    AdaRoundTempDecayType,
)
from utils.utils import DotDict


class AdaRoundConfig(DotDict):
    pass


DEFAULT_ADAROUND_CONFIG = AdaRoundConfig(
    # Base options
    layers=('all',),
    num_samples=1024,
    init=AdaRoundInitMode.range_estimator,

    # Method and continuous relaxation options
    round_mode=AdaRoundMode.learned_hard_sigmoid,
    asym=True,
    include_act_func=True,
    lr=1e-3,
    iters=1000,
    weight=0.01,
    annealing=(20, 2),
    decay_type=AdaRoundTempDecayType.cosine,
    decay_shape=1.0,
    decay_start=0.0,
    warmup=0.2,

    # Activation quantization
    act_quant_mode=AdaRoundActQuantMode.post_adaround,
)
