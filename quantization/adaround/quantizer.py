# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

import torch
import torch.nn as nn

from quantization.quantizers import (
    QuantizerBase,
    AsymmetricUniformQuantizer,
    SymmetricUniformQuantizer,
)
from quantization.adaround.utils import AdaRoundMode


# setup logger
logger = logging.getLogger('AdaRound')
logger.setLevel(logging.INFO)


def logit(p, eps=1e-16):
    p = torch.clamp(p, eps, 1 - eps)
    return -torch.log(1 / p - 1)


def hard_sigmoid(x, zeta=1.1, gamma=-0.1):
    p = torch.sigmoid(x)
    return torch.clamp(p * (zeta - gamma) + gamma, 0.0, 1.0)


def hard_logit(p, zeta=1.1, gamma=-0.1):
    # NOTE: argument of log is between 1/11 and 11 (for default values of zeta and gamma)
    return -torch.log((zeta - p) / (p - gamma))


class AdaRoundQuantizer(QuantizerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = None
        self.round_mode = AdaRoundMode.nearest
        self.soft_targets = False
        self.temperature = None  # for sigmoid temperature annealing

    def to_integer_forward(self, x_float):
        if self.round_mode == AdaRoundMode.nearest:
            return super().to_integer_forward(x_float)

        if self.round_mode not in AdaRoundMode.RELAXATION:
            raise ValueError(f'Unknown rounding mode: {self.round_mode}')

        # cont. relaxation
        x = x_float / self.scale
        x_floor = torch.floor(x)

        # initialize alpha, if needed
        if self.alpha is None:
            logger.info('Init alpha to be FP32')

            rest = x - x_floor  # rest of rounding [0, 1)
            if self.round_mode == AdaRoundMode.learned_sigmoid:
                alpha = logit(rest)  # => sigmoid(alpha) = rest
            elif self.round_mode == AdaRoundMode.learned_hard_sigmoid:
                alpha = hard_logit(rest)  # => hard_sigmoid(alpha) = rest
            elif self.round_mode == AdaRoundMode.sigmoid_temp_decay:
                alpha = self.temperature * logit(rest)  # => sigmoid(alpha/temperature) = rest
            else:
                raise ValueError(f'Unknown rounding mode: {self.round_mode}')

            self.alpha = nn.Parameter(alpha, requires_grad=True)

        # compute final x_int
        x_int = x_floor + (self.get_rest() if self.soft_targets else (self.alpha >= 0).float())

        if not self.symmetric:
            x_int += self.zero_point

        x_int = torch.clamp(x_int, self.int_min, self.int_max)
        return x_int

    def get_rest(self):
        if self.round_mode == AdaRoundMode.learned_sigmoid:
            return torch.sigmoid(self.alpha)
        elif self.round_mode == AdaRoundMode.learned_hard_sigmoid:
            return hard_sigmoid(self.alpha)
        elif self.round_mode == AdaRoundMode.sigmoid_temp_decay:
            return torch.sigmoid(self.alpha / self.temperature)
        else:
            raise ValueError(f'Unknown rounding mode: {self.round_mode}')

    def extra_repr(self):
        return ', '.join([
            f'n_bits={self.n_bits}',
            f'per_channel={self.per_channel}',
            f'is_initialized={self.is_initialized}',
            f'round_mode={self.round_mode}',
            f'soft_targets={self.soft_targets}',
            f'temperature={self.temperature}',
        ])


class AdaRoundSymmetricUniformQuantizer(AdaRoundQuantizer, SymmetricUniformQuantizer):
    pass


class AdaRoundAsymmetricUniformQuantizer(AdaRoundQuantizer, AsymmetricUniformQuantizer):
    pass


ADAROUND_QUANTIZER_MAP = {
    SymmetricUniformQuantizer: AdaRoundSymmetricUniformQuantizer,
    AsymmetricUniformQuantizer: AdaRoundAsymmetricUniformQuantizer,
}
