# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from enum import Flag, auto
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import StopForwardException


# setup logger
logger = logging.getLogger('AdaRound')
logger.setLevel(logging.INFO)


def sigmoid(x):
    return (1.0 + np.exp(-x)) ** -1.0


class BaseOption(Flag):
    def __str__(self):
        return self.name

    @property
    def cls(self):
        return self.value.cls

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class AdaRoundActQuantMode(BaseOption):
    # Activation quantization is disabled
    no_act_quant = auto()

    # AdaRound with FP32 acts, activations are quantized afterwards if applicable (default):
    post_adaround = auto()


class AdaRoundInitMode(BaseOption):
    """Weight quantization grid initialization."""

    range_estimator = auto()
    mse = auto()  # old implementation
    mse_out = auto()
    mse_out_asym = auto()


class AdaRoundLossType(BaseOption):
    """Regularization terms."""

    relaxation = auto()
    temp_decay = auto()


class AdaRoundMode(BaseOption):
    nearest = auto()  # (default)

    # original AdaRound relaxation methods
    learned_sigmoid = auto()
    learned_hard_sigmoid = auto()
    sigmoid_temp_decay = auto()

    RELAXATION = learned_sigmoid | learned_hard_sigmoid | sigmoid_temp_decay

    @classmethod
    def list_names(cls):
        exclude = (AdaRoundMode.nearest, AdaRoundMode.RELAXATION)
        return [m.name for m in cls if not m in exclude]


MODE_TO_LOSS_TYPE = {
    AdaRoundMode.learned_hard_sigmoid: AdaRoundLossType.relaxation,
    AdaRoundMode.learned_sigmoid: AdaRoundLossType.relaxation,
    AdaRoundMode.sigmoid_temp_decay: AdaRoundLossType.temp_decay,
}


class AdaRoundTempDecayType(BaseOption):
    linear = auto()
    cosine = auto()
    sigmoid = auto()  # https://arxiv.org/abs/1811.09332
    power = auto()
    exp = auto()
    log = auto()


class TempDecay:
    def __init__(self, t_max, b_range=(20.0, 2.0), rel_decay_start=0.0,
                 decay_type=AdaRoundTempDecayType.linear, decay_shape=1.0):
        self.t_max = t_max
        self.start_b, self.end_b = b_range
        self.decay_type = decay_type
        self.decay_shape = decay_shape
        self.decay_start = rel_decay_start * t_max

    def __call__(self, t):
        if t < self.decay_start:
            return self.start_b

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)
        if self.decay_type == AdaRoundTempDecayType.linear:
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
        elif self.decay_type == AdaRoundTempDecayType.cosine:
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
        elif self.decay_type == AdaRoundTempDecayType.sigmoid:
            d = self.decay_shape
            offset = sigmoid(-d / 2)
            rel_progress = (sigmoid(d * (rel_t - 0.5)) - offset) / (1 - 2 * offset)
            return self.start_b + (self.end_b - self.start_b) * rel_progress
        elif self.decay_type == AdaRoundTempDecayType.power:
            return self.end_b + (self.start_b - self.end_b) * (1 - rel_t ** self.decay_shape)
        elif self.decay_type == AdaRoundTempDecayType.exp:
            r = self.decay_shape
            rel_progress = (1.0 - np.exp(-r * rel_t)) / (1.0 - np.exp(-r))
            return self.start_b + (self.end_b - self.start_b) * rel_progress
        elif self.decay_type == AdaRoundTempDecayType.log:
            r = self.decay_shape
            C = np.exp(self.end_b / r)
            c = np.exp(self.start_b / r)
            return r * np.log((C - c) * rel_t + c)
        else:
            raise ValueError(f'Unknown temp decay type {self.decay_type}')


class CombinedLoss:
    def __init__(self, quantizer, loss_type=AdaRoundLossType.relaxation, weight=0.01,
                 max_count=1000, b_range=(20, 2), warmup=0.0, decay_start=0.0, **temp_decay_kw):
        self.quantizer = quantizer
        self.loss_type = loss_type
        self.weight = weight

        self.loss_start = max_count * warmup
        self.temp_decay = TempDecay(
            max_count,
            b_range=b_range,
            rel_decay_start=warmup + (1.0 - warmup) * decay_start,
            **temp_decay_kw,
        )
        self.iter = 0

    def __call__(self, pred, tgt, *args, **kwargs):
        self.iter += 1

        rec_loss = F.mse_loss(pred, tgt, reduction='none').sum(1).mean()

        if self.iter < self.loss_start:
            b = self.temp_decay(self.iter)
            round_loss = 0
        elif self.loss_type == AdaRoundLossType.temp_decay:
            b = self.temp_decay(self.iter)
            self.quantizer.temperature = b
            round_loss = 0
        elif self.loss_type == AdaRoundLossType.relaxation:  # 1 - |(h-0.5)*2|**b
            b = self.temp_decay(self.iter)
            round_vals = self.quantizer.get_rest().view(-1)
            round_loss = self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
        else:
            raise ValueError(f'Unknown loss type {self.loss_type}')

        total_loss = rec_loss + round_loss
        if self.iter == 1 or self.iter % 100 == 0:
            logger.info(
                f'Total loss:\t{total_loss:.4f} (rec:{rec_loss:.4f}, '
                f'round:{round_loss:.3f})\tb={b:.2f}\titer={self.iter}'
            )
        return total_loss


class StopForwardHook:
    def __call__(self, module, *args):
        raise StopForwardException


class DataSaverHook:
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model, layer, asym=False, act_quant=False, store_output=True):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = layer.weight.device
        self.act_quant = act_quant
        self.store_output = store_output
        self.data_saver = DataSaverHook(
            store_input=True, store_output=self.store_output, stop_forward=True
        )

    def __call__(self, model_input):
        self.model.full_precision()
        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

            if self.asym:  # recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.full_precision()
        self.layer.quantized_weights()
        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()


class LayerOutputMSE:
    def __init__(self, layer, get_inp_out, data_tensor, batch_size, name='mse_out'):
        cur_inp, cur_out = get_inp_out(data_tensor)
        self.input = cur_inp
        self.exp_out = cur_out
        self.layer = layer
        self.batch_size = batch_size
        self.name = name

    def __call__(self):
        loss = 0.0
        x = self.input
        for i in range(ceil(x.size(0) / self.batch_size)):
            cur_out = self.layer(x[i * self.batch_size:(i + 1) * self.batch_size])
            exp_out = self.exp_out[i * self.batch_size:(i + 1) * self.batch_size]
            loss += F.mse_loss(cur_out, exp_out).item()
        return loss
