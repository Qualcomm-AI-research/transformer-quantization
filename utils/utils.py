# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn

from quantization.range_estimators import RangeEstimators


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))


def count_embedding_params(model):
    return sum(count_params(m) for m in model.modules() if isinstance(m, nn.Embedding))


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


def pass_data_for_range_estimation(
    loader, model, act_quant, weight_quant, max_num_batches=20, cross_entropy_layer=None, inp_idx=0
):
    model.set_quant_state(weight_quant, act_quant)
    model.eval()

    if cross_entropy_layer is not None:
        layer_xent = get_layer_by_name(model, cross_entropy_layer)
        if layer_xent:
            print(f'Set cross entropy estimator for layer "{cross_entropy_layer}"')
            act_quant_mgr = layer_xent.activation_quantizer
            act_quant_mgr.range_estimator = RangeEstimators.cross_entropy.cls(
                per_channel=act_quant_mgr.per_channel,
                quantizer=act_quant_mgr.quantizer,
                **act_quant_mgr.init_params,
            )
        else:
            raise ValueError('Cross-entropy layer not found')

    device = next(model.parameters()).device
    for i, data in enumerate(loader):
        try:
            if isinstance(data, (tuple, list)):
                x = data[inp_idx].to(device=device)
                model(x)
            else:
                x = {k: v.to(device=device) for k, v in data.items()}
                model(**x)
        except StopForwardException:
            pass

        if i >= max_num_batches - 1 or not act_quant:
            break


class DotDict(dict):
    """
    A dictionary that allows attribute-style access.

    Examples
    --------
    >>> config = DotDict(a=None)
    >>> config.a = 42
    >>> config.b = 'egg'
    >>> config  # can be used as dict
    {'a': 42, 'b': 'egg'}
    """
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")


class Stopwatch:
    """
    A simple cross-platform context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as st:
    ...     time.sleep(0.101)  #doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    """
    def __init__(self, name=None, verbose=False):
        self._name = name
        self._verbose = verbose

        self._start_time_point = 0.0
        self._total_duration = 0.0
        self._is_running = False

        if sys.platform == 'win32':
            # on Windows, the best timer is time.clock()
            self._timer_fn = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_fn = time.time

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._verbose:
            self.print()

    def start(self):
        if not self._is_running:
            self._start_time_point = self._timer_fn()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total_duration += self._timer_fn() - self._start_time_point
            self._is_running = False
        return self

    def reset(self):
        self._start_time_point = 0.0
        self._total_duration = 0.0
        self._is_running = False
        return self

    def _update_state(self):
        now = self._timer_fn()
        self._total_duration += now - self._start_time_point
        self._start_time_point = now

    def _format(self):
        prefix = f'[{self._name}]' if self._name is not None else 'Elapsed time'
        info = f'{prefix}: {self._total_duration:.3f} sec'
        return info

    def format(self):
        if self._is_running:
            self._update_state()
        return self._format()

    def print(self):
        print(self.format())

    def get_total_duration(self):
        if self._is_running:
            self._update_state()
        return self._total_duration
