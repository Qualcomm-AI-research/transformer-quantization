# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from collections import namedtuple
from enum import Enum

import torch
from torch.autograd import Function
from torch import nn


class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class FloorStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


round_ste_func = RoundStraightThrough.apply
floor_ste_func = FloorStraightThrough.apply


class QuantizerBase(nn.Module):
    def __init__(self, n_bits, per_channel=False, axis=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.axis = axis

    @property
    def is_initialized(self):
        raise NotImplementedError()

    @property
    def x_max(self):
        raise NotImplementedError()

    @property
    def symmetric(self):
        raise NotImplementedError()

    @property
    def x_min(self):
        raise NotImplementedError()

    def forward(self, x_float):
        raise NotImplementedError()

    def _adjust_params_per_axis(self, x):
        raise NotImplementedError()

    def _adjust_params_per_channel(self, x):
        raise NotImplementedError()

    def set_quant_range(self, x_min, x_max):
        raise NotImplementedError()

    def extra_repr(self):
        return (
            f'n_bits={self.n_bits}, per_channel={self.per_channel}, axis={self.axis}, '
            f'is_initalized={self.is_initialized}'
        )

    def reset(self):
        self._delta = None


class AsymmetricUniformQuantizer(QuantizerBase):
    """
    PyTorch Module that implements Asymmetric Uniform Quantization using STE.
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel quantization
    """
    def __init__(self, n_bits, scale_domain='linear', per_channel=False, axis=None, eps=1e-8):

        super().__init__(n_bits, per_channel)

        assert scale_domain in ('linear', 'log')
        self.register_buffer('_delta', None)
        self.register_buffer('_zero_float', None)
        self.n_bits = n_bits
        self.scale_domain = scale_domain
        self.per_channel = per_channel
        self.axis = axis
        self.eps = eps

    # A few useful properties
    @property
    def delta(self):
        if self._delta is not None:
            return self._delta
        else:
            raise QuantizerNotInitializedError()

    @property
    def zero_float(self):
        if self._zero_float is not None:
            return self._zero_float
        else:
            raise QuantizerNotInitializedError()

    @property
    def is_initialized(self):
        return self._delta is not None

    @property
    def symmetric(self):
        return False

    @property
    def int_min(self):
        # integer grid minimum
        return 0.0

    @property
    def int_max(self):
        # integer grid maximum
        return 2.0 ** self.n_bits - 1

    @property
    def scale(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta)

    @property
    def zero_point(self):
        zero_point = round_ste_func(self.zero_float)
        zero_point = torch.clamp(zero_point, self.int_min, self.int_max)
        return zero_point

    @property
    def x_max(self):
        return self.scale * (self.int_max - self.zero_point)

    @property
    def x_min(self):
        return self.scale * (self.int_min - self.zero_point)

    def _clamp(self, x_int):
        with torch.no_grad():
            clampled_left = (x_int > self.int_max).float().sum()
            clampled_right = (x_int < self.int_min).float().sum()
            self._clamped = (clampled_left + clampled_right) / x_int.numel()
        x_clamped = torch.clamp(x_int, self.int_min, self.int_max)

        return x_clamped

    def to_integer_forward(self, x_float):
        """
        Qunatized input to its integer represantion
        Parameters
        ----------
        x_float: PyTorch Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_int: PyTorch Float Tensor of integers
        """
        x_int = round_ste_func(x_float / self.scale) + self.zero_point
        x_int = torch.clamp(x_int, self.int_min, self.int_max)

        return x_int

    def forward(self, x_float):
        """
        Quantizes (quantized to integer and the scales back to original domain)
        Parameters
        ----------
        x_float: PyTorch Float Tensor
            Full-precision Tensor

        Returns
        -------
        x_quant: PyTorch Float Tensor
            Quantized-Dequantized Tensor
        """
        if self.axis is not None:
            self._adjust_params_per_axis(x_float)

        if self.per_channel:
            self._adjust_params_per_channel(x_float)

        x_int = self.to_integer_forward(x_float)
        x_quant = self.scale * (x_int - self.zero_point)

        return x_quant

    def _adjust_params_per_axis(self, x_float):
        r = len(x_float.size())
        new_shape = [1] * self.axis + [-1] + [1] * (r - self.axis - 1)
        self._delta = self._delta.view(new_shape)
        self._zero_float = self._zero_float.view(new_shape)

    def _adjust_params_per_channel(self, x):
        """
        Adjusts the quantization parameter tensors (delta, zero_float)
        to the input tensor shape if they don't match

        Parameters
        ----------
        x: input tensor
        """
        if x.ndim != self.delta.ndim:
            new_shape = [-1] + [1] * (len(x.shape) - 1)
            self._delta = self.delta.view(new_shape)
            if self._zero_float is not None:
                self._zero_float = self._zero_float.view(new_shape)

    def _tensorize_min_max(self, x_min, x_max):
        """
        Converts provided min max range into tensors
        Parameters
        ----------
        x_min: float or PyTorch 1D tensor
        x_max: float or PyTorch 1D tensor

        Returns
        -------
        x_min: PyTorch Tensor 0 or 1-D
        x_max: PyTorch Tensor 0 or 1-D
        """
        # Ensure a torch tensor
        if not torch.is_tensor(x_min):
            x_min = torch.tensor(x_min).float()
            x_max = torch.tensor(x_max).float()

        if x_min.dim() > 0 and len(x_min) > 1 and not self.per_channel and self.axis is None:
            raise ValueError(
                'x_min and x_max must be a float or 1-D Tensor'
                ' for per-tensor quantization (per_channel=False)'
            )
        # Ensure we always use zero and avoid division by zero
        x_min = torch.min(x_min, torch.zeros_like(x_min))
        x_max = torch.max(x_max, torch.ones_like(x_max) * self.eps)

        return x_min, x_max

    def set_quant_range(self, x_min, x_max):
        """
        Instantiates the quantization parameters based on the provided
        min and max range

        Parameters
        ----------
        x_min: tensor or float
                Quantization range minimum limit
        x_max: tensor of float
                Quantization range minimum limit
        """
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        self._delta = (x_max - x_min) / self.int_max
        self._zero_float = (-x_min / self.delta).detach()

        if self.scale_domain == 'log':
            self._delta = torch.log(self.delta)

        self._delta = self._delta.detach()

    def make_range_trainable(self):
        # Converts trainable parameters to nn.Parameters
        if self.delta not in self.parameters():
            self._delta = torch.nn.Parameter(self._delta)
            self._zero_float = torch.nn.Parameter(self._zero_float)


class SymmetricUniformQuantizer(AsymmetricUniformQuantizer):
    """
    PyTorch Module that implements Symmetric Uniform Quantization using STE.
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel quantization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('_signed', None)

    @property
    def signed(self):
        if self._signed is not None:
            return self._signed.item()
        else:
            raise QuantizerNotInitializedError()

    @property
    def symmetric(self):
        return True

    @property
    def int_min(self):
        return -(2.0 ** (self.n_bits - 1)) if self.signed else 0

    @property
    def int_max(self):
        pos_n_bits = self.n_bits - self.signed
        return 2.0 ** pos_n_bits - 1

    @property
    def zero_point(self):
        return 0.0

    def set_quant_range(self, x_min, x_max):
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        self._signed = x_min.min() < 0

        x_absmax = torch.max(x_min.abs(), x_max)
        self._delta = x_absmax / self.int_max

        if self.scale_domain == 'log':
            self._delta = torch.log(self._delta)

        self._delta = self._delta.detach()

    def make_range_trainable(self):
        # Converts trainable parameters to nn.Parameters
        if self.delta not in self.parameters():
            self._delta = torch.nn.Parameter(self._delta)


QMethodMap = namedtuple('QMethodMap', ['value', 'cls'])


class QMethods(Enum):
    symmetric_uniform = QMethodMap(0, SymmetricUniformQuantizer)
    asymmetric_uniform = QMethodMap(1, AsymmetricUniformQuantizer)

    @property
    def cls(self):
        return self.value.cls

    @classmethod
    def list(cls):
        return [m.name for m in cls]


class QuantizerNotInitializedError(Exception):
    """Raised when a quantizer has not initialized"""

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__('Quantizer has not been initialized yet')
