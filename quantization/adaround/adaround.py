# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F

from quantization.adaround.quantizer import ADAROUND_QUANTIZER_MAP
from quantization.adaround.utils import (
    MODE_TO_LOSS_TYPE,
    AdaRoundInitMode,
    CombinedLoss,
    GetLayerInpOut,
    LayerOutputMSE,
)
from utils.utils import DotDict


# setup logger
logger = logging.getLogger('AdaRound')
logger.setLevel(logging.INFO)


def apply_adaround_to_layer(model, layer, data_tensor, batch_size, act_quant, adaround_config,
                            keep_gpu=True):
    """Apply AdaRound to a `layer` in the `model`."""

    # disable caching of quantized params
    layer.caching = False

    # grid initialization
    if adaround_config.init == AdaRoundInitMode.range_estimator:
        pass  # already initialized
    elif adaround_config.init == AdaRoundInitMode.mse:
        apply_mse_init(layer)
    elif adaround_config.init == AdaRoundInitMode.mse_out:
        apply_mse_out_init(model, layer, data_tensor, batch_size)
    elif adaround_config.init == AdaRoundInitMode.mse_out_asym:
        apply_mse_out_init(model, layer, data_tensor, batch_size, asym=True)
    else:
        raise ValueError(f'Unknown initialization for AdaRound: {adaround_config.init}')

    # activation function
    if not adaround_config.include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = None

    # replace quantizer with AdaRound quantizer
    org_w_quantizer = layer.weight_quantizer.quantizer
    org_w_quant_cls = org_w_quantizer.__class__

    if not org_w_quant_cls in ADAROUND_QUANTIZER_MAP:
        raise NotImplementedError(f'AdaRound is not supported for "{org_w_quant_cls}"')

    new_w_quant_cls = ADAROUND_QUANTIZER_MAP[org_w_quant_cls]
    w_quantizer = new_w_quant_cls(
        n_bits=org_w_quantizer.n_bits,
        scale_domain=org_w_quantizer.scale_domain,
        per_channel=org_w_quantizer.per_channel,
        eps=org_w_quantizer.eps,
    )
    w_quantizer.register_buffer('_delta', org_w_quantizer._delta)
    w_quantizer.register_buffer('_zero_float', org_w_quantizer._zero_float)
    if hasattr(org_w_quantizer, '_signed'):
        w_quantizer.register_buffer('_signed', org_w_quantizer._signed)
    layer.weight_quantizer.quantizer = w_quantizer

    # set AdaRound attributes
    w_quantizer.round_mode = adaround_config.round_mode
    w_quantizer.temperature = adaround_config.annealing[0]

    # single test (and init alpha)
    get_inp_out = GetLayerInpOut(model, layer, asym=adaround_config.asym, act_quant=act_quant)
    inp, out = get_inp_out(data_tensor[:batch_size])
    loss_soft_before, loss_hard_before = _compute_and_display_local_losses(
        w_quantizer, layer, inp, out, infix='before optimization'
    )
    w_quantizer.soft_targets = True

    # define loss
    loss_type = MODE_TO_LOSS_TYPE[w_quantizer.round_mode]
    loss_fn = CombinedLoss(
        quantizer=w_quantizer,
        loss_type=loss_type,
        weight=adaround_config.weight,
        max_count=adaround_config.iters,
        b_range=adaround_config.annealing,
        warmup=adaround_config.warmup,
        decay_type=adaround_config.decay_type,
        decay_shape=adaround_config.decay_shape,
        decay_start=adaround_config.decay_start,
    )

    # define optimizer
    opt_params = [w_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params, lr=adaround_config.lr)

    # main loop
    optimize_local_loss(
        layer,
        get_inp_out,
        data_tensor,
        optimizer,
        loss_fn,
        batch_size,
        adaround_config.iters,
        keep_gpu=keep_gpu,
    )

    # check afterwards
    logger.info(f'Local loss before optimization (hard quant): {loss_hard_before:.7f}')
    loss_soft_after, loss_hard_after = _compute_and_display_local_losses(
        w_quantizer, layer, inp, out, infix='after optimization'
    )

    # set to hard decision up/down
    w_quantizer.soft_targets = False

    # restore original activation function
    if not adaround_config.include_act_func:
        layer.activation_function = org_act_func

    # restore caching of quantized params
    layer.caching = True

    # prepare output
    out = DotDict(
        loss_soft_before=loss_soft_before,
        loss_hard_before=loss_hard_before,
        loss_soft_after=loss_soft_after,
        loss_hard_after=loss_hard_after,
    )
    return out


def _compute_and_display_local_losses(quantizer, layer, inp, out, infix=''):
    org_soft_targets = quantizer.soft_targets

    quantizer.soft_targets = True
    out_soft_quant = layer(inp)
    quantizer.soft_targets = False
    out_hard_quant = layer(inp)

    soft_quant_loss = F.mse_loss(out_soft_quant, out)
    hard_quant_loss = F.mse_loss(out_hard_quant, out)

    if infix:
        infix = infix.strip() + ' '

    logger.info(f'Local loss {infix}(soft quant): {soft_quant_loss:.7f}')
    logger.info(f'Local loss {infix}(hard quant): {hard_quant_loss:.7f}')

    quantizer.soft_targets = org_soft_targets
    return float(soft_quant_loss), float(hard_quant_loss)


def apply_mse_init(layer):
    w = layer.weight
    q = layer.weight_quantizer.quantizer

    with torch.no_grad():
        w_absmax = torch.max(w.max(), torch.abs(w.min()))
        best_score = np.inf
        best_max = w_absmax
        for i in range(80):
            s = w_absmax * (1.0 - 0.01 * i)
            q.set_quant_range(-s, s)
            score = F.mse_loss(w, q(w)).item()

            if score < best_score:
                best_score = score
                best_max = s

        logger.info(f'Finished: set max={best_max:.3f} (mse={best_score:.7f})')
        q.set_quant_range(-best_max, best_max)


def apply_mse_out_init(model, layer, data_tensor, batch_size, asym=False):
    w = layer.weight
    q = layer.weight_quantizer.quantizer

    get_inp_out = GetLayerInpOut(model, layer, asym=asym)
    loss_fn = LayerOutputMSE(layer, get_inp_out, data_tensor, batch_size)

    with torch.no_grad():
        w_absmax = torch.max(w.max(), torch.abs(w.min()))
        best_score = np.inf
        best_max = w_absmax
        for i in range(80):
            s = w_absmax * (1.0 - 0.01 * i)
            q.set_quant_range(-s, s)
            score = loss_fn()

            if score < best_score:
                best_score = score
                best_max = s
        logger.info(f'Finished: set max={best_max:.3f} (mse={best_score:.7f})')
        q.set_quant_range(-best_max, best_max)


def optimize_local_loss(layer, get_inp_out, data_tensor, optimizer, loss_fn, batch_size, iters,
                        use_cached_data=True, keep_gpu=True):
    """AdaRound optimization loop."""
    if use_cached_data:
        logger.info('Caching data for local loss optimization')

        cached_batches = []
        if keep_gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for i in range(ceil(data_tensor.size(0) / batch_size)):
                cur_inp, cur_out = get_inp_out(data_tensor[i * batch_size:(i + 1) * batch_size])
                cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

            cached_inps = torch.cat([x[0] for x in cached_batches])
            cached_outs = torch.cat([x[1] for x in cached_batches])
            device = cur_inp.device

            del cached_batches
            if keep_gpu:  # put all cached data on GPU for faster optimization
                torch.cuda.empty_cache()
                try:
                    cached_inps = cached_inps.to(device)
                    cached_outs = cached_outs.to(device)
                except RuntimeError as e:
                    logger.warning(
                        f"WARNING: could not cache training data on GPU, keep on CPU ({e})"
                    )
                    cached_inps = cached_inps.cpu()
                    cached_outs = cached_outs.cpu()

    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        if use_cached_data:
            cur_inp = cached_inps[idx].to(device)
            cur_out = cached_outs[idx].to(device)
        else:
            cur_inp, cur_out = get_inp_out(data_tensor[idx])

        optimizer.zero_grad()

        try:
            out_quant = layer(cur_inp)
            loss = loss_fn(out_quant, cur_out)
            loss.backward()
        except RuntimeError as e:
            if use_cached_data and 'cuda' in str(cached_inps.device):
                logger.warning(
                    f"WARNING: not enough CUDA memory for forward pass, "
                    f"move cached data to CPU ({e})"
                )
                cached_inps = cached_inps.cpu()
                cached_outs = cached_outs.cpu()
            else:
                raise e

        optimizer.step()
