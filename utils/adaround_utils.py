# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

import torch

from quantization.adaround import apply_adaround_to_layer
from quantization.adaround.utils import AdaRoundActQuantMode
from quantization.base_quantized_classes import QuantizedModule
from utils.utils import pass_data_for_range_estimation, Stopwatch


# setup logger
logger = logging.getLogger('AdaRound')
logger.setLevel(logging.INFO)


def get_train_samples(data_loader, num_samples, return_labels=False, inp_idx=0, lbl_idx=1):
    X, y = [], []
    for data in data_loader:
        X.append(data[inp_idx])
        if return_labels:
            y.append(data[lbl_idx])
        if len(X) * data[inp_idx].size(0) >= num_samples:
            break

    X = torch.cat(X, dim=0)[:num_samples]
    if return_labels:
        y = torch.cat(y, dim=0)[:num_samples]
        return X, y
    return X


def apply_adaround_to_model(config, model, data_loader, range_est_data_loader, batch_size,
                            driver=None, get_samples_fn=get_train_samples, inp_idx=0):
    """
    Apply AdaRound to `model`.

    Parameters
    ----------
    config : DotDict
        DotDict with quantization parameters
    model : QuantizedModel
        model to apply AdaRound on
    data_loader : torch.utils.data.DataLoader
        Training or other data used for AdaRound optimization
    driver : SupervisedDriver
        Used for validation. This is only used fore reporting accuracy, not for optimization
    inp_idx : int, str
        batch index in the input data from the dataloader
    """
    train_data = get_samples_fn(data_loader, num_samples=config.adaround.num_samples)

    device = next(model.parameters()).device
    train_data = train_data.to(device)

    # check and prepare list of layers to optimize for
    all_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedModule) and hasattr(module, 'weight'):
            all_layer_names.append(name)

    if 'all' in config.adaround.layers:
        adaround_layer_names = all_layer_names
    else:
        adaround_layer_names = []
        for name in config.adaround.layers:
            if name in all_layer_names:
                adaround_layer_names.append(name)
            else:
                logger.warning(f'skipping unknown layer {name}')

    if not len(adaround_layer_names):
        logger.warning('No layers to apply AdaRound for, exiting...')
        return

    # deal with activation quantization
    if config.adaround.act_quant_mode in (
        AdaRoundActQuantMode.no_act_quant,
        AdaRoundActQuantMode.post_adaround,
    ):
        config.quant.act_quant = False
        model.reset_act_ranges()
        model.full_precision_acts()
    else:
        raise NotImplementedError(f"act mode '{config.adaround.act_quant_mode}' is not implemented")

    # main loop
    s_all = Stopwatch()
    for name, module in model.named_modules():
        if not name in adaround_layer_names:
            continue

        logger.info(f'Started AdaRound for layer {name}')

        model.full_precision()
        module.quantized_weights()

        s_all.start()
        with Stopwatch() as s_layer:
            apply_adaround_to_layer(
                model,
                module,
                train_data,
                batch_size=batch_size,
                act_quant=config.quant.act_quant,
                adaround_config=config.adaround,
            )
        logger.info(f'Done AdaRound for layer {name}. {s_layer.format()}\n')
        s_all.stop()

    s_all.stop()
    logger.info(f'Done optimizing all layers. {s_all.format()}')

    if config.adaround.act_quant_mode == AdaRoundActQuantMode.post_adaround:
        if driver is not None:
            # validate before activation quantization
            model.quantized_weights()
            state = driver.validate()
            acc_quant = state.metrics['top_1_accuracy']
            logger.info(f'FINAL res (without acts quant):\t{acc_quant * 100:.2f}%')

        # activate activation quantization and estimate ranges
        config.quant.act_quant = True
        model.estimate_act_ranges()
        pass_data_for_range_estimation(
            loader=range_est_data_loader,
            model=model,
            act_quant=True,
            weight_quant=True,
            max_num_batches=config.act_quant.num_batches,
            cross_entropy_layer=config.act_quant.cross_entropy_layer,
            inp_idx=inp_idx,
        )
        model.fix_act_ranges()

    # set state
    model.set_quant_state(weight_quant=True, act_quant=config.quant.act_quant)
