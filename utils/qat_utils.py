# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from utils.utils import pass_data_for_range_estimation


# setup logger
logger = logging.getLogger('QAT')
logger.setLevel('INFO')


def prepare_model_for_quantization(config, model, loader):

    # estimate ranges using training data
    pass_data_for_range_estimation(
        loader=loader,
        model=model,
        act_quant=config.quant.act_quant,
        weight_quant=config.quant.weight_quant,
        max_num_batches=config.act_quant.num_batches,
        cross_entropy_layer=config.act_quant.cross_entropy_layer,
    )

    # put quantizers in desirable state
    if config.qat.learn_ranges:
        logger.info('Make quantizers learnable')
        model.learn_ranges()
    else:
        logger.info(
            f'Fix quantizer ranges to fixW={config.qat.fix_weight_ranges} and '
            f'fixA={config.qat.fix_act_ranges}'
        )

        # freeze quantization ranges if applicable
        model.estimate_ranges_train()  # we use updating ranges in training as default
        if config.qat.fix_weight_ranges:
            model.fix_weight_ranges()
        if config.qat.fix_act_ranges:
            model.fix_act_ranges()

    # ensure we have the desired quant state
    model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)
    return model
