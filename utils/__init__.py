# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from utils.adaround_utils import apply_adaround_to_model
from utils.glue_tasks import GLUE_Task, TASK_TO_FINAL_METRIC, load_task_data, make_compute_metric_fn
from utils.hf_models import HF_Models, load_model_and_tokenizer
from utils.per_embd_quant_utils import (
    hijack_act_quant,
    hijack_weight_quant,
    hijack_act_quant_modules,
    set_act_quant_axis_and_groups,
)
from utils.qat_utils import prepare_model_for_quantization
from utils.quant_click_options import (
    quantization_options,
    activation_quantization_options,
    qat_options,
    adaround_options,
    make_qparams,
    split_dict,
)
from utils.tb_utils import _tb_advance_global_step, _tb_advance_token_counters, _tb_hist
from utils.transformer_click_options import (
    glue_options,
    transformer_base_options,
    transformer_data_options,
    transformer_model_options,
    transformer_training_options,
    transformer_progress_options,
    transformer_quant_options,
)
from utils.utils import (
    seed_all,
    count_params,
    count_embedding_params,
    pass_data_for_range_estimation,
    DotDict,
    Stopwatch,
    StopForwardException,
)
