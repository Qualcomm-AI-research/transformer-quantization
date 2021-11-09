# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from functools import wraps
from pathlib import Path

import click
import torch
from transformers.trainer_utils import EvaluationStrategy

from utils.hf_models import HF_Models
from utils.glue_tasks import GLUE_Task
from utils.quant_click_options import split_dict
from utils.utils import seed_all


def transformer_base_options(func):
    @click.option(
        '--cuda/--no-cuda', is_flag=True, default=torch.cuda.is_available(), help='Use GPU'
    )
    @click.option('--seed', type=int, default=1000, help='Random number generator seed to set.')
    @click.option(
        '--num-workers',
        type=int,
        default=0,
        help='Number of PyTorch data loader subprocesses. 0 means that the data will be '
        'loaded in the main process.',
    )
    @click.option(
        '--output-dir',
        default=None,
        type=click.Path(file_okay=False, writable=True, resolve_path=True),
        help='The output directory where the model predictions and checkpoints will be written. '
        'The model and the tokenizer will be saved in the `out` sub-folder.',
    )
    @click.option(
        '--overwrite-output',
        is_flag=True,
        default=False,
        help='Overwrite the content of the output directory and log file. Use this to '
        'continue training if output directory contains model checkpoint.',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = ['cuda', 'seed', 'num_workers', 'output_dir', 'overwrite_output']
        config.base, other_kw = split_dict(kwargs, attrs)
        seed = config.base.seed
        if seed is not None:
            seed_all(seed)  # must be set before initializing the model
        return func(config, *args, **other_kw)

    return func_wrapper


def transformer_data_options(func):
    @click.option(
        '--max-seq-length',
        type=int,
        default=128,
        help='The maximum total input sequence length after tokenization. Sequences '
        'longer than this will be truncated, sequences shorter will be padded.',
    )
    @click.option(
        '--pad-to-max-length/--no-pad-to-max-length',
        is_flag=True,
        default=True,
        help='Whether to pad all samples to `max_seq_length`. If False, will pad the '
        'samples dynamically when batching to the maximum length in the batch.',
    )
    @click.option('--line-by-line', is_flag=True, default=False)
    @click.option(
        '--overwrite-cache',
        is_flag=True,
        default=False,
        help='Overwrite the cached preprocessed datasets or not.',
    )
    @click.option('--num-train-samples', type=int, default=None)
    @click.option(
        '--num-val-samples',
        type=int,
        default=None,
        help='Number of samples to use for validation. If not specified, '
        'validate on the entire set(s).',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            'max_seq_length',
            'pad_to_max_length',
            'line_by_line',
            'overwrite_cache',
            'num_train_samples',
            'num_val_samples',
        ]

        config.data, other_kw = split_dict(kwargs, attrs)
        return func(config, *args, **other_kw)

    return func_wrapper


def glue_options(func):
    @click.option(
        '--task',
        type=click.Choice(GLUE_Task.list_names(), case_sensitive=False),
        default=(GLUE_Task.mrpc.name,),
        multiple=True,
        help='The name of the task to train on.',
    )
    @click.option(
        '--data-dir',
        default=str(Path.home() / '.glue_data'),
        type=click.Path(file_okay=False, writable=True, resolve_path=True),
        help='Directory where both raw and preprocessed GLUE datasets are stored.',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = ['task', 'data_dir']

        config.glue, other_kw = split_dict(kwargs, attrs)
        return func(config, *args, **other_kw)

    return func_wrapper


def transformer_model_options(func):
    # GLUE
    @click.option(
        '--model-name',
        type=click.Choice(HF_Models.list_names(), case_sensitive=False),
        default=HF_Models.bert_base_uncased.name,
        help='Model identifier from huggingface.co/models.',
    )
    @click.option(
        '--model-path',
        default=None,
        type=click.Path(exists=True, file_okay=False, resolve_path=True),
        help='For training (both FP32 and quantized), it is a path to a pretrained model, together '
        'with a tokenizer (can be used to resume training). For validation, it is a path that '
        'should contain fine-tuned checkpoints for all the requested tasks (each in a separate '
        'sub-folder named as a corresponding task).',
    )
    @click.option(
        '--quant-model-path',
        default=None,
        type=click.Path(exists=True, file_okay=False, resolve_path=True),
        help='State dict of quantized model.',
    )
    @click.option(
        '--use-fast-tokenizer',
        is_flag=True,
        default=True,
        help='Whether to use one of the fast tokenizer (backed by the HuggingFace '
        'tokenizers library) or not.',
    )
    @click.option(
        '--cache-dir',
        default=str(Path.home() / '.hf_cache'),
        type=click.Path(file_okay=False, writable=True, resolve_path=True),
        help='Where to store downloaded pretrained HuggingFace models (together with '
        'respective config and a tokenizer).',
    )
    @click.option(
        '--attn-dropout', default=None, type=float, help='Dropout rate to set for attention probs.'
    )
    @click.option(
        '--hidden-dropout', default=None, type=float, help='Dropout rate to set for hidden states.'
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            'model_name',
            'model_path',
            'quant_model_path',
            'use_fast_tokenizer',
            'cache_dir',
            'attn_dropout',
            'hidden_dropout',
        ]

        config.model, other_kw = split_dict(kwargs, attrs)
        return func(config, *args, **other_kw)

    return func_wrapper


def transformer_training_options(func):
    # standard settings
    @click.option(
        '--do-eval/--no-eval',
        is_flag=True,
        default=True,
        help='Whether to run eval on the dev set after training.',
    )
    @click.option('--batch-size', type=int, default=8, help='Batch size for training.')
    @click.option(
        '--eval-batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation. Defaults to the batch size for training.',
    )
    @click.option(
        '--learning-rate', type=float, default=5e-5, help='The initial learning rate for Adam.'
    )
    @click.option(
        '--lr-scheduler-type',
        default='cosine',
        type=click.Choice(['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                           'constant_with_warmup']),
        help='The scheduler type to use.',
    )
    @click.option('--weight-decay', type=float, default=0.0, help='Weight decay for AdamW.')
    @click.option(
        '--max-grad-norm',
        type=float,
        default=None,
        help='Max gradient norm. If set to 0, no clipping will be applied.',
    )
    @click.option(
        '--num-epochs', type=int, default=3, help='Total number of training epochs to perform.'
    )
    @click.option(
        '--max-steps',
        type=int,
        default=0,
        help='If > 0, set total number of training steps to perform. Overrides `num_epochs`.',
    )
    @click.option('--warmup-steps', type=int, default=0, help='Linear warmup over `warmup_steps`.')

    # hw optimizations
    @click.option(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    @click.option('--amp', is_flag=True, default=False, help='Whether to use Apex AMP.')
    @click.option(
        '--amp-opt-level',
        type=click.Choice(('O0', 'O1', 'O2', 'O3')),
        default='O2',
        help='Apex AMP optimization level.',
    )

    # custom regularization
    @click.option('--ffn-weight-decay', type=float, default=0, help='Weight decay for FFN weights.')
    @click.option('--gamma', type=float, default=0, help='Activation regularization strength.')
    @click.option('--margin', type=float, default=0, help='Activation regularization margin.')

    # custom functionality
    @click.option(
        '--save-attn',
        is_flag=True,
        default=False,
        help='Save attention probabilities from the training set and skip training.',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            'do_eval',
            'batch_size',
            'eval_batch_size',
            'learning_rate',
            'lr_scheduler_type',
            'weight_decay',
            'max_grad_norm',
            'num_epochs',
            'max_steps',
            'warmup_steps',
            'gradient_accumulation_steps',
            'amp',
            'amp_opt_level',
            'ffn_weight_decay',
            'gamma',
            'margin',
            'save_attn',
        ]

        config.training, other_kw = split_dict(kwargs, attrs)
        if config.training.eval_batch_size is None:
            config.training.eval_batch_size = config.training.batch_size

        return func(config, *args, **other_kw)

    return func_wrapper


def transformer_progress_options(func):
    @click.option('--tqdm/--no-tqdm', default=True)
    @click.option(
        '--eval-during-training',
        is_flag=True,
        default=False,
        help='Run evaluation during training at each logging step.',
    )
    @click.option(
        '--eval-strategy',
        default=EvaluationStrategy.NO.value,
        type=click.Choice([m.value for m in EvaluationStrategy], case_sensitive=False),
        help='Evaluation frequency level.',
    )
    @click.option(
        '--eval-steps', type=int, default=None, help='Run an evaluation every `eval_steps` steps.'
    )
    @click.option(
        '--tb-logging-dir',
        default=None,
        type=click.Path(exists=False, writable=True, resolve_path=True),
        help='Tensorboard log dir.',
    )
    @click.option(
        '--tb',
        is_flag=True,
        default=False,
        help='Whether to create and log (additional) stuff to the TensorBoard writer',
    )
    @click.option(
        '--tb-graph',
        is_flag=True,
        default=False,
        help='Whether to log computational graph into the TensorBoard writer',
    )
    @click.option(
        '--logging-first-step',
        is_flag=True,
        default=False,
        help='Log and eval the first global_step.',
    )
    @click.option(
        '--logging-steps', type=int, default=500, help='Log every `logging_steps` updates steps.'
    )
    @click.option(
        '--save-steps',
        type=int,
        default=0,
        help='Save checkpoint every `save_steps` updates steps.',
    )
    @click.option(
        '--save-total-limit',
        type=int,
        default=None,
        help='Limit the total amount of checkpoints. Deletes the older checkpoints in '
        'the `output_dir`. Default is unlimited checkpoints.',
    )
    @click.option(
        '--save-model',
        is_flag=True,
        default=False,
        help='Whether to save model and tokenizer after the training.',
    )
    @click.option(
        '--run-name',
        type=str,
        default=None,
        help='An optional descriptor for the run. Notably used for wandb logging.',
    )
    @click.option(
        '--load-best-model-at-end',
        is_flag=True,
        default=False,
        help='Whether or not to load the best model found during training at the end of '
        'training.',
    )
    @click.option(
        '--metric-for-best-model',
        type=str,
        default=None,
        help='The metric to use to compare two different models.',
    )
    @click.option(
        '--greater-is-better',
        type=bool,
        default=None,
        help='Whether the `metric_for_best_model` should be maximized or not.',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            'tqdm',
            'eval_during_training',
            'eval_strategy',
            'eval_steps',
            'tb_logging_dir',
            'tb',
            'tb_graph',
            'logging_first_step',
            'logging_steps',
            'save_steps',
            'save_total_limit',
            'save_model',
            'run_name',
            'load_best_model_at_end',
            'metric_for_best_model',
            'greater_is_better',
        ]

        config.progress, other_kw = split_dict(kwargs, attrs)
        return func(config, *args, **other_kw)

    return func_wrapper


def transformer_quant_options(func):
    @click.option(
        '--est-ranges-pad/--est-ranges-no-pad',
        is_flag=True,
        default=None,
        help='Specify whether to pad to max sequence length during range estimation.'
        'If None, inherit the value of --pad-to-max-length.',
    )
    @click.option(
        '--est-ranges-batch-size',
        type=int,
        default=None,
        help='Batch size for range estimation. Defaults to the batch size for training.',
    )
    @click.option('--quant-dict', type=str, default=None)
    @click.option('--double', is_flag=True)
    @click.option('--dynamic', is_flag=True)
    @click.option('--per-token', is_flag=True)
    @click.option('--per-embd', is_flag=True)
    @click.option('--per-groups', type=int, default=None)
    @click.option('--per-groups-permute', is_flag=True)
    @click.option('--per-groups-permute-shared-h', is_flag=True)
    @wraps(func)
    def func_wrapper(config, est_ranges_pad, est_ranges_batch_size, quant_dict, double, dynamic,
                     per_token, per_embd, per_groups, per_groups_permute,
                     per_groups_permute_shared_h, *a, **kw):
        config.quant.est_ranges_pad = est_ranges_pad
        config.quant.est_ranges_batch_size = (
            est_ranges_batch_size if est_ranges_batch_size is not None
            else config.training.batch_size
        )

        if quant_dict is not None:
            quant_dict = eval(quant_dict)
            config.quant.quant_dict = quant_dict
        config.double = double

        config.quant.dynamic = dynamic
        config.quant.per_token = per_token
        if config.quant.per_token:
            config.quant.dynamic = True

        config.quant.per_embd = per_embd
        config.quant.per_groups = per_groups
        config.quant.per_groups_permute = per_groups_permute
        config.quant.per_groups_permute_shared_h = per_groups_permute_shared_h

        return func(config, *a, **kw)

    return func_wrapper
