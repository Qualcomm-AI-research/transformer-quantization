#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
import random
import warnings

warnings.filterwarnings('ignore')  # ignore TF warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from pprint import pformat

import click
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.integrations import TensorBoardCallback

from models import (
    QuantizedBertForSequenceClassification,
    QuantizedMobileBertForSequenceClassification,
    QuantizedRobertaForSequenceClassification,
)
from quantization.adaround import AdaRoundActQuantMode
from utils import (
    # click options
    quantization_options,
    activation_quantization_options,
    qat_options,
    adaround_options,
    make_qparams,
    glue_options,
    transformer_base_options,
    transformer_data_options,
    transformer_model_options,
    transformer_training_options,
    transformer_progress_options,
    transformer_quant_options,

    # quantization
    apply_adaround_to_model,
    prepare_model_for_quantization,
    pass_data_for_range_estimation,
    hijack_act_quant,
    hijack_weight_quant,
    hijack_act_quant_modules,
    set_act_quant_axis_and_groups,

    # pipeline
    load_model_and_tokenizer,
    load_task_data,
    make_compute_metric_fn,
    HF_Models,
    GLUE_Task,
    TASK_TO_FINAL_METRIC,

    # misc
    DotDict,
    Stopwatch,
)


# setup logger
logger = logging.getLogger('main')
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))


# setup stuff
class Config(DotDict):
    pass


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
def glue():
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))


# show default values for all options
click.option = partial(click.option, show_default=True)


def _is_non_empty_dir(path):
    return path.exists() and len(list(path.iterdir()))


def _make_huggingface_training_args(config):
    """Create Training Arguments as required by HuggingFace Trainer."""
    output_dir = config.base.output_dir
    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'out')

    tb_logging_dir = config.progress.tb_logging_dir
    if tb_logging_dir is None:
        if config.base.output_dir is not None:
            tb_logging_dir = os.path.join(config.base.output_dir, 'tb_logs')
        else:
            tb_logging_dir = None

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=config.base.overwrite_output,
        seed=config.base.seed,
        dataloader_num_workers=config.base.num_workers,
        do_train=config.training.do_train,
        do_eval=config.training.do_eval,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        disable_tqdm=not config.progress.tqdm,
        evaluation_strategy=config.progress.eval_strategy,
        eval_steps=config.progress.eval_steps,
        logging_dir=tb_logging_dir,
        logging_first_step=config.progress.logging_first_step,
        logging_steps=config.progress.logging_steps,
        save_steps=config.progress.save_steps,
        save_total_limit=config.progress.save_total_limit,
        run_name=config.progress.run_name,
        load_best_model_at_end=config.progress.load_best_model_at_end,
        metric_for_best_model=config.progress.metric_for_best_model,
        greater_is_better=config.progress.greater_is_better,
    )
    return args


def _make_datasets_and_trainer(config, model, model_enum, tokenizer, task, task_data,
                               compute_metrics, training_args, padding=None):
    # define padding strategy
    if padding:
        padding = 'max_length'
    if padding is None:
        padding = 'max_length' if config.data.pad_to_max_length else False
        # if False, pad later, dynamically at batch creation,
        # to the max sequence length in each batch

    max_length = config.data.max_seq_length

    # tokenize text and define datasets
    def preprocess_fn(examples):
        # tokenize the texts
        args = (
            (examples[task_data.sentence1_key],)
            if task_data.sentence2_key is None
            else (examples[task_data.sentence1_key], examples[task_data.sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
        return result

    datasets = task_data.datasets.map(
        preprocess_fn, batched=True, load_from_cache_file=not config.data.overwrite_cache
    )

    train_dataset = datasets['train']
    eval_dataset = datasets['validation_matched' if task == GLUE_Task.mnli else 'validation']

    if model_enum in (
        HF_Models.bert_base_uncased,
        HF_Models.bert_large_uncased,
        HF_Models.bert_base_cased,
        HF_Models.mobilebert_uncased,
    ):
        logger.info('First ten examples tokenized: (#, [SEP] idx, length):')
        for i in range(10):
            tokens = tokenizer.convert_ids_to_tokens(eval_dataset[i]['input_ids'])
            sep_pos_idx = tokens.index('[SEP]')
            len_ = len(tokens)
            logger.info(f'{i + 1}, {sep_pos_idx}, {len_}, {tokens}')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # data collator will default to DataCollatorWithPadding,
        # so we change it if we already did the padding:
        data_collator=default_data_collator if padding else None,
    )
    return trainer, datasets, train_dataset, eval_dataset


def _log_results(task_scores_map):
    if any([v is not None for v in task_scores_map.values()]):
        logger.info('*** FINAL results (task -> score) ***')
        all_scores = []
        all_scores_excluding_wnli = []

        for task, score in task_scores_map.items():
            logger.info(f'\t{task.name} -> {100. * score:.2f}')
            all_scores.append(score)
            if task != GLUE_Task.wnli:
                all_scores_excluding_wnli.append(score)

        logger.info(f'Macro-avg (incl. WNLI) = {100. * np.mean(all_scores):.2f}')
        if len(all_scores_excluding_wnli):
            logger.info(
                f'Macro-avg (excl. WNLI) = ' f'{100. * np.mean(all_scores_excluding_wnli):.2f}'
            )


def _quantize_model(config, model, model_enum):
    qparams = make_qparams(config)
    qparams['quant_dict'] = config.quant.get('quant_dict', {})

    if model_enum in (HF_Models.bert_base_uncased, HF_Models.bert_large_uncased):
        model = QuantizedBertForSequenceClassification(model, **qparams)
    elif model_enum == HF_Models.mobilebert_uncased:
        model = QuantizedMobileBertForSequenceClassification(model, **qparams)
    elif model_enum in (HF_Models.distilroberta_base, HF_Models.roberta_base):
        model = QuantizedRobertaForSequenceClassification(model, **qparams)
    else:
        raise NotImplementedError(
            f'Model {config.model.model_name} is not supported for ' f'quantization.'
        )

    # use double precision if necessary
    if config.double:
        for m in model.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                m.double()

    # set state
    model.set_quant_state(weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant)

    # print model
    logger.info('Quantized model:')
    logger.info(model)

    return model


def _prepare_quantized_model(config, model, loader):
    """Prepare quantized model for training/validation."""
    if config.training.do_train:
        model = prepare_model_for_quantization(config, model, loader)

    else:
        if not config.quant.dynamic:
            # 1) estimate & fix ranges for validation
            logger.info('** Estimate quantization ranges on training data **')
            pass_data_for_range_estimation(
                loader=loader,
                model=model,
                act_quant=config.quant.act_quant,
                weight_quant=config.quant.weight_quant,
                max_num_batches=config.act_quant.num_batches,
                cross_entropy_layer=config.act_quant.cross_entropy_layer,
            )
            model.fix_ranges()

        # 2) set quant state
        model.set_quant_state(
            weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
        )
    return model


class TransformerInput(tuple):
    def __getitem__(self, index):
        return TransformerInput([t[index] for t in self])

    def to(self, device):
        out = []
        for v in self:
            out.append(v.to(device) if isinstance(v, torch.Tensor) else v)
        return TransformerInput(out)

    def size(self, *args, **kw):
        out = []
        for v in self:
            out.append(v.size(*args, **kw))
        return out[0]


def adaround_get_samples_fn(data_loader, num_samples):
    X_dict = {}
    n = 0
    m = None
    for x_dict in data_loader:
        for i, (k, v) in enumerate(x_dict.items()):
            if i == 0:
                if n + len(v) > num_samples:
                    m = num_samples - n
                    n = num_samples
                else:
                    n += len(v)
            if m is not None:
                v = v[:m]
            if k in X_dict:
                X_dict[k].append(v)
            else:
                X_dict[k] = [v]
        if n == num_samples:
            break
    for k, v in X_dict.items():
        X_dict[k] = torch.cat(v)

    inp_tuple = (X_dict['input_ids'], X_dict['attention_mask'])
    if 'token_type_ids' in X_dict:
        inp_tuple = inp_tuple + (X_dict['token_type_ids'],)
    train_data = TransformerInput(inp_tuple)
    return train_data


def _run_task(config, task: GLUE_Task, task_data, model_data):
    """Common routine to run training/validation on a signle task."""
    model = model_data.model
    model_enum = model_data.model_enum
    tokenizer = model_data.tokenizer

    # log options
    logger.info(f'Running task {task.name} with options:\n' + pformat(config))

    if config.training.do_train:
        # create dirpath if not exist
        os.makedirs(config.base.output_dir, exist_ok=True)

        # log config additionaly into a separate file
        with open(os.path.join(config.base.output_dir, 'config.out'), 'w') as f:
            f.write(pformat(config) + '\n')

    # get metric
    compute_metrics = make_compute_metric_fn(task)

    # prepare training arguments for huggingface Trainer
    training_args = _make_huggingface_training_args(config)
    logger.info(f'Training/evaluation parameters for Trainer: {training_args}')

    ## attach layer number
    backbone_attr = model_data.backbone_attr
    if backbone_attr is None:
        raise NotImplementedError(
            f'Model {config.model.model_name} not yet supported for ' f'TensorBoard visualization.'
        )
    layers = getattr(model, backbone_attr).encoder.layer
    num_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        for m in layer.modules():
            m.layer_idx = layer_idx
            m.num_layers = num_layers

    # Quantization!
    if 'quant' in config:
        # replace model with a quantized one
        model = _quantize_model(config, model, model_enum)

    # Per-embedding / per-token quantization
    per_token = config.get('quant', {}).get('per_token', False)
    per_embd = config.get('quant', {}).get('per_embd', False)
    per_groups = config.get('quant', {}).get('per_groups', None)
    permute = config.get('quant', {}).get('per_groups_permute', False)
    base_axis = 2 if (per_embd or per_groups) else 1

    if (per_token or per_embd or per_groups) and model_enum in (
        HF_Models.bert_base_uncased,
        HF_Models.bert_large_uncased,
    ):
        # Per-embedding:
        # * for shapes (B, T, d) -> axis=2
        # * for shapes (B, d) -> axis=1
        # * for other shapes not applicable: (B, H, T, T); (B, T, D); (B, K)

        # Per-token:
        # * for shapes (B, T, d), (B, T, D) -> axis=1
        # * for other shapes not applicable: (B, H, T, T); (B, d); (B, K)

        # Embeddings
        E = model.bert.embeddings
        set_act_quant_axis_and_groups(
            E.sum_input_token_type_embd_act_quantizer,
            axis=base_axis,
            n_groups=per_groups,
            permute=permute,
        )
        set_act_quant_axis_and_groups(
            E.sum_pos_embd_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
        )
        set_act_quant_axis_and_groups(
            E.LayerNorm, axis=base_axis, n_groups=per_groups, permute=permute
        )

        # Encoder
        for layer_idx in range(12):
            L = model.bert.encoder.layer[layer_idx]

            # Self-attention
            A = L.attention.self
            set_act_quant_axis_and_groups(
                A.query, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                A.key, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                A.value, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                A.context_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
            )

            # Self-output
            S = L.attention.output
            set_act_quant_axis_and_groups(
                S.dense, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                S.res_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                S.LayerNorm, axis=base_axis, n_groups=per_groups, permute=permute
            )

            # Output
            O = L.output
            set_act_quant_axis_and_groups(
                O.dense, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                O.res_act_quantizer, axis=base_axis, n_groups=per_groups, permute=permute
            )
            set_act_quant_axis_and_groups(
                O.LayerNorm, axis=base_axis, n_groups=per_groups, permute=permute
            )

        # Pooling, (B, d)
        if per_embd:
            set_act_quant_axis_and_groups(
                model.bert.pooler.dense_act[0], axis=1, n_groups=per_groups, permute=permute
            )

    # Mixed-precision control for act. quantizers
    quant_dict = config.get('quant', {}).get('quant_dict', {})
    if quant_dict and model_enum in (HF_Models.bert_base_uncased, HF_Models.bert_large_uncased):
        # Embeddings
        E = model.bert.embeddings
        hijack_act_quant(quant_dict, 'e', E.sum_input_token_type_embd_act_quantizer)
        hijack_act_quant(quant_dict, 'e', E.sum_pos_embd_act_quantizer)

        hijack_weight_quant(quant_dict, 'Et', E.word_embeddings)

        # Encoder
        for layer_idx in range(12):
            L = model.bert.encoder.layer[layer_idx]

            # Self-attention
            A = L.attention.self
            hijack_act_quant(quant_dict, f's{layer_idx}', A.attn_scores_act_quantizer)
            hijack_act_quant(quant_dict, 's', A.attn_scores_act_quantizer)

            hijack_act_quant(quant_dict, f'p{layer_idx}', A.attn_probs_act_quantizer)
            hijack_act_quant(quant_dict, 'p', A.attn_probs_act_quantizer)

            hijack_act_quant(quant_dict, f'c{layer_idx}', A.context_act_quantizer)
            hijack_act_quant(quant_dict, 'c', A.context_act_quantizer)

            # Self-output
            S = L.attention.output
            hijack_act_quant(quant_dict, f'g{layer_idx}', S.dense)
            hijack_act_quant(quant_dict, 'g', S.dense)

            hijack_act_quant(quant_dict, f'u{layer_idx}', S.res_act_quantizer)
            hijack_act_quant(quant_dict, 'u', S.res_act_quantizer)

            hijack_act_quant(quant_dict, f'x{layer_idx}', S.LayerNorm)
            hijack_act_quant(quant_dict, 'x', S.LayerNorm)

            # Output
            O = L.output
            hijack_act_quant(quant_dict, f'h{layer_idx}', O.dense)
            hijack_act_quant(quant_dict, 'h', O.dense)

            hijack_act_quant(quant_dict, f'y{layer_idx}', O.res_act_quantizer)
            hijack_act_quant(quant_dict, 'y', O.res_act_quantizer)

            hijack_act_quant(quant_dict, f'z{layer_idx}', O.LayerNorm)
            hijack_act_quant(quant_dict, 'z', O.LayerNorm)

            # ** All **
            hijack_act_quant_modules(quant_dict, f'L{layer_idx}', L)
            hijack_act_quant_modules(quant_dict, 'L', L)

        # Head
        hijack_act_quant(quant_dict, 'P', model.bert.pooler.dense_act[0])
        hijack_act_quant(quant_dict, 'C', model.classifier)

        hijack_act_quant(quant_dict, 'wP', model.bert.pooler.dense_act[0])
        hijack_weight_quant(quant_dict, 'wC', model.classifier)

    # Prepare quantized model for training/validation
    if 'quant' in config:
        # make another trainer with individually controlled padding strategy & batch size for
        # range estimation
        config_ = deepcopy(config)
        config_.training.batch_size = config.quant.est_ranges_batch_size
        training_args_ = _make_huggingface_training_args(config_)
        trainer_range_est, _, _, _ = _make_datasets_and_trainer(
            config, model, model_enum, tokenizer, task, task_data, compute_metrics, training_args_,
            padding=config.quant.est_ranges_pad,
        )

        # estimate (FP32) ranges for per-group act quant permutation:
        if config.quant.per_groups_permute or config.quant.per_groups_permute_shared_h:
            trainer_per_group, _, _, _ = _make_datasets_and_trainer(
                config, model, model_enum, tokenizer, task, task_data, compute_metrics,
                training_args_,
            )

            model.full_precision()
            pass_data_for_range_estimation(
                loader=trainer_per_group.get_train_dataloader(),
                model=model,
                act_quant=True,  # simply to not exit immediately
                weight_quant=False,
                max_num_batches=10,
                cross_entropy_layer=None,
            )
            model.set_quant_state(
                weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
            )

            # flip the state back to normal
            from quantization.range_estimators import RangeEstimatorBase

            for m in model.modules():
                if isinstance(m, RangeEstimatorBase):
                    m.per_group_range_estimation = False

            # share ranges
            if config.quant.per_groups_permute_shared_h:
                for layer_idx, layer in enumerate(model.bert.encoder.layer):

                    range_estimators = {}
                    for name, m in layer.named_modules():
                        if isinstance(m, RangeEstimatorBase):
                            if m.ranges is not None:
                                range_estimators[name] = m

                    source_ranges = None
                    for k, v in range_estimators.items():
                        print(k)
                        if 'dense' in k:
                            source_ranges = v.ranges.clone()

                    assert source_ranges is not None

                    for k, v in range_estimators.items():
                        v.ranges = source_ranges

        # prepare quantized model (e.g. estimate ranges)
        model = _prepare_quantized_model(
            config, model, loader=trainer_range_est.get_train_dataloader()
        )

        # Apply AdaRound
        if (
            not config.training.do_train
            and config.quant.weight_quant
            and config.adaround.layers is not None
        ):

            trainer_weight_opt, _, _, _ = _make_datasets_and_trainer(
                config, model, model_enum, tokenizer, task, task_data, compute_metrics,
                training_args_, padding=True,
            )

            apply_adaround_to_model(
                config, model, data_loader=trainer_weight_opt.get_train_dataloader(),
                range_est_data_loader=trainer_range_est.get_train_dataloader(),
                batch_size=config.training.batch_size,
                get_samples_fn=adaround_get_samples_fn,
            )

            if config.progress.save_model:
                trainer_range_est.model = model
                trainer_range_est.save_model()  # saves the tokenizer too
                path = Path(config.base.output_dir)
                torch.save(model.state_dict(), path / 'state_dict_adaround.pth')  # contains alpha

    # make datasets and Trainer
    trainer, datasets, train_dataset, eval_dataset = _make_datasets_and_trainer(
        config, model, model_enum, tokenizer, task, task_data, compute_metrics, training_args
    )

    # log a few random samples from the training set
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.\n')

    ## TensorBoard
    tb_writer = None
    if config.progress.tb:
        # attach callback
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=training_args.logging_dir)
        tb_callback = TensorBoardCallback(tb_writer=tb_writer)
        trainer.add_callback(tb_callback)

        # make tb_writer available for all (desired) modules
        for m in model.modules():
            m.tb_writer = tb_writer

        # logging computational graph
        if config.progress.tb_graph:
            logger.info('Logging computational graph ...')

            # prepare data
            x_dict = next(iter(trainer.get_train_dataloader()))
            x_dict = {k: v.cuda() for k, v in x_dict.items()}
            inp_tuple = (x_dict['input_ids'], x_dict['attention_mask'])
            if 'token_type_ids' in x_dict:
                inp_tuple = inp_tuple + (x_dict['token_type_ids'],)

            # log graph
            tb_writer.add_graph(model, inp_tuple, verbose=False)

    ## attach some helper attributes for TB, saving, logging etc.
    # TB counters
    for m in model.modules():
        m.global_step = 0
        m.tb_token_count = DotDict({'total': 0, 'sample_idx': 0, 'last': 0})

    # task name
    for m in model.modules():
        m.task = task.name

    # layer number
    backbone_attr = model_data.backbone_attr
    if backbone_attr is None:
        raise NotImplementedError(
            f'Model {config.model.model_name} not yet supported for ' f'TensorBoard visualization.'
        )
    layers = getattr(model, backbone_attr).encoder.layer
    num_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        for m in layer.modules():
            m.layer_idx = layer_idx
            m.num_layers = num_layers

    # Training!
    model_name_or_path = model_data.model_name_or_path
    if config.training.do_train:
        logger.info('*** Training ***')
        trainer.train(model_path=model_name_or_path if os.path.isdir(model_name_or_path) else None)
        if config.progress.save_model:
            trainer.save_model()  # saves the tokenizer too

    # fix ranges after training, for final evaluation
    if 'quant' in config:
        model.eval()
        model.fix_ranges()
        trainer.model.eval()
        trainer.model.fix_ranges()

    # Validation!
    final_score = None
    if config.training.do_eval:
        logger.info('*** Evaluation ***')

        # if AdaRound, evaluate with multiple range settings for activations
        if config.get('adaround', {}).get('layers', None) is not None:
            # I. FP activations
            model.full_precision_acts()
            trainer, datasets, train_dataset, eval_dataset = _make_datasets_and_trainer(
                config, model, model_enum, tokenizer, task, task_data, compute_metrics,
                training_args,
            )

            score_fp_acts = _eval_task(config, task, trainer, eval_dataset, datasets)
            logger.info(f'Score (FP32 acts) {task.name} -> {100. * score_fp_acts:.2f}')

            # II. quantized activations
            if config.adaround.act_quant_mode == AdaRoundActQuantMode.no_act_quant:
                final_score = score_fp_acts
            else:

                model.quantized_acts()
                config.quant.act_quant = True

                scores = {}
                for batch_size in (1, 4, 16):
                    # reset act ranges
                    model.reset_act_ranges()

                    # (re-)estimate act ranges
                    model.estimate_act_ranges()
                    config_ = deepcopy(config)
                    config_.training.batch_size = batch_size
                    training_args_ = _make_huggingface_training_args(config_)
                    trainer_range_est, _, _, _ = _make_datasets_and_trainer(
                        config, model, model_enum, tokenizer, task, task_data, compute_metrics,
                        training_args_, padding=config.quant.est_ranges_pad,
                    )

                    pass_data_for_range_estimation(
                        loader=trainer_range_est.get_train_dataloader(),
                        model=model,
                        act_quant=config.quant.act_quant,
                        weight_quant=config.quant.weight_quant,
                        max_num_batches=config.act_quant.num_batches,
                        cross_entropy_layer=config.act_quant.cross_entropy_layer,
                    )
                    model.fix_act_ranges()

                    # eval
                    trainer, datasets, train_dataset, eval_dataset = _make_datasets_and_trainer(
                        config, model, model_enum, tokenizer, task, task_data, compute_metrics,
                        training_args,
                    )

                    scores[batch_size] = sc = _eval_task(
                        config, task, trainer, eval_dataset, datasets
                    )
                    logger.info(f'Score (bs={batch_size}) {task.name} -> {100. * sc:.2f}')

                logger.info(f'Score (FP32 acts) {task.name} -> {100. * score_fp_acts:.2f}')
                for k, v in scores.items():
                    logger.info(f'Score (bs={k}) {task.name} -> {100. * v:.2f}')

                final_score = np.max(list(scores.values()))
        else:
            final_score = _eval_task(config, task, trainer, eval_dataset, datasets)

        logger.info(f'Final score {task.name} -> {100. * final_score:.2f}')

        # save final score to file
        if config.training.do_train:
            with open(os.path.join(config.base.output_dir, 'final_score.txt'), 'w') as f:
                f.write(f'{final_score}\n')

    # close tb writer
    if tb_writer is not None:
        tb_writer.close()

    return final_score


def _eval_task(config, task, trainer, eval_dataset, datasets):
    # loop to handle MNLI double evaluation (matched and mis-matched accuracy)
    subtask_names = [task.name]
    eval_datasets = [eval_dataset]
    if task == GLUE_Task.mnli:
        subtask_names.append('mnli-mm')
        eval_datasets.append(datasets['validation_mismatched'])

    subtask_final_scores = []
    for subtask, eval_dataset in zip(subtask_names, eval_datasets):
        if config.data.num_val_samples is not None:
            n = min(len(eval_dataset), config.data.num_val_samples)
            eval_dataset = eval_dataset.select(range(n))

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        # log eval results
        logger.info(f'***** Eval results {subtask} *****')
        for key, value in eval_result.items():
            logger.info(f'\t{key} = {value:.4f}')

        final_score = eval_result[f'eval_{TASK_TO_FINAL_METRIC[task]}']
        subtask_final_scores.append(final_score)

        if config.training.do_train:
            # save eval results to files
            subtask_eval_fpath = os.path.join(config.base.output_dir, f'eval_results_{subtask}.txt')
            with open(subtask_eval_fpath, 'w') as f:
                for key, value in eval_result.items():
                    f.write(f'{key} = {value}\n')

        if config.data.num_val_samples is not None:
            break

    # compute and log final score
    final_score = np.mean(subtask_final_scores)
    return final_score


def _run(config):
    """Common routine to run training/validation on a set of tasks."""
    do_train = config.training.do_train
    mode_str = 'Training' if do_train else 'Validating'
    logger.info(f'{mode_str} with options:\n' + pformat(config))

    # parse tasks
    task_flag = GLUE_Task.from_str(*config.glue.task)
    logger.info(f'{mode_str} on tasks: {list(task_flag.iter_names())}')

    # main task loop
    s = Stopwatch().start()
    task_scores_map = {}
    for task in task_flag.iter():
        logger.info(f'{mode_str} on task {task.name} ...')

        # prepare task-specific config, if necessary
        if config.model.model_path is None:  # use pre-trained backbone for training/validation
            task_config = config
        else:
            # load the suitable checkpoint
            if do_train:
                # simply load the checkpoint given by --model-path
                task_config = config
            else:
                # for validation, load the checkpoint from the corresponding subfolder given by task
                task_dirpath = Path(config.model.model_path) / task.name
                task_out_dirpaths = task_dirpath.glob('**/out')
                non_empty_task_out_dirpaths = list(filter(_is_non_empty_dir, task_out_dirpaths))
                if not len(non_empty_task_out_dirpaths):
                    raise RuntimeError(f'Task directory ({task_dirpath}) is empty.')
                if len(non_empty_task_out_dirpaths) > 1:
                    msg = [f'Task directory ({task_dirpath}) contains multiple checkpoints:']
                    for dirpath in non_empty_task_out_dirpaths:
                        msg.append(f'* {dirpath}')
                    raise RuntimeError('\n'.join(msg))

                task_out_dirpath = str(non_empty_task_out_dirpaths[0])
                task_config = deepcopy(config)
                if config.base.output_dir is None:
                    task_config.base.output_dir = task_out_dirpath
                task_config.model.model_path = task_out_dirpath

        # load data
        task_data = load_task_data(task=task, data_dir=task_config.glue.data_dir)

        # load model and tokenizer
        model_data = load_model_and_tokenizer(**task_config.model, num_labels=task_data.num_labels)

        # run on a task
        task_scores_map[task] = _run_task(task_config, task, task_data, model_data)

    # log task results
    _log_results(task_scores_map)

    # log elapsed time
    logger.info(s.format())


def _train(config):
    # check and set training-specific options
    if config.base.output_dir is None:
        raise ValueError('--output-dir must be provided for training')
    config.training.do_train = True

    _run(config)


def _validate(config):
    # check and set validation-specific options
    config.base.overwrite_output = False
    config.training.do_eval = True
    config.training.do_train = False

    _run(config)


@glue.command()
@pass_config
@glue_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
def train_baseline(config):
    _train(config)


@glue.command()
@pass_config
@glue_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
@quantization_options
@activation_quantization_options
@qat_options
@adaround_options
@transformer_quant_options
def train_quantized(config):
    _train(config)


@glue.command()
@pass_config
@glue_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
def validate_baseline(config):
    _validate(config)


@glue.command()
@pass_config
@glue_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
@quantization_options
@activation_quantization_options
@adaround_options
@transformer_quant_options
def validate_quantized(config):
    _validate(config)


if __name__ == '__main__':
    glue()
