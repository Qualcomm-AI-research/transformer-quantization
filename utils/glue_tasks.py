# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from enum import Flag, auto
from functools import reduce
from operator import or_

import numpy as np
from datasets import load_dataset, load_metric
from transformers import EvalPrediction

from utils.utils import DotDict


# setup logger
logger = logging.getLogger('GLUE')
logger.setLevel(logging.INFO)


class GLUE_Task(Flag):
    cola = auto()
    sst2 = auto()
    mrpc = auto()
    stsb = auto()
    qqp = auto()
    mnli = auto()
    qnli = auto()
    rte = auto()
    wnli = auto()
    all = cola | sst2 | mrpc | stsb | qqp | mnli | qnli | rte | wnli

    def __contains__(self, item):
        return (self.value & item.value) == item.value

    @classmethod
    def from_str(cls, *names):
        """Construct flag from strings."""
        assert len(names)
        return reduce(or_, map(cls.__getattr__, names))

    @classmethod
    def list_names(cls):
        """List all flags, including `all`."""
        return [m.name for m in cls]

    def iter(self):
        """List all member flags which are set (excluding `all`)."""
        for x in self.__class__.__members__.values():
            if x in self and x != self.__class__.all:
                yield x

    def iter_names(self):
        """List all member flag names which are set (excluding `all`)."""
        for x in self.iter():
            yield x.name


TASK_TO_SENTENCE_KEYS = {
    GLUE_Task.cola: ('sentence', None),
    GLUE_Task.sst2: ('sentence', None),
    GLUE_Task.mrpc: ('sentence1', 'sentence2'),
    GLUE_Task.stsb: ('sentence1', 'sentence2'),
    GLUE_Task.qqp: ('question1', 'question2'),
    GLUE_Task.mnli: ('premise', 'hypothesis'),
    GLUE_Task.qnli: ('question', 'sentence'),
    GLUE_Task.rte: ('sentence1', 'sentence2'),
    GLUE_Task.wnli: ('sentence1', 'sentence2'),
}


TASK_TO_FINAL_METRIC = {
    GLUE_Task.cola: 'matthews_correlation',
    GLUE_Task.sst2: 'accuracy',
    GLUE_Task.mrpc: 'combined_score',
    GLUE_Task.stsb: 'combined_score',
    GLUE_Task.qqp: 'combined_score',
    GLUE_Task.mnli: 'accuracy',
    GLUE_Task.qnli: 'accuracy',
    GLUE_Task.rte: 'accuracy',
    GLUE_Task.wnli: 'accuracy',
}


TASK_N = {
    GLUE_Task.mnli: 392702,
    GLUE_Task.qqp: 363846,
    GLUE_Task.qnli: 104743,
    GLUE_Task.sst2: 67349,
    GLUE_Task.cola: 8551,
    GLUE_Task.stsb: 5749,
    GLUE_Task.mrpc: 3665,
    GLUE_Task.rte: 2490,
    GLUE_Task.wnli: 635,
}


def load_task_data(task: GLUE_Task, data_dir: str):
    out = DotDict()

    # download and load data
    logger.info(f'Getting {task.name} dataset ...\n')
    out.datasets = load_dataset('glue', task.name, cache_dir=data_dir)

    # determine number of labels
    logger.info('Determine labels ...\n')
    if task == GLUE_Task.stsb:  # regression
        out.num_labels = 1
        logger.info(f'{task.name}: 1 label -- <Regression>')
    else:
        label_list = out.datasets["train"].features["label"].names
        out.num_labels = n_labels = len(label_list)
        logger.info(f'{task.name}: {n_labels} labels -- {label_list}')

    # store sentence keys
    out.sentence1_key, out.sentence2_key = TASK_TO_SENTENCE_KEYS[task]
    return out


def make_compute_metric_fn(task: GLUE_Task):
    metric = load_metric('glue', task.name)
    logger.info('Metric:')
    logger.info(metric)

    def fn(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task == GLUE_Task.stsb else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result['combined_score'] = np.mean(list(result.values())).item()
        return result

    return fn
