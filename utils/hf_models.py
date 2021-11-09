# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from enum import Enum

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from utils.utils import count_embedding_params, count_params, DotDict


logger = logging.getLogger('GLUE')
logger.setLevel(logging.ERROR)


class HF_Models(Enum):
    # vanilla BERT
    bert_base_uncased = 'bert-base-uncased'
    bert_large_uncased = 'bert-large-uncased'
    bert_base_cased = 'bert-base-cased'

    # RoBERTa
    roberta_base = 'roberta-base'

    # Distilled vanilla models
    distilbert_base_uncased = 'distilbert-base-uncased'
    distilroberta_base = 'distilroberta-base'

    # Models optimized for runtime on mobile devices
    mobilebert_uncased = 'google/mobilebert-uncased'
    squeezebert_uncased = 'squeezebert/squeezebert-uncased'

    # ALBERT: very small set of models (optimized for memory)
    albert_base_v2 = 'albert-base-v2'
    albert_large_v2 = 'albert-large-v2'

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


MODEL_TO_BACKBONE_ATTR = {  # model.<backbone attr>.<layers etc.>
    HF_Models.bert_base_uncased: 'bert',
    HF_Models.bert_large_uncased: 'bert',
    HF_Models.bert_base_cased: 'bert',
    HF_Models.distilroberta_base: 'roberta',
    HF_Models.roberta_base: 'roberta',
    HF_Models.mobilebert_uncased: 'mobilebert',
}


def load_model_and_tokenizer(model_name, model_path, use_fast_tokenizer, cache_dir, attn_dropout,
                             hidden_dropout, num_labels, **kw):
    del kw  # unused

    out = DotDict()

    # Config
    if model_path is not None:
        model_name_or_path = model_path
    else:
        model_name_or_path = HF_Models[model_name].value  # use HF identifier

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )

    # set dropout rates
    if attn_dropout is not None:
        logger.info(f'Setting attn dropout to {attn_dropout}')
        if hasattr(config, 'attention_probs_dropout_prob'):
            setattr(config, 'attention_probs_dropout_prob', attn_dropout)

    if hidden_dropout is not None:
        logger.info(f'Setting hidden dropout to {hidden_dropout}')
        if hasattr(config, 'hidden_dropout_prob'):
            setattr(config, 'hidden_dropout_prob', attn_dropout)

    logger.info('HuggingFace model config:')
    logger.info(config)
    out.config = config
    out.model_name_or_path = model_name_or_path

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        cache_dir=cache_dir,
    )
    logger.info('Tokenizer:')
    logger.info(tokenizer)
    out.tokenizer = tokenizer

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=cache_dir,
    )
    logger.info('Model:')
    logger.info(model)
    out.model = model

    # Parameter counts
    total_params = count_params(model)
    embedding_params = count_embedding_params(model)
    non_embedding_params = total_params - embedding_params
    logger.info(f'Parameters (embedding): {embedding_params}')
    logger.info(f'Parameters (non-embedding): {non_embedding_params}')
    logger.info(f'Parameters (total): {total_params}')
    out.total_params = total_params
    out.embedding_params = embedding_params
    out.non_embedding_params = non_embedding_params

    # Additional attributes
    out.model_enum = HF_Models[model_name]
    out.backbone_attr = MODEL_TO_BACKBONE_ATTR.get(out.model_enum, None)
    return out
