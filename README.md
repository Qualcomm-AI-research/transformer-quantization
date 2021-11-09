# Transformer Quantization
This repository contains the implementation and experiments for the paper presented in

**Yelysei Bondarenko<sup>1</sup>, Markus Nagel<sup>1</sup>, Tijmen Blankevoort<sup>1</sup>, 
"Understanding and Overcoming the Challenges of Efficient Transformer Quantization", EMNLP 2021.** [[ACL Anthology]](https://aclanthology.org/2021.emnlp-main.627/) [[ArXiv]](https://arxiv.org/abs/2109.12948)

<sup>1</sup> Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.)


## Reference
If you find our work useful, please cite
```
@inproceedings{bondarenko-etal-2021-understanding,
    title = "Understanding and Overcoming the Challenges of Efficient Transformer Quantization",
    author = "Bondarenko, Yelysei  and
      Nagel, Markus  and
      Blankevoort, Tijmen",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.627",
    pages = "7947--7969",
    abstract = "Transformer-based architectures have become the de-facto standard models for a wide range of Natural Language Processing tasks. However, their memory footprint and high latency are prohibitive for efficient deployment and inference on resource-limited devices. In this work, we explore quantization for transformers. We show that transformers have unique quantization challenges {--} namely, high dynamic activation ranges that are difficult to represent with a low bit fixed-point format. We establish that these activations contain structured outliers in the residual connections that encourage specific attention patterns, such as attending to the special separator token. To combat these challenges, we present three solutions based on post-training quantization and quantization-aware training, each with a different set of compromises for accuracy, model size, and ease of use. In particular, we introduce a novel quantization scheme {--} per-embedding-group quantization. We demonstrate the effectiveness of our methods on the GLUE benchmark using BERT, establishing state-of-the-art results for post-training quantization. Finally, we show that transformer weights and embeddings can be quantized to ultra-low bit-widths, leading to significant memory savings with a minimum accuracy loss. Our source code is available at \url{https://github.com/qualcomm-ai-research/transformer-quantization}.",
}
```

## How to install
First, ensure locale variables are set as follows:
```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

Second, make sure to have Python ≥3.6 (tested with Python 3.6.8) and 
ensure the latest version of `pip` (tested with 21.2.4):
```bash
pip install --upgrade --no-deps pip
```

Next, install PyTorch 1.4.0 with the appropriate CUDA version (tested with CUDA 10.0, CuDNN 7.6.3):
```bash
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Finally, install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```

To run the code, the project root directory needs to be added to your pythonpath:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/this/dir"
```

## Running experiments
The main run file to reproduce all experiments is `main.py`. 
It contains 4 commands to train and validate FP32 and quantized model:
```bash
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  train-baseline
  train-quantized
  validate-baseline
  validate-quantized
```
You can see the full list of options for each command using `python main.py [COMMAND] --help`.

### A. FP32 fine-tuning
To start with, you need to get the fune-tuned model(s) for the GLUE task of interest.
Example run command for fine-tuning:
```bash
python main.py train-baseline --cuda --save-model --model-name bert_base_uncased --task rte \
    --learning-rate 3e-05 --batch-size 8 --eval-batch-size 8 --num-epochs 3 --max-seq-length 128 \
    --seed 1000 --output-dir /path/to/output/dir/
```
You can also do it directly using HuggingFace library [[examples]](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification).
In all experiments we used seeds 1000 - 1004 and reported the median score.
The sample output directory looks as follows:
```bash
/path/to/output/dir
├── config.out
├── eval_results_rte.txt
├── final_score.txt
├── out
│   ├── config.json  # Huggingface model config
│   ├── pytorch_model.bin  # PyTorch model checkpoint
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json  # Huggingface tokenizer config
│   ├── training_args.bin
│   └── vocab.txt  # Vocabulary
└── tb_logs  # TensorBoard logs
    ├── 1632747625.1250594
    │   └── events.out.tfevents.*
    └── events.out.tfevents.*
```

For validation (both full-precision and quantized), it is assumed that these output directories with the fine-tuned 
checkpoints are aranged as follows (you can also use a subset of GLUE tasks):
```bash
/path/to/saved_models/
├── rte/rte_model_dir
│   ├── out
│   │   ├── config.json  # Huggingface model config
│   │   ├── pytorch_model.bin  # PyTorch model checkpoint
│   │   ├── tokenizer_config.json  # Huggingface tokenizer config
│   │   ├── vocab.txt  # Vocabulary
│   │   ├── (...)
├── cola/cola_model_dir
│   ├── out
│   │   ├── (...)
├── mnli/mnli_model_dir
│   ├── out
│   │   ├── (...)
├── mrpc/mrpc_model_dir
│   ├── out
│   │   ├── (...)
├── qnli/qnli_model_dir
│   ├── out
│   │   ├── (...)
├── qqp/qqp_model_dir
│   ├── out
│   │   ├── (...)
├── sst2/sst2_model_dir
│   ├── out
│   │   ├── (...)
└── stsb/stsb_model_dir
    ├── out
    │   ├── (...)
```
Note, that you have to create this file structure manually.

The model can then be validated as follows:
```bash
python main.py validate-baseline --eval-batch-size 32 --seed 1000 --model-name bert_base_uncased \
    --model-path /path/to/saved_models/ --task rte
```
You can also validate multiple or all checkpoints by specifying 
`--task <task1> --task <task2> [...]` or `--task all`, respectively.

### B. Post-training quantization (PTQ)

#### 1) Standard (naïve) W8A8 per-tensor PTQ / base run command for all PTQ experiments
```bash
python main.py validate-quantized --act-quant --weight-quant --no-pad-to-max-length \
	--est-ranges-no-pad --eval-batch-size 16 --seed 1000 --model-path /path/to/saved_models/ \
	--task rte --n-bits 8 --n-bits-act 8 --qmethod symmetric_uniform \
	--qmethod-act asymmetric_uniform --weight-quant-method MSE --weight-opt-method golden_section \
	--act-quant-method current_minmax --est-ranges-batch-size 1 --num-est-batches 1 \
	--quant-setup all
```
Note that the range estimation settings are slightly different for each task.

#### 2) Mixed precision W8A{8,16} PTQ
Specify `--quant-dict "{'y': 16, 'h': 16, 'x': 16}"`:
* `'x': 16` will set FFN's input to 16-bit
* `'h': 16` will set FFN's output to 16-bit
* `'y': 16` will set FFN's residual sum to 16-bit

For STS-B regression task, you will need to specify `--quant-dict "{'y': 16, 'h': 16, 'x': 16, 'P': 16, 'C': 16}"` 
and `--quant-setup MSE_logits`, which will also quantize pooler and the final classifier to 16-bit and use MSE estimator for the output.

#### 3) Per-embedding and per-embedding-group (PEG) activation quantization
* `--per-embd` -- Per-embedding quantization for all activations
* `--per-groups [N_GROUPS]` -- PEG quantization for all activations, no permutation
* `--per-groups [N_GROUPS] --per-groups-permute` -- PEG quantization for all activations, apply range-based permutation (separate for each quantizer)
* `--quant-dict "{'y': 'ng6', 'h': 'ng6', 'x': 'ng6'}"` -- PEG quantization using 6 groups for FFN's input, output and residual sum, no permutation
* `--quant-dict "{'y': 'ngp6', 'h': 'ngp6', 'x': 'ngp6'}" --per-groups-permute-shared-h` -- PEG quantization using 6 groups for FFN's input, output and residual sum, apply range-based permutation (shared between tensors in the same layer)

#### 4) W4A32 PTQ with AdaRound
```bash
python main.py validate-quantized --weight-quant --no-act-quant --no-pad-to-max-length \
	--est-ranges-no-pad --eval-batch-size 16 --seed 1000 --model-path /path/to/saved_models/ \
	--task rte --qmethod symmetric_uniform --qmethod-act asymmetric_uniform --n-bits 4 \
	--weight-quant-method MSE --weight-opt-method grid --num-candidates 100 --quant-setup all \
	--adaround all --adaround-num-samples 1024 --adaround-init range_estimator \
	--adaround-mode learned_hard_sigmoid --adaround-asym --adaround-iters 10000 \
	--adaround-act-quant no_act_quant
```

### C. Quantization-aware training (QAT)
Base run command for QAT experiments (using W4A8 for example):
```bash
python main.py train-quantized --cuda --do-eval --logging-first-step --weight-quant --act-quant \
	--pad-to-max-length --learn-ranges --tqdm --batch-size 8 --seed 1000 \
	--model-name bert_base_uncased --learning-rate 5e-05 --num-epochs 6 --warmup-steps 186 \
	--weight-decay 0.0 --attn-dropout 0.0 --hidden-dropout 0.0 --max-seq-length 128 --n-bits 4 \
	--n-bits-act 8 --qmethod symmetric_uniform --qmethod-act asymmetric_uniform \
	--weight-quant-method MSE --weight-opt-method golden_section --act-quant-method current_minmax \
	--est-ranges-batch-size 16 --num-est-batches 1 --quant-setup all \
	--model-path /path/to/saved_models/rte/out --task rte --output-dir /path/to/qat_output/dir
```
Note that the settings are slightly different for each task (see Appendix).

To run mixed-precision QAT with 2-bit embeddings and 4-bit weights, add `--quant-dict "{'Et': 2}"`.
