# AG-GRPO

We propose AG-GRPO, which mixes answer-free and answer-guided rollouts for masked diffusion LMs and uses shared group-relative advantages to transfer answer-aligned signals to the answer-free policy, improving verifiable reasoning across math, puzzles, and code.

## Environment Setup

```
conda env create -f env.yml
conda activate d1
```

## Train

We provide a default script:

```
bash train.sh
```

### Key arguments

MODEL_PATH: HF model id (e.g., GSAI-ML/LLaDA-8B-Instruct)

DATASET: dataset key used by train.py (e.g., math)

NUM_ITER: policy-gradient inner update iterations

GEN_BATCH: per-device train batch size and number of generations per prompt

DIFFUSION_STEPS: total diffusion denoising steps (e.g., 64)

DIFFUSION_TYPE: oracle or d1

BLOCK_LENGTH: token block length for answer-free completions

ORACLE_BLOCK_LENGTH: token block length for answer-conditioned (oracle) completions

NO_ANSWER_PADDING: if True, do not pad the answer segment

ADDITIONAL_ANSWER_LENGTH: extra answer tokens to allow when conditioning

ANSWER_STEPS: number of denoising steps within the answer span

BETA: KL/regularization strength

NUM_ORACLE_GENERATIONS: number of oracle-conditioned samples per prompt

WANDB_PROJECT: W&B project name (set WANDB_MODE=offline to avoid network use)

LOGDIR: output root (defaults to checkpoints/)
