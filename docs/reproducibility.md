# Reproducibility Notes

This document records the public training settings used by the AG-GRPO implementation. It is intended to make the repository easier to audit and reproduce without changing the core trainer code.

## Common Training Setup

All public configs under `configs/` use:

- Base model: `GSAI-ML/LLaDA-8B-Instruct`
- LoRA rank: `128`
- LoRA alpha: `64`
- LoRA dropout: `0.05`
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Learning rate: `3e-6`
- Weight decay: `0.1`
- Adam betas: `0.9`, `0.99`
- Max prompt length: `200`
- Training completion length: `256`
- Total rollout budget: `G = 6`
- Answer-guided rollout count: `G_AG = 3`
- Answer-free rollout count: `G_AF = 3`
- Answer re-prediction steps: `S_ans = 16`

## Dataset-Specific Margins

The answer suffix margin is controlled by `answer_additional_length`:

| Dataset | Margin |
| --- | ---: |
| GSM8K | 2 |
| MATH-500 | 4 |
| Sudoku | 4 |
| KodCode | 10 |

## Tokens Per Step

The paper reports results for several tokens-per-step values. In this codebase, with training completion length `L = 256`, set diffusion steps as:

| Tokens per step `N` | `diffusion_steps` |
| ---: | ---: |
| 8 | 32 |
| 4 | 64 |
| 2 | 128 |

The default public configs use `diffusion_steps = 64` for math/puzzle tasks and `128` for KodCode.

## Launch Examples

```bash
bash scripts/train_gsm8k.sh
bash scripts/train_math.sh
bash scripts/train_sudoku.sh
bash scripts/train_kodcode.sh
```

Override a tokens-per-step setting from the command line:

```bash
bash scripts/train_ag_grpo.sh math --diffusion_steps 128
```

The generic launcher forwards only common arguments. For full control, call `train.py` directly:

```bash
python -u train.py \
  --config configs/math_ag_grpo.yaml \
  --diffusion_steps 128 \
  --beta 0.1 \
  --output_dir checkpoints/math_s128
```

## Evaluation Status

The current public release exposes training and reward code. A full standalone evaluator for all paper tables should include:

- answer extraction for GSM8K, MATH-500, and Sudoku,
- pass@1 execution for HumanEval and MBPP,
- fixed generation settings for each reported `L` and `N`,
- checkpoint loading from `--checkpoint_path` or merged LoRA adapters.

Until that evaluator is added, use the reward functions in `reward_func.py` and `reward_func_code.py` as the reference scoring logic.
