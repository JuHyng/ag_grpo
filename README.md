# AG-GRPO: Answer-Guided GRPO for Masked Diffusion Language Models

Official implementation of **AG-GRPO: Answer-Guided GRPO for Masked Diffusion Language Models**.

AG-GRPO asks a simple question: **can ground-truth answers guide rollout generation during training while improving the answer-free policy used at test time?** In standard RLVR, the ground-truth answer is usually used only after generation, as a verifier target. This gives sparse outcome feedback, but it does not help the model produce reasoning that is consistent with the correct answer when early rollouts are weak. Masked diffusion language models make a different strategy possible because they restore masked tokens bidirectionally within a generation span: an answer suffix can guide earlier reasoning tokens during denoising.

## Method

AG-GRPO trains a masked diffusion LM with two rollout conditions for the same prompt-answer pair `(q, y)`.

**Answer-Free (AF) rollouts** match the test-time setting. The model receives the prompt followed by a fully masked generation span and produces a completion without seeing the answer:

```text
q + <mask> ... <mask>  ->  reasoning + answer
```

These samples reflect the policy that will be used at inference time, but early in RL training they often receive little useful reward signal because the final answer is wrong.

**Answer-Guided (AG) rollouts** use the ground-truth answer as a temporary suffix anchor. The answer is wrapped as an answer span and padded with a small margin, then fixed at the end of the generation segment while the model restores the masked reasoning prefix:

```text
q + <mask> ... <mask> + y_tilde  ->  reasoning + y_tilde
```

The fixed suffix is not allowed to directly determine the reward. After answer-guided reasoning is generated, AG-GRPO removes the anchored suffix and asks the model to re-predict the answer from the generated reasoning:

```text
q + reasoning + <mask> ... <mask>  ->  reasoning + predicted_answer
```

The final AG completion is therefore scored using a model-predicted answer, not the copied ground-truth suffix. This keeps the reward verifiable while still letting the answer guide the reasoning trajectory during rollout generation.

For each prompt, AG-GRPO groups AF and AG completions together. With `G_AF` answer-free samples and `G_AG` answer-guided samples, all rewards share the same group baseline:

```text
R_bar = mean({R_AF_1, ..., R_AF_GAF, R_AG_1, ..., R_AG_GAG})
A_AF_i = R_AF_i - R_bar
A_AG_j = R_AG_j - R_bar
```

This shared baseline is the core transfer mechanism. When AG rollouts obtain stronger rewards, they shape the relative advantages in the same group as AF rollouts, so the policy update contrasts answer-free and answer-guided behavior under shared model parameters. The policy is then optimized with the GRPO clipped objective plus a KL penalty to the reference policy, using the masked diffusion per-token likelihood estimator from the diffu-GRPO line of work.

The released code uses the paper terminology directly: `diffusion_type=ag_grpo` runs AG-GRPO, and `num_ag_generations` corresponds to `G_AG`.

## Results and Analysis

AG-GRPO is evaluated on math reasoning, puzzle solving, and code generation:

- GSM8K, MATH-500, and 4x4 Sudoku for exact-match style reasoning evaluation.
- KodCode for RL training in the code domain.
- HumanEval and MBPP for pass@1 code-generation evaluation.

The paper uses the same total rollout budget as diffu-GRPO, with `G = 6`, `G_AF = 3`, and `G_AG = 3`. Across the reported settings, AG-GRPO improves over both the pretrained LLaDA-8B-Instruct model and the diffu-GRPO baseline. The gains are especially visible on sparse or constrained-reward tasks such as Sudoku, where answer-guided rollouts provide a stronger training signal early in optimization.

The analysis in the paper studies two behaviors:

- During answer-guided reasoning, denoising trajectories show that the anchored answer suffix influences restoration of earlier reasoning tokens.
- AG rewards rise quickly early in training, and the shared group-relative baseline helps transfer that signal to the answer-free policy.

Figure assets from the poster will be added under `assets/` once exported. See `docs/figure_assets.md` for the requested crops.

## Setup

```bash
conda env create -f env.yml
conda activate d1
```

or:

```bash
pip install -r requirements.txt
```

Full training expects a CUDA machine suitable for loading `GSAI-ML/LLaDA-8B-Instruct` with 4-bit quantization and LoRA.

## Training

Run the default MATH-500 AG-GRPO configuration:

```bash
bash scripts/train_math.sh
```

Other launchers:

```bash
bash scripts/train_gsm8k.sh
bash scripts/train_sudoku.sh
bash scripts/train_kodcode.sh
```

Common overrides:

```bash
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct \
WANDB_MODE=offline \
bash scripts/train_math.sh
```

The main paper-style settings are provided in `configs/`. The most important arguments are:

| Argument | Meaning |
| --- | --- |
| `diffusion_type=ag_grpo` | use AG-GRPO |
| `num_generations` | total rollout count `G` |
| `num_ag_generations` | answer-guided rollout count `G_AG` |
| `answer_additional_length` | answer suffix margin |
| `answer_steps` | denoising steps for answer re-prediction |
| `ag_block_length` | block length for answer-guided reasoning |

## Data

Hugging Face datasets are loaded automatically for GSM8K, Countdown, and MATH-500. Local files are expected for Sudoku and KodCode:

```text
../dataset/4x4_sudoku_unique_puzzles.csv
kodcode_data/kodcode-9k/train.parquet
```

`reward_func_code.py` executes generated Python code with `firejail`; use code rewards only in an environment where running sandboxed generated code is acceptable.

## Citation

The official ACL 2026 proceedings BibTeX is not available yet and will be added once released.
