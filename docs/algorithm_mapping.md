# Algorithm-to-Code Mapping

This document maps the AG-GRPO paper algorithm to the current implementation. The code uses the same public terminology as the paper: AF for answer-free rollouts and AG for answer-guided rollouts.

## Main Entry Point

| Paper component | Code location |
| --- | --- |
| Training entrypoint | `train.py` |
| AG-GRPO trainer selection | `train.py`, `diffusion_type == "ag_grpo"` |
| diffu-GRPO baseline trainer | `diffu_grpo_trainer.py` |
| AG-GRPO trainer | `ag_grpo_trainer.py` |
| Config dataclass | `diffu_grpo_config.py` |

## Algorithm 1 Mapping

| Algorithm step | Implementation |
| --- | --- |
| Sample prompt-answer pair `(q, y)` | Hugging Face dataset row prepared by `data_utils.py` and consumed in `_generate_and_score_completions` |
| AF rollout `o_AF` | `AGGRPOTrainer._generate_and_score_completions`, answer-free generation branch without `answer` |
| Construct answer anchor `y_tilde = y + pad^m` | Dataset-specific `solutions = ...` plus `answer_additional_length` |
| AG reasoning with suffix anchor | `generate(..., answer=batch_answer_ids)` in `ag_grpo_trainer.py` |
| Remove answer suffix | `batch_cot_completion_ids = batch_cot_completion_ids[:, :-batch_answer_ids.size(1)]` |
| Answer re-prediction | second `generate(...)` call with `gen_length=batch_answer_ids.size(1)` |
| Build AG completion | `prompt_completion_ids_all.append(batch_prompt_completion_ids)` after re-prediction |
| Task reward computation | `reward_func.py` and `reward_func_code.py` |
| Shared group baseline | `mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)` |
| Advantage computation | `advantages = rewards - mean_grouped_rewards` when `advantage_type=default` |
| PPO/GRPO clipped objective | `compute_loss` in both trainer files |
| KL penalty to reference policy | `ref_per_token_logps` and `beta * per_token_kl` in `compute_loss` |

## Key Arguments

| Code argument | Paper meaning |
| --- | --- |
| `diffusion_type=ag_grpo` | run AG-GRPO |
| `num_ag_generations` | `G_AG` |
| `num_generations - num_ag_generations` | `G_AF` |
| `ag_block_length` | AG reasoning block length |
| `answer_additional_length` | answer margin length `m` |
| `answer_steps` | answer re-prediction denoising steps `S_ans` |
