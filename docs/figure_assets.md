# Figure Assets Needed for README

The poster is a single-slide PPTX. For the README, please export cropped images rather than the full poster. The following assets are the most useful.

## 1. Method Overview

Suggested filename: `assets/ag_grpo_method.png`

Crop the central method panel titled **"Method: Answer-Guided GRPO for Masked dLLMs"**. It should include:

- Answer-Guided Reasoning
- Answer Re-prediction
- Group-Relative Advantage & Optimize

This is the most important README figure because it explains AF/AG rollout construction at a glance.

## 2. Main Reasoning Results

Suggested filename: `assets/reasoning_results.png`

Crop the evaluation plot/table under the text:

> AG-GRPO improves average performance across reasoning benchmarks, with especially large gains on Sudoku.

This should summarize GSM8K, MATH-500, and Sudoku results.

## 3. Code Generation Results

Suggested filename: `assets/code_results.png`

Crop the lower-left code-generation result panel under:

> AG-GRPO consistently improves on HumanEval and MBPP.

This should show HumanEval and MBPP pass@1 gains.

## 4. Denoising Trajectory Analysis

Suggested filename: `assets/denoising_trajectory.png`

Crop the analysis panel under:

> During answer-guided reasoning, model uses the suffix as a guide.

This should show how the answer suffix affects token restoration during AG reasoning.

## 5. Reward Dynamics / Signal Transfer

Suggested filename: `assets/reward_dynamics.png`

Crop the analysis panel under:

> Answer-guided rewards rise quickly early in training, and AF rewards also improve under the shared group-relative objective.

This supports the shared-baseline signal-transfer story.

## Optional

Suggested filename: `assets/diffusion_prelim.png`

Crop the top-right **"Preliminaries: Masked Diffusion LLM"** panel only if you want the README to teach block-wise denoising visually. I would treat this as optional because the method figure is more important.
