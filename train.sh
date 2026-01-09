#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

WANDB_PROJECT="diffu-grpo"

DATASET="math"
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
NUM_ITER=12 # number of policy gradient inner updates iterations
GEN_BATCH=6
# ANSWER_LENGTH=None
ADVANTAGE_TYPE="default"
NO_ANSWER_PADDING=True
ADDITIONAL_ANSWER_LENGTH=4
ANSWER_STEPS=16

# steps 128, block 32, o_block 256
BLOCK_LENGTH=32
ORACLE_BLOCK_LENGTH=256
BETA=0.04
NUM_ORACLE_GENERATIONS=3

for DIFFUSION_STEPS in 64;
do
DIFFUSION_TYPE="oracle" # choose from ['d1', 'oracle']
TIME_STAMP=$(date +%Y%m%d_%H%M%S)

if  [[ $MODEL_PATH == *"LLaDA"* ]]; then
    MODEL_NAME="llada"
fi

RUN_NAME=${MODEL_NAME}/${DIFFUSION_TYPE}/${DATASET}/time:${TIME_STAMP}_iter:${NUM_ITER}_gen:${GEN_BATCH}_diff_steps:${DIFFUSION_STEPS}_block_len:${BLOCK_LENGTH}_ans_len:${ANSWER_LENGTH}_adv_type:${ADVANTAGE_TYPE}_oracle_block_len:${ORACLE_BLOCK_LENGTH}_no_ans_pad:${NO_ANSWER_PADDING}_add_ans_len:${ADDITIONAL_ANSWER_LENGTH}_ans_steps:${ANSWER_STEPS}_beta:${BETA}

python -u train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME/$TIME_STAMP  \
    --generation_batch_size 3 \
    --per_device_train_batch_size $GEN_BATCH \
    --num_generations $GEN_BATCH \
    --block_length $BLOCK_LENGTH \
    --diffusion_steps $DIFFUSION_STEPS \
    --max_steps 4000 \
    --diffusion_type $DIFFUSION_TYPE \
    --log_completions \
    --advantage_type $ADVANTAGE_TYPE \
    --oracle_block_length $ORACLE_BLOCK_LENGTH \
    --no_answer_padding $NO_ANSWER_PADDING \
    --answer_additional_length $ADDITIONAL_ANSWER_LENGTH \
    --answer_steps $ANSWER_STEPS \
    --beta $BETA \
    --num_oracle_generations $NUM_ORACLE_GENERATIONS
done
