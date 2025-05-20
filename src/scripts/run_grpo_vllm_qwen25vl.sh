#!/bin/bash


# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e



export DEBUG_MODE="true"
export LOG_PATH="your log path"
# 
QWEN_PATH="your model path" 
HF_DATASET="your data path" 
OUTPUT_DIR="your save model path"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="your_wandb_name"
DS_CONFIG="./configs/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage.

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ./src/open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 8192 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epoch 1 \
    --run_name ${RUN_NAME} \
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to wandb \
    --temperature 1.0 \
    --num_generations 8 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt" 
 

