MODEL= 'your model path'
OUTPUT_DIR='your output results path'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path $MODEL \
    --eval_dataset <your data path, save in the dataset_info.json> \
    --dataset_dir ./data \
    --template qwen2_vl \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 100 \
    --predict_with_generate

