BASE_MODEL=outputs/qwen2_5-7B-deepseek
OUTPUT_PATH=outputs/qwen2_5-7B-deepseek-ft6B
DATA_PATH=fxmeng/transmla_pretrain_6B_tokens

export HF_ENDPOINT=https://hf-mirror.com
# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
accelerate launch --config_file zero3.yaml \
    train.py \
    --model_name_or_path $BASE_MODEL \
    --bf16 \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --seq_len 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard"
