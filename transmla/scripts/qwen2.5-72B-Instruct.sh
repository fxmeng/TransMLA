model_path=Qwen/Qwen2.5-72B-Instruct
save_path=outputs/qwen2_5-72B-Instruct-deepseek
eval_batch_size=4

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size