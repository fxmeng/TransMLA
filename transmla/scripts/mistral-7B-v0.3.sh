model_path=mistralai/Mistral-7B-v0.3
save_path=outputs/mistral-7B-deepseek
eval_batch_size=8

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size