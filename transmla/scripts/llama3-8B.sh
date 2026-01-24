model_path=meta-llama/Meta-Llama-3-8B
save_path=outputs/llama3-8B-deepseek
eval_batch_size=4

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size