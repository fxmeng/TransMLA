model_path=google/gemma-2-9b
save_path=outputs/gemma2-9B-deepseek
eval_batch_size=4

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 8 \
    --ppl-eval-batch-size $eval_batch_size \
    --cal-dataset alpaca