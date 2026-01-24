model_path=XiaomiMiMo/MiMo-7B-Base
save_path=outputs/MiMo-7B-Base-deepseek
eval_batch_size=8

python transmla/converter.py \
    --model-path $model_path \
    --save-path $save_path \
    --freqfold 4 \
    --ppl-eval-batch-size $eval_batch_size \
    --deepseek-style