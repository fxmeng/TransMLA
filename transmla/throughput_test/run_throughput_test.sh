export HF_ENDPOINT=https://hf-mirror.com

device=2
original_model=Meta-Llama/Llama-2-7b-hf
transmla_model=outputs/llama2-7b-deepseek
N_REQUESTS=100

for length in 1024 2048 4096 8192 16384; do
    INPUT_LEN=$length
    OUTPUT_LEN=$length

    for model in $original_model $transmla_model; do
        echo "Running $model with input length $INPUT_LEN and output length $OUTPUT_LEN"

        CUDA_VISIBLE_DEVICES=$device python benchmark_throughput.py \
            --model $model \
            --dataset-name random \
            --num-prompts $N_REQUESTS \
            --input_len $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --prefix-len 0 \
            --random-range-ratio 0

        sleep 5
    done
done