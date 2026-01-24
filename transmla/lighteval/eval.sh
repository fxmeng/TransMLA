#! /bin/bash
pip install lighteval lighteval[math]
pip install more-itertools
pip install word2number

#lighteval vllm \
#"model_name=$1,revision=main,dtype=bfloat16,tensor_parallel_size=8" \
accelerate launch -m lighteval accelerate \
"model_name=$1,revision=main,dtype=bfloat16,max_length=2048,trust_remote_code=True" \
"custom|hellaswag|0|1,custom|arc|0|1,custom|piqa|0|1,custom|winogrande|0|1,custom|openbook_qa|0|1,custom|mmlu|0|1" \
--custom-tasks "tasks.py"


