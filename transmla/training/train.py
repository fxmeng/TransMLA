import torch
import transformers
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset
import warnings

def preprocess_function(examples, tokenizer, seq_len):
    model_inputs = {"input_ids": [[]]}
    acc_len = 0
    for message in examples['text']:
        message_ids = tokenizer.encode(message, add_special_tokens=False)
        input_ids_list = []
        for i in range(0, len(message_ids), seq_len - 1):
            input_ids_list.append(message_ids[i:i + seq_len - 1] + [tokenizer.eos_token_id])
        for input_ids in input_ids_list:
            if acc_len + len(input_ids) > seq_len:
                model_inputs["input_ids"].append([input_ids])
                acc_len = len(input_ids)
            else:
                model_inputs["input_ids"][-1].append(input_ids)
                acc_len += len(input_ids)
    return model_inputs

@dataclass
class DataCollatorWithFlattening(transformers.DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, *args, return_position_ids=True, separator_id=-100, max_len=8192, pad_token_id=128001, label_ignore_id=-100, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.label_ignore_id = label_ignore_id
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None, separator_id=None):
        def padding_ret(ret):
            padding_len = self.max_len - len(ret["input_ids"])
            if self.return_position_ids:
                padded_position_ids = list(range(padding_len))
                ret["position_ids"] += padded_position_ids
            ret["input_ids"] += [self.pad_token_id] * padding_len
            ret["labels"] += [self.label_ignore_id] * padding_len
            ret["input_ids"] = ret["input_ids"][:self.max_len]
            ret["labels"] = ret["labels"][:self.max_len]
            return ret
        
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id

        rets = []
        for idx in range(0, len(features)):
            ret = {"input_ids": [], "labels": []}
            if self.return_position_ids:
                ret.update({"position_ids": []})
            for f_input_ids in features[idx]["input_ids"]:
                ret["input_ids"] += f_input_ids
                ret["labels"] += [separator_id] + f_input_ids[1:]
                if self.return_position_ids:
                    ret["position_ids"] += list(range(len(f_input_ids)))
            rets.append(padding_ret(ret))

        return transformers.default_data_collator(rets, return_tensors)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    attn_implementation : Optional[str] = field(default="sdpa")
    seq_len: int = field(default=2048,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)

parser = transformers.HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]

model = transformers.AutoModelForCausalLM.from_pretrained(
    training_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=training_args.attn_implementation, 
    trust_remote_code=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.model_name_or_path,)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

train_dataset = load_dataset(training_args.data_path, split="train")
processed_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1024,
    remove_columns=train_dataset.column_names,
    num_proc=128,
    fn_kwargs={"tokenizer": tokenizer, "seq_len": training_args.seq_len}
)

trainer = transformers.Trainer(
    args=training_args,
    model=model,
    train_dataset=processed_dataset,
    data_collator=DataCollatorWithFlattening(max_len=training_args.seq_len, pad_token_id=tokenizer.pad_token_id)
)
trainer.train()
trainer.save_state()
trainer.save_model(output_dir=training_args.output_dir)