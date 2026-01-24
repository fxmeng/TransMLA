import logging
import time
import torch
import datasets
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
import logging

def get_dataset(name: str) -> datasets.DatasetDict:
    """
    Get the dataset from the HuggingFace datasets library.

    Args:
        name: The name of the HuggingFace dataset to load. Must be one of "wikitext2", "ptb", "c4" or "alpaca".

    Returns:
        The dataset.
    """
    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "en",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    # if alpaca, create a test and validation set from the training set
    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    logging.info("Loading dataset done")
    return ds

def prepare_test_dataloader(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizerBase, seqlen: int = 2048, batch_size: int = 1
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    logging.info(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""

            tokenized_ds = tokenizer("\n\n".join(ds['text']), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    loader = DataLoader(test_ds, batch_size=batch_size)
    logging.info(f"Preparing test dataloader done")
    return loader

def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to produce.
        varied_seqlen: If False, concatenate multiple examples from the dataset into one example until max_seqlen is reached.
        seed: The seed for sampling the dataset.

    Returns:
        A DataLoader.
    """
    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader done")
    return loader

def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
        
def map_tensors(obj, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj
    
@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, 
    pad_token_id: int | None, 
    testloader: DataLoader[dict[str, torch.Tensor]], 
    message: str = "Evaluating perplexity"
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info(message)
    for batch in tqdm(testloader, desc=message):
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = map_tensors(batch, model.model.embed_tokens.weight.device)
        logits = model(**batch, use_cache=False).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()

def insert_qkv_hooks(model):
    query_hooks = []
    key_hooks = []
    value_hooks = []
    q_a_proj_hooks = []
    kv_a_proj_with_mqa_hooks = []
    query_outputs = {}
    key_outputs = {}
    value_outputs = {}
    q_a_proj_outputs = {}
    kv_a_proj_with_mqa_outputs = {}

    def query_hook_fn(module, input, output, index):
        if index not in query_outputs:
            query_outputs[index] = []
        query_outputs[index].append(output.to('cpu'))

    def key_hook_fn(module, input, output, index):
        if index not in key_outputs:
            key_outputs[index] = []
        key_outputs[index].append(output.to('cpu'))
        
    def value_hook_fn(module, input, output, index):
        if index not in value_outputs:
            value_outputs[index] = []
        value_outputs[index].append(output.to('cpu'))

    def q_a_proj_hook_fn(module, input, output, index):
        if index not in q_a_proj_outputs:
            q_a_proj_outputs[index] = []
        q_a_proj_outputs[index].append(output.to('cpu'))

    def kv_a_proj_with_mqa_hook_fn(module, input, output, index):
        if index not in kv_a_proj_with_mqa_outputs:
            kv_a_proj_with_mqa_outputs[index] = []
        kv_a_proj_with_mqa_outputs[index].append(output.to('cpu'))

    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "q_proj"):
            query_hook = layer.self_attn.q_proj.register_forward_hook(lambda module, input, output, idx=idx: query_hook_fn(module, input, output, idx))
            query_hooks.append(query_hook)
        if hasattr(layer.self_attn, "k_proj"):
            key_hook = layer.self_attn.k_proj.register_forward_hook(lambda module, input, output, idx=idx: key_hook_fn(module, input, output, idx))
            key_hooks.append(key_hook)
        if hasattr(layer.self_attn, "v_proj"):
            value_hook = layer.self_attn.v_proj.register_forward_hook(lambda module, input, output, idx=idx: value_hook_fn(module, input, output, idx))
            value_hooks.append(value_hook)
        if hasattr(layer.self_attn, "q_a_proj"):
            q_a_proj_hook = layer.self_attn.q_a_proj.register_forward_hook(lambda module, input, output, idx=idx: q_a_proj_hook_fn(module, input, output, idx))
            q_a_proj_hooks.append(q_a_proj_hook)
        if hasattr(layer.self_attn, "kv_a_proj_with_mqa"):
            kv_a_proj_with_mqa_hook = layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(lambda module, input, output, idx=idx: kv_a_proj_with_mqa_hook_fn(module, input, output, idx))
            kv_a_proj_with_mqa_hooks.append(kv_a_proj_with_mqa_hook)
    
    return query_hooks, key_hooks, value_hooks, q_a_proj_hooks, kv_a_proj_with_mqa_hooks, query_outputs, key_outputs, value_outputs, q_a_proj_outputs, kv_a_proj_with_mqa_outputs

@torch.no_grad()
def get_qkv_calibrate_outputs(
    model: torch.nn.Module, 
    trainloader: DataLoader[dict[str, torch.Tensor]], 
    message: str = "Calibrating QKV"
):
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    """

    start_time = time.time()

    model.eval()
    query_hooks, key_hooks, value_hooks, q_a_proj_hooks, kv_a_proj_with_mqa_hooks, query_outputs, key_outputs, value_outputs, q_a_proj_outputs, kv_a_proj_with_mqa_outputs = insert_qkv_hooks(model)
    ignore_masks = []
    logging.info(message)
    for batch in tqdm(trainloader, desc=message):
        batch = map_tensors(batch, model.model.embed_tokens.weight.device)
        ignore_masks.append(batch["attention_mask"].to('cpu'))
        model(**batch, use_cache=False)

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    for hook in query_hooks:
        hook.remove()
    for hook in key_hooks:
        hook.remove()
    for hook in value_hooks:
        hook.remove()
    for hook in q_a_proj_hooks:
        hook.remove()
    for hook in kv_a_proj_with_mqa_hooks:
        hook.remove()

    for value in query_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in key_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in value_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in q_a_proj_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in kv_a_proj_with_mqa_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    qkv_outputs = {
        "query": query_outputs,
        "key": key_outputs,
        "value": value_outputs,
        "q_a_proj": q_a_proj_outputs,
        "kv_a_proj": kv_a_proj_with_mqa_outputs,
    }
    return qkv_outputs

@torch.no_grad()
def pca_calc(X: list[torch.Tensor], device: str) -> torch.Tensor:
    H = None
    for idx, X_batch in enumerate(X):

        X_batch = X_batch.double().to(device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eigen_vec = X_eig[1][:, index]
    return eigen_vec

def statistics_qkv_rmsnorm(self_attn, q_a_outputs, kv_a_outputs):
    if q_a_outputs is not None:
        self_attn.q_a_layernorm.weight.data.to(self_attn.q_a_proj.weight.device).to(self_attn.dtype)
        q_a_proj = torch.cat(q_a_outputs)
        q_a_rmsnorm = torch.rsqrt(q_a_proj.pow(2).mean(-1) + self_attn.q_a_layernorm.eps).mean()
        self_attn.q_a_layernorm.weight.data = torch.full_like(self_attn.q_a_layernorm.weight.data, q_a_rmsnorm)

    self_attn.kv_a_layernorm.weight.data.to(self_attn.kv_a_proj_with_mqa.weight.device).to(self_attn.dtype)
    kv_a_proj = torch.cat(kv_a_outputs)
    kv_a_rmsnorm = torch.rsqrt(kv_a_proj.pow(2).mean(-1) + self_attn.kv_a_layernorm.eps).mean()
    self_attn.kv_a_layernorm.weight.data = torch.full_like(self_attn.kv_a_layernorm.weight.data, kv_a_rmsnorm)
