import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Tuple
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from utils import get_qkv_calibrate_outputs, evaluate_ppl

def rotate_half(x, group):
    rotate_x = []
    dh = x.shape[-1] // group
    for i in range(group):
        rotate_x.append(-x[..., i * dh + dh // 2 : (i + 1) * dh])
        rotate_x.append(x[..., i * dh : i * dh + dh // 2])
    return torch.cat(rotate_x, dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, rope_head=1):
    head_dim = cos.shape[-1]
    rope_dim = head_dim * rope_head
    nope_dim = q.shape[-1] - rope_dim
    q_rope, q_nope = q.split([rope_dim, nope_dim], dim=-1)
    k_rope, k_nope = k.split([rope_dim, nope_dim], dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    ###### this is for rotate-specific deepseek model (rotate not chunk but interval) #########
    b, h, s, d = q_rope.shape
    q_rope = q_rope.view(b, h, s, d // head_dim, head_dim // 2, 2).transpose(4, 5).reshape(b, h, s, d)
    ###### this is for rotate-specific deepseek model (rotate not chunk but interval) #########

    rope_repeat = q_rope.shape[-1] // cos.shape[-1]
    q_rope_embed = q_rope * cos.repeat(1,1,1,rope_repeat) + rotate_half(q_rope, rope_repeat) * sin.repeat(1,1,1,rope_repeat)

    ###### this is for rotate-specific deepseek model (rotate not chunk but interval) #########
    b, h, s, d = k_rope.shape
    k_rope = k_rope.view(b, h, s, d // head_dim, head_dim // 2, 2).transpose(4, 5).reshape(b, h, s, d)
    ###### this is for rotate-specific deepseek model (rotate not chunk but interval) #########

    rope_repeat = k_rope.shape[-1] // cos.shape[-1]
    k_rope_embed = k_rope * cos.repeat(1,1,1,rope_repeat) + rotate_half(k_rope, rope_repeat) * sin.repeat(1,1,1,rope_repeat)

    q_embed = torch.cat([q_rope_embed, q_nope], dim=-1)
    k_embed = torch.cat([k_rope_embed, k_nope], dim=-1)
    return q_embed, k_embed

class PartialRope(nn.Module):
    def __init__(self, self_attn, key_outputs=None, freqfold=1, rope_head=1, collapse=1):
        super().__init__()
        self.config = self_attn.config
        self.layer_idx = self_attn.layer_idx
        self.hidden_size = self_attn.config.hidden_size
        self.num_attention_heads = self_attn.config.num_attention_heads
        self.head_dim = self_attn.head_dim
        self.num_key_value_heads = self_attn.config.num_key_value_heads
        self.latent_dim = self.num_key_value_heads * self.head_dim
        self.attention_dropout = self_attn.attention_dropout
        self.rope_head = rope_head
        self.collapse = collapse
        self.scaling = self.head_dim**(-0.5)
        self.attention_function = ALL_ATTENTION_FUNCTIONS["sdpa"]
        assert freqfold % self.collapse == 0, f"freqfold ({freqfold}) must be divisible by collapse ({self.collapse})"

        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self._insert_kv_up_proj()
        if key_outputs is not None:
            Rk = self.joint_complex_pca(key_outputs, freqfold)
            self.rotate_k_proj(Rk, freqfold=freqfold)
            self.rotate_k_up_proj(Rk, freqfold=freqfold)
            
    def _insert_kv_up_proj(self):
        self.k_up_proj = nn.Linear(self.latent_dim, self.hidden_size, bias=False, dtype=self.k_proj.weight.dtype, device=self.k_proj.weight.device)
        self.v_up_proj = nn.Linear(self.latent_dim, self.hidden_size, bias=False, dtype=self.v_proj.weight.dtype, device=self.v_proj.weight.device)
        kv_groups = self.num_attention_heads // self.num_key_value_heads
        k_up_eye = torch.eye(self.latent_dim, dtype=self.k_proj.weight.dtype, device=self.k_proj.weight.device)
        v_up_eye = torch.eye(self.latent_dim, dtype=self.v_proj.weight.dtype, device=self.v_proj.weight.device)
        k_up_eye = k_up_eye.reshape(self.num_key_value_heads, self.head_dim, self.latent_dim)
        v_up_eye = v_up_eye.reshape(self.num_key_value_heads, self.head_dim, self.latent_dim)
        self.k_up_proj.weight.data = torch.stack([k_up_eye]*kv_groups,dim=1).reshape(-1, self.latent_dim).contiguous()
        self.v_up_proj.weight.data = torch.stack([v_up_eye]*kv_groups,dim=1).reshape(-1, self.latent_dim).contiguous()

    @torch.no_grad()
    def joint_complex_pca(self, Z: list[torch.Tensor], freqfold: int = 1) -> torch.Tensor:
        dtype = self.k_proj.weight.dtype
        eigen_vecs = []
        for i in range(self.head_dim//2//freqfold):
            H = None
            for Z_batch in Z:
                b,n,d = Z_batch.shape
                head_batch = deepcopy(Z_batch).view(b,n, self.num_key_value_heads, 2, self.head_dim//2//freqfold, freqfold//self.collapse, self.collapse)
                head_batch = head_batch.permute(0, 1, 3, 6, 2, 5, 4)
                head_batch = head_batch.reshape(b,n*2, self.num_key_value_heads*freqfold, self.head_dim//2//freqfold)
                head_batch_i = head_batch[:,:,:,i].double().to(self.k_proj.weight.device)
                head_batch_i = torch.sum(head_batch_i.mT @ head_batch_i, dim=0)  # sum over the batch dimension.
                H = head_batch_i if H is None else H + head_batch_i
            damp = 0.01 * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[-1]).to(self.k_proj.weight.device)
            H[diag, diag] = H[diag, diag] + damp
            X_eig = torch.linalg.eigh(H)
            del H
            index = torch.argsort(X_eig[0], descending=True)
            eigen_vecs.append(X_eig[1][:, index])
        return torch.stack(eigen_vecs+eigen_vecs).to(dtype)

    def rotate_k_proj(self, U, freqfold=1):
        k_weight = deepcopy(self.k_proj.weight.data)
        U = U.to(k_weight.dtype).to(k_weight.device)
        if self.k_proj.bias is not None:
            k_bias = deepcopy(self.k_proj.bias.data)
            k_weight = torch.cat([k_weight, k_bias.unsqueeze(1)], dim=1)
        k_weight = k_weight.reshape(self.num_key_value_heads, self.head_dim//freqfold, freqfold//self.collapse, self.collapse, -1)
        k_weight = k_weight.permute(3, 0, 2, 1, 4).reshape(self.num_key_value_heads*freqfold, self.head_dim//freqfold, -1)
        k_weight = torch.einsum("dhc,hdD->cdD", U, k_weight)

        # premute the dimensions to align with deepseek's implementation
        k_weight = k_weight.reshape(self.collapse, self.num_key_value_heads, freqfold//self.collapse, 2, self.head_dim//freqfold // 2, -1)
        k_weight = k_weight.permute(0, 1, 4, 2, 3, 5).reshape(self.num_key_value_heads, self.head_dim, -1)

        if self.k_proj.bias is not None:
            k_bias = k_weight[:, :, -1]
            k_weight = k_weight[:, :, :-1]

            self.k_proj.bias.data = k_bias.flatten().contiguous()
        self.k_proj.weight.data = k_weight.reshape(self.latent_dim, self.hidden_size).contiguous()
        
    def rotate_k_up_proj(self, U, freqfold=1):
        k_up_weight = deepcopy(self.k_up_proj.weight.data)
        U = U.to(k_up_weight.dtype).to(k_up_weight.device)
        k_up_weight = k_up_weight.reshape(-1, self.num_key_value_heads, self.head_dim//freqfold, freqfold//self.collapse, self.collapse)
        k_up_weight = k_up_weight.permute(0, 4, 1, 3, 2).reshape(-1, self.num_key_value_heads*freqfold, self.head_dim//freqfold)
        k_up_weight = torch.einsum("dhc,Dhd->Dcd", U, k_up_weight)

        # same, premute the dimensions to align with deepseek's implementation
        k_up_weight = k_up_weight.reshape(-1, self.collapse, self.num_key_value_heads, freqfold//self.collapse, 2, self.head_dim//freqfold//2)
        k_up_weight = k_up_weight.permute(0, 1, 2, 5, 3, 4).reshape(-1, self.latent_dim)

        self.k_up_proj.weight.data = k_up_weight.contiguous()
  
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        k_up_weight = self.k_up_proj.weight.view(self.num_attention_heads, self.head_dim, self.latent_dim)
        query_states = torch.einsum("bthd,hdc->bhtc", query_states, k_up_weight)
    
        key_states = key_states.view(bsz, 1, q_len, self.latent_dim)
        value_states = value_states.view(bsz, 1, q_len, self.latent_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:,:,::self.collapse], sin[:,:,::self.collapse], self.rope_head)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
       
        v_up_weight = self.v_up_proj.weight.view(self.num_key_value_heads, self.num_attention_heads//self.num_key_value_heads, self.head_dim, self.latent_dim)
        value_states = torch.einsum("bhtc,hgdc->bhgtd", value_states, v_up_weight)
        value_states = value_states.reshape(bsz, self.num_attention_heads, -1, self.head_dim)

        
        attn_output, attn_weights = self.attention_function(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softcap=getattr(self.config, "attn_logit_softcapping", None)
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights



def partial_rope(model, tokenizer, train_loader, test_loader, **kwargs):

    freqfold = kwargs["freqfold"]
    collapse = kwargs["collapse"]

    message = "Calibrating original model's qkv outputs"
    ori_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)

    def partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold: int, collapse):
        for layer_idx, layer in enumerate(model.model.layers):
            setattr(layer, "self_attn", PartialRope(
                layer.self_attn, 
                ori_qkv_outputs["key"][layer_idx], 
                freqfold=freqfold,
                collapse=collapse,
            ))
            
        if test_loader:
            message = f"Evaluating partial-rope model's ppl, freqfold={freqfold}"
            dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
            print(f'Partial RoPE ppl, freqfold={freqfold}: {dataset_ppl:.4f}')
            return model, dataset_ppl
        else:
            return model, None

    if freqfold != "auto":
        freqfold = int(freqfold)
        return partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)[0]
    else:
        assert test_loader is not None, "test_loader is required for auto freqfold detection"
        device = model.device
        model_original = model.to("cpu")

        print(f"Auto freqfold detection...")

        best_freqfold = freqfold = collapse
        best_ppl = float("inf")
        while freqfold <= model_original.config.head_dim // 2:
            model = deepcopy(model_original)
            model = model.to(device)
            model, ppl = partial_rope_freqfold(model, ori_qkv_outputs, test_loader, freqfold, collapse)
            if ppl < best_ppl:
                best_ppl = ppl
                best_freqfold = freqfold
                freqfold *= 2
            else:
                break

        model = deepcopy(model_original)
        model = model.to(device)
        model, _ = partial_rope_freqfold(model, ori_qkv_outputs, None, best_freqfold, collapse)

        print(f"Best freqfold: {best_freqfold}")

        return model, best_freqfold
