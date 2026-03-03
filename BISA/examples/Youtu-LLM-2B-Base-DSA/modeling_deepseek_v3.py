import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union
from einops import rearrange
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    apply_rotary_pos_emb_interleave,
    DeepseekV3RMSNorm,
    DeepseekV3Attention as HFDeepseekV3Attention,
    DeepseekV3DecoderLayer as HFDeepseekV3DecoderLayer,
    DeepseekV3Model as HFDeepseekV3Model,
    DeepseekV3PreTrainedModel as HFDeepseekV3PreTrainedModel,
    DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM
)
from .configuration_deepseek_v3 import DeepseekV3Config
from dsa_kernel import prepare_cu_seqlens_from_position_ids, deepseek_sparse_attention
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

def dense_mha(
    module: nn.Module,
    q_pass: torch.Tensor,
    q_rot: torch.Tensor,
    k_pass: torch.Tensor,
    k_rot: torch.Tensor,
    position_ids: torch.Tensor,
):
    query_states = torch.cat((q_pass, q_rot), dim=-1)
    k_pass = module.kv_b_proj(k_pass)
    k_pass = rearrange(k_pass, 'b s (h d) -> b h s d', h=module.num_heads)
    k_pass, value_states = torch.split(k_pass, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    value_states = F.pad(value_states, [0, module.qk_head_dim - module.v_head_dim])

    attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
        module,
        query_states,
        key_states,
        value_states,
        None,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        position_ids=position_ids
    )

    attn_output = attn_output[:, :, :, : module.v_head_dim]
    return attn_output, attn_weights

class TransDSAIndexer(nn.Module):
    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.scaling = config.index_head_dim ** (-0.5) 
        if self.config.index_absorb:
            self.wq_b = nn.Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False)
            self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
            if config.indexer_norm == 'rmsnorm':
                self.k_norm = DeepseekV3RMSNorm(config.index_head_dim-config.qk_rope_head_dim, config.rms_norm_eps)
            elif config.indexer_norm == 'layernorm':
                self.k_norm = nn.LayerNorm(config.index_head_dim, config.rms_norm_eps)
            else:
                self.k_norm = None # TransMLA have no kv norm
            self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=True)
        elif self.config.index_topk is not None:
            self.Rk = nn.Linear(config.kv_lora_rank, config.index_head_dim-config.qk_rope_head_dim, bias=False)
            self.Rv = nn.Linear(config.kv_lora_rank, config.index_n_heads, bias=False)

    def forward(
        self,
        kv_b_proj: nn.Module,
        q_latent: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],

        q_pass: torch.Tensor,
        q_rot: torch.Tensor,
        k_pass: torch.Tensor,
        k_rot: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        kv_b_weight = rearrange(kv_b_proj.weight, '(h d) r -> h d r', h=self.config.num_attention_heads)
        k_b_weight, v_b_weight = torch.split(kv_b_weight, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=1)
        q_pass = torch.einsum("bhsd,hdr->bhsr", q_pass, k_b_weight)
        q = torch.cat([q_pass, q_rot], dim=-1).transpose(1, 2).contiguous()
        kv = torch.cat([k_pass, k_rot.squeeze(1)], dim=-1)
        if self.config.index_absorb:
            key_nope_shape = (batch_size, seq_length, -1, self.config.index_head_dim-self.config.qk_rope_head_dim)
            q_latent = self.wq_b(q_latent) #(bsz, seq, dim)
            q_latent = rearrange(q_latent, 'b s (h d) -> b h s d', d=self.config.index_head_dim)
            # [q_pass, q_rot] for MLA, [q_rot, q_pass] for Indexer
            index_q_rot, index_q_pass = torch.split(q_latent, [self.config.qk_rope_head_dim, self.config.index_head_dim-self.config.qk_rope_head_dim], dim=-1)

            compressed_kv = self.wk(hidden_states)
            if self.config.indexer_norm == 'layernorm':
                compressed_kv = self.k_norm(compressed_kv) ## original_paper
            # [k_pass, k_rot] for MLA, [k_rot, k_pass] for Indexer
            index_k_rot, index_k_pass = torch.split(compressed_kv, [self.config.qk_rope_head_dim, self.config.index_head_dim-self.config.qk_rope_head_dim], dim=-1)
            if self.config.indexer_norm == 'rmsnorm':
                index_k_pass = self.k_norm(index_k_pass)
            index_k_pass = index_k_pass.unsqueeze(1)
            index_k_rot = index_k_rot.view(batch_size, 1, seq_length, self.config.qk_rope_head_dim)
            cos, sin = position_embeddings
            index_q_rot, index_k_rot = apply_rotary_pos_emb_interleave(index_q_rot, index_k_rot, cos, sin)
            index_q = torch.cat([index_q_rot, index_q_pass], dim=-1).transpose(1,2).contiguous()
            index_k = torch.cat([index_k_rot, index_k_pass], dim=-1).squeeze(1)
            weights = self.weights_proj(hidden_states)
        else:
            index_q = torch.cat([q_rot, self.Rk(q_pass)], dim=-1).transpose(1,2).contiguous()
            index_k = torch.cat([k_rot.squeeze(1), self.Rk(k_pass)], dim=-1)
            weights = self.Rv(k_pass)

        if self.config.index_weights=="value":
            weights = weights.abs()
        position_ids = position_ids.view(-1)
        offsets = prepare_cu_seqlens_from_position_ids(position_ids)
        
        q = rearrange(q, 'b s h d -> (b s) h d', b=batch_size).contiguous()
        kv = rearrange(kv, 'b s d -> (b s) d', b=batch_size).contiguous()
        index_q = rearrange(index_q, 'b s h d -> (b s) h d', b=batch_size).contiguous()
        index_k = rearrange(index_k, 'b s d -> (b s) d', b=batch_size).contiguous()
        weights = rearrange(weights, 'b s h -> (b s) h', b=batch_size).contiguous()
        attn_output, _ = deepseek_sparse_attention(q, kv, index_q, index_k, weights, offsets, self.config.index_topk, self.config.kv_lora_rank, self.scaling)
        attn_output = torch.einsum("shr,hdr->shd", attn_output, v_b_weight)
        attn_output = rearrange(attn_output, '(b s) h d -> b s h d', b=batch_size).contiguous()
        return attn_output, None

class DeepseekV3Attention(HFDeepseekV3Attention):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.index_topk = config.index_topk
        if not config.qk_latent_norm:
            delattr(self, "q_a_layernorm")
            delattr(self, "kv_a_layernorm")
        self.indexer = TransDSAIndexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        assert "position_ids" in kwargs.keys()
        assert self.config.rope_interleave # deepseek style RoPE
        assert self.q_lora_rank is not None
        batch_size, seq_length = hidden_states.shape[:-1]
        q_states = self.q_a_proj(hidden_states)
        if self.config.qk_latent_norm: # TransMLA has no latent norm
            q_states = self.q_a_layernorm(q_states)
        q_latent = q_states
        q_states = self.q_b_proj(q_states)
        q_states = rearrange(q_states, 'b s (h d) -> b h s d', d=self.qk_head_dim)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.config.qk_latent_norm: # TransMLA has no latent norm
            k_pass = self.kv_a_layernorm(k_pass)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        attn_output, attn_weights = self.indexer(self.kv_b_proj, q_latent, hidden_states, position_embeddings, q_pass, q_rot, k_pass, k_rot, kwargs["position_ids"])

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class DeepseekV3DecoderLayer(HFDeepseekV3DecoderLayer):

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekV3Attention(config, layer_idx)

class DeepseekV3PreTrainedModel(HFDeepseekV3PreTrainedModel):
    config_class = DeepseekV3Config
    _no_split_modules = ["DeepseekV3DecoderLayer"]

class DeepseekV3Model(DeepseekV3PreTrainedModel, HFDeepseekV3Model):

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.layers = nn.ModuleList([DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

def fixed_cross_entropy(shift_hidden_states, shift_labels, lm_head_weights, num_items_in_batch=None, ignore_index=-100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    lce = LigerFusedLinearCrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
    loss = lce(lm_head_weights, shift_hidden_states, shift_labels)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss



def ForCausalLMLoss(
    hidden_states, labels, lm_head_weights, hidden_size, vocab_size, num_items_in_batch=None, ignore_index=-100, **kwargs
):
    shift_hidden_states = hidden_states[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # flatten tokens
    shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1)

    loss = fixed_cross_entropy(shift_hidden_states=shift_hidden_states, shift_labels=shift_labels, lm_head_weights=lm_head_weights,
                               num_items_in_batch=num_items_in_batch, ignore_index=ignore_index, **kwargs)
    return loss

class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, HFDeepseekV3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(hidden_states=hidden_states[:, slice_indices, :], labels=labels, lm_head_weights=self.lm_head.weight, hidden_size=self.config.hidden_size, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "DeepseekV3ForCausalLM",
    "DeepseekV3Model",
]

