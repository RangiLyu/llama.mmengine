"""Full definition of a LLaMA Language Model, all of it in this single file.

Modified from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
"""

import math
from functools import partial

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.logging import print_log
from mmengine.model import BaseModel
from torch.nn import functional as F

from mmllama.registry import MODELS


def build_rope_cache(seq_len: int,
                     n_elem: int,
                     dtype: torch.dtype,
                     base: int = 10000) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base**(torch.arange(0, n_elem, 2, dtype=dtype) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Cache them
    cache = torch.polar(torch.ones_like(idx_theta), idx_theta)  # complex64
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because `view_as_complex` does not support 16 bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed



class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, rope_cache: torch.Tensor) -> None:
        super().__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer('rope_cache', rope_cache, persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size(
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1,
                                                           2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1,
                                                           2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1,
                                                           2)  # (B, nh, T, hs)

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(
            B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, n_embd, expand=4) -> None:
        super().__init__()
        hidden_dim = expand * n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        # ensure n_hidden is multiple of N
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head, rope_cache: torch.Tensor) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd, n_head=n_head, rope_cache=rope_cache)
        self.rms_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), attention_mask)
        x = x + self.mlp(self.rms_2(x))
        return x


@MODELS.register_module()
class LLaMA(BaseModel):

    def __init__(self,
                 block_size: int = 4096,
                 vocab_size: int = 32000,
                 n_layer: int = 32,
                 n_head: int = 32,
                 n_embd: int = 4096,
                 pretrained=None,
                 test_cfg=dict(
                    max_new_tokens=50,
                    temperature=1.0,
                    top_k=200,)) -> None:
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.pretrained = pretrained
        self.test_cfg = Config(test_cfg)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        rope_cache = build_rope_cache(
            seq_len=block_size,
            n_elem=n_embd // n_head,
            dtype=self.lm_head.weight.dtype)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                h=nn.ModuleList([
                    Block(n_embd, n_head, rope_cache) for _ in range(n_layer)
                ]),
                ln_f=RMSNorm(n_embd),
            ))

    def init_weights(self):
        if self.pretrained is not None:
            checkpoint = torch.load(self.pretrained)
            self.load_state_dict(checkpoint, strict=False)
            print_log(f'Load pretrained model from {self.pretrained}')


    # def _init_weights(self, module: nn.Module) -> None:
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(
    #             module.weight,
    #             mean=0.0,
    #             std=0.02 / math.sqrt(2 * self.n_layer))
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(
    #             module.weight,
    #             mean=0.0,
    #             std=0.02 / math.sqrt(2 * self.n_layer))

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask=None,
                labels=None,
                mode='tensor') -> torch.Tensor:


        if mode == 'tensor':
            return self._forward(input_ids, attention_mask)
        elif mode == 'loss':
            return self.loss(input_ids, attention_mask, labels)
        elif mode == 'predict':
            return self.predict(input_ids)

    def _forward(self, input_ids, attention_mask=None):
        _, t = input_ids.size()
        assert (
            t <= self.block_size
        ), f'Cannot forward sequence of length {t}, block size is only {self.block_size}'

        # forward the LLaMA model itself
        x = self.transformer.wte(
            input_ids)  # token embeddings of shape (b, t, n_embd)

        # TODO: prepare attn mask
        if attention_mask is not None:
            attention_mask = None

        for block in self.transformer.h:
            x = block(x, attention_mask)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    def loss(self, input_ids, attention_mask, labels):
        logits = self._forward(input_ids, attention_mask)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return dict(loss=loss)

    @torch.no_grad()
    def predict(self, input_ids):
        logits = self._forward(input_ids)
        # create an empty tensor of the expected final shape and fill in the current tokens
        B, T = input_ids.shape
        T_new = T + self.test_cfg.max_new_tokens
        empty = torch.empty(B, T_new, dtype=input_ids.dtype, device=input_ids.device)
        empty[:, :T] = input_ids
        input_ids = empty
        max_seq_length = self.block_size

        # generate max_new_tokens tokens
        for t in range(T, T_new):
            # ignore the not-filled-yet tokens
            idx_cond = input_ids[:, :t]
            # if the sequence context is growing too long we must crop it at max_seq_length
            idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

            # forward
            logits = self._forward(idx_cond)
            logits = logits[:, -1] / self.test_cfg.temperature

            # optionally crop the logits to only the top k options
            if self.test_cfg.get('top_k', None) is not None:
                v, _ = torch.topk(logits, min(self.test_cfg.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # concatenate the new column
            input_ids[:, t:] = idx_next

        return input_ids


llama_configs = {
    'toy': dict(n_layer=1, n_head=1,
                n_embd=128, block_size=1024,
                vocab_size=32000, pretrained=None),  # for debug
    '7B': dict(n_layer=32, n_head=32,
               n_embd=4096, block_size=4096,
               vocab_size=32000, pretrained='checkpoints/mm-llama/7B/state_dict.pth'),
    '13B': dict(n_layer=40, n_head=40,
                n_embd=5120, block_size=4096,
                vocab_size=32000, pretrained='checkpoints/mm-llama/13B/state_dict.pth'),
    '30B': dict(n_layer=60, n_head=52,
                n_embd=6656, block_size=4096,
                vocab_size=32000, pretrained='checkpoints/mm-llama/30B/state_dict.pth'),
    '65B': dict(n_layer=80, n_head=64,
                n_embd=8192, block_size=4096,
                vocab_size=32000, pretrained='checkpoints/mm-llama/65B/state_dict.pth'),
}


# TODO: mmengine support for partial
# MODELS.register_module('LLaMA-toy', module=partial(LLaMA, **llama_configs['toy']))  # for debug
# MODELS.register_module('LLaMA-7B', module=partial(LLaMA, **llama_configs['7B']))
# MODELS.register_module('LLaMA-13B', module=partial(LLaMA, **llama_configs['13B']))
# MODELS.register_module('LLaMA-30B', module=partial(LLaMA, **llama_configs['30B']))
# MODELS.register_module('LLaMA-65B', module=partial(LLaMA, **llama_configs['65B']))


@MODELS.register_module()
class LLaMAToy(LLaMA):
    def __init__(self, **kwargs):
        super().__init__(**llama_configs['toy'], **kwargs)

@MODELS.register_module()
class LLaMA7B(LLaMA):
    def __init__(self, **kwargs):
        super().__init__(**llama_configs['7B'], **kwargs)


@MODELS.register_module()
class LLaMA13B(LLaMA):
    def __init__(self, **kwargs):
        super().__init__(**llama_configs['13B'], **kwargs)


@MODELS.register_module()
class LLaMA30B(LLaMA):
    def __init__(self, **kwargs):
        super().__init__(**llama_configs['30B'], **kwargs)


@MODELS.register_module()
class LLaMA65B(LLaMA):
    def __init__(self, **kwargs):
        super().__init__(**llama_configs['65B'], **kwargs)
