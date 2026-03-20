import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from itertools import repeat
import numpy as np
class Model(nn.Module):
    def __init__(self, args):
        super().__init__() 
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=7, stride=1, padding=3),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(),
                                   nn.AvgPool2d(3))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ReLU(),
                                   nn.AvgPool2d(3),)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 512, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(num_features=512),
                                   nn.ReLU(),
                                   nn.Dropout())
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.spatialdropout = SpatialDropout(0.2)
        # self.encoder = timm.create_model("efficientnet_b4", pretrained=True, features_only=True)
        # first_conv_layer = self.encoder.conv_stem
        # new_conv_layer = torch.nn.Conv2d(2, first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size, 
        #                                 stride=first_conv_layer.stride, padding=first_conv_layer.padding, 
        #                                 bias=first_conv_layer.bias is not None)
        # self.encoder.conv_stem = new_conv_layer
        # self.gru = nn.GRU(input_size=2688, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        self.gru = nn.GRU(input_size=4 * 4 * 512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        # self.mamba = ResidualBlock(MambaConfig(d_model=2 * 128))
        self.transformer = Transformer(
            num_heads=4,
            num_layers=6,
            attn_size=256 // 4,
            dropout_rate=0.,
            widening_factor=4,
        )
        self.fc = nn.Sequential(nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Linear(64, 2))

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        # pdb.set_trace()
        x = x.view(batch_size*seq_len, channels, height, width)
        # x = x.permute(0, 1, 3, 2)
        # x = self.encoder(x)[-1]
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.pool(x)
        
        x = self.spatialdropout(x)
        # x = x.mean(dim=[2, 3])
        x = x.view(batch_size, seq_len, -1)
        
        x, _ = self.gru(x)
        # pdb.set_trace()
        x = self.transformer(x)
        # x = self.mamba(x)
        # x = self.mlp_attention_layer(x)
        x = x.contiguous().view(batch_size, seq_len, -1)
        x = self.fc(x)
        return x

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1]) 
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)
def get_relative_positions(seq_len, reverse=False, device='cuda'):
    x = torch.arange(seq_len, device=device)[None, :]
    y = torch.arange(seq_len, device=device)[:, None]
    return torch.tril(x - y) if not reverse else torch.triu(y - x)


def get_alibi_slope(num_heads, device='cuda'):
    x = (24) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device, dtype=torch.float32).view(-1, 1, 1)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module."""

    def __init__(self, num_heads, key_size, w_init_scale=None, w_init=None, with_bias=True, b_init=None, value_size=None, model_size=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        self.with_bias = with_bias

        self.query_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.key_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=with_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=with_bias)

    def forward(self, query, key, value, mask=None):
        batch_size, sequence_length, _ = query.size()

        query_heads = self._linear_projection(query, self.key_size, self.query_proj)  # [T', H, Q=K]
        key_heads = self._linear_projection(key, self.key_size, self.key_proj)  # [T, H, K]
        value_heads = self._linear_projection(value, self.value_size, self.value_proj)  # [T, H, V]
        attn_scores = torch.einsum("bhsd,bhqd->bhqs", [key_heads, query_heads])  # [batch_size, num_heads, seq_len, seq_len]
        scale = self.key_size ** 0.5
        attn_scores /= scale
        if mask is not None:
            attn_scores += mask
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_output = torch.einsum("bhqs,bhsd->bhqd", [attn_weights, value_heads])  # [batch_size, num_heads, seq_len, value_size]
        attn_output = attn_output.reshape(batch_size, sequence_length, -1)  # [batch_size, seq_len, num_heads * value_size]
        return self.final_proj(attn_output)  # [batch_size, seq_len, model_size]


    def _linear_projection(self, x, head_size, proj_layer):
        y = proj_layer(x)
        batch_size, sequence_length, _= x.shape
        return y.reshape((batch_size, sequence_length, self.num_heads, head_size)).permute(0, 2, 1, 3)


class MultiHeadAttentionRelative(nn.Module):
    def __init__(self, num_heads, key_size, w_init_scale=None, w_init=None, with_bias=True, b_init=None, value_size=None, model_size=None):
        super(MultiHeadAttentionRelative, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        self.with_bias = with_bias

        self.query_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.key_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=with_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=with_bias)

    def forward(self, query, key, value, mask=None):
        batch_size, sequence_length, _ = query.size()

        query_heads = self._linear_projection(query, self.key_size, self.query_proj)  # [T', H, Q=K]
        key_heads = self._linear_projection(key, self.key_size, self.key_proj)  # [T, H, K]
        value_heads = self._linear_projection(value, self.value_size, self.value_proj)  # [T, H, V]

        device = query.device
        bias_forward = get_alibi_slope(self.num_heads // 2, device=device) * get_relative_positions(sequence_length, device=device)
        bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
        bias_backward = get_alibi_slope(self.num_heads // 2, device=device) * get_relative_positions(sequence_length, reverse=True, device=device)
        bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)
        attn_bias = torch.cat([bias_forward, bias_backward], dim=0)

        attn = F.scaled_dot_product_attention(query_heads, key_heads, value_heads, attn_mask=attn_bias, scale=1 / np.sqrt(self.key_size))
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(attn)  # [T', D']

    def _linear_projection(self, x, head_size, proj_layer):
        y = proj_layer(x)
        batch_size, sequence_length, _= x.shape
        return y.reshape((batch_size, sequence_length, self.num_heads, head_size)).permute(0, 2, 1, 3)

class Transformer(nn.Module):
    """A transformer stack."""

    def __init__(self, num_heads, num_layers, attn_size, dropout_rate, widening_factor=4):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadAttentionRelative(num_heads, attn_size, model_size=attn_size * num_heads),
                'dense': nn.Sequential(
                    nn.Linear(attn_size * num_heads, widening_factor * attn_size * num_heads),
                    nn.GELU(),
                    nn.Linear(widening_factor * attn_size * num_heads, attn_size * num_heads)
                ),
                'layer_norm1': nn.LayerNorm(attn_size * num_heads),
                'layer_norm2': nn.LayerNorm(attn_size * num_heads)
            })
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(attn_size * num_heads)

    def forward(self, embeddings, mask=None):
        h = embeddings
        for layer in self.layers:
            h_norm = layer['layer_norm1'](h)
            h_attn = layer['attn'](h_norm, h_norm, h_norm, mask=mask)
            h_attn = F.dropout(h_attn, p=self.dropout_rate, training=self.training)
            h = h + h_attn

            h_norm = layer['layer_norm2'](h)
            h_dense = layer['dense'](h_norm)
            h_dense = F.dropout(h_dense, p=self.dropout_rate, training=self.training)
            h = h + h_dense

        return self.ln_out(h)