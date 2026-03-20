#Transformer 
#多帧

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from modules import LocalGrouper
from mamba_layer import MambaBlock
import numpy as np



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, output):
        attn_weights = self.linear(output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs
    
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class EventMamba(nn.Module):
    def __init__(self,num_classes=6,num=1024):
        super().__init__()
        self.n = num
        bimamba_type = "v2"
        # bimamba_type = None
        # self.feature_list = [6,16,32,64]
        # self.feature_list = [6,32,64,128]
        self.feature_list = [6,64,128,256]
        # self.feature_list = [6,128,256,512]
        self.group = LocalGrouper(3, 512, 24, False, "anchor")
        self.group_1 =LocalGrouper(self.feature_list[1], 256, 24, False, "anchor")
        self.group_2 =LocalGrouper(self.feature_list[2], 128, 24, False, "anchor")
        # self.group = LocalGrouper(3, 1024, 24, False, "anchor")
        # self.group_1 =LocalGrouper(self.feature_list[1], 512, 24, False, "anchor")
        # self.group_2 =LocalGrouper(self.feature_list[2], 256, 24, False, "anchor")
        self.embed_dim = Linear1Layer(self.feature_list[0],self.feature_list[1],1)
        self.conv1 = Linear2Layer(self.feature_list[1],1,1)
        self.conv1_1 = Linear2Layer(self.feature_list[1],1,1)
        self.conv2 = Linear2Layer(self.feature_list[2],1,1)
        self.conv2_1 = Linear2Layer(self.feature_list[2],1,1)
        self.conv3 = Linear2Layer(self.feature_list[3],1,1)
        self.conv3_1 = Linear2Layer(self.feature_list[3],1,1)
        self.mamba1 = MambaBlock(dim = self.feature_list[1], layer_idx = 0, bimamba_type = bimamba_type)
        self.mamba2 = MambaBlock(dim = self.feature_list[2], layer_idx = 1,bimamba_type = bimamba_type)
        self.mamba3 = MambaBlock(dim = self.feature_list[3], layer_idx = 2,bimamba_type = bimamba_type)
        self.attention_1 = Attention(self.feature_list[1])
        self.attention_2 = Attention(self.feature_list[2])
        self.attention_3 = Attention(self.feature_list[3])
        self.attention_4 = Attention(self.feature_list[3])
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[3], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2),         # 修改
            # nn.Sigmoid()
        )
        self.transformer = Transformer(
            num_heads=4,
            num_layers=6,
            attn_size=256 // 4,
            dropout_rate=0.,
            widening_factor=4,
        )
    
    def forward(self, x: torch.Tensor):
        #[b,t,f,p]
        batch_size, timebin, feature_number, point_number = x.size()
        x = x.reshape(-1,feature_number,point_number)
        xyz = x.permute(0,2,1)
        xyz, x = self.group(xyz, x.permute(0, 2, 1))
        x= x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)
        x = self.embed_dim(x)
        x = self.conv1(x)
       
        x = x.permute(0,2,1)
        att = self.attention_1(x)
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1)
        x , _= self.mamba1(x)
        x = x.permute(0,2,1)
        x = self.conv1_1(x)
        x = x.permute(0,2,1)
       
        xyz,x = self.group_1(xyz, x)
        x= x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)
        x = self.conv2(x)
        x = x.permute(0,2,1)
        att = self.attention_2(x)
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1)
        x , _= self.mamba2(x)
        x = x.permute(0,2,1)
        x = self.conv2_1(x)
        x = x.permute(0,2,1)

        xyz,x = self.group_2(xyz, x)
        x= x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)
        x = self.conv3(x)
        x = x.permute(0,2,1)
        att = self.attention_3(x)
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1)
        x,_= self.mamba3(x)
        x = x.permute(0,2,1)
        x = self.conv3_1(x)
        x = x.permute(0,2,1)

        attn = self.attention_4(x)
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        # Transformer
        x = x.reshape(batch_size,timebin,-1)      #[4,40,256]
        x = self.transformer(x)
        x = x.reshape(-1,x.shape[-1])
        x = self.classifier(x)
        ## x的shape[batchsize*timebin ,1024]
        return x




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