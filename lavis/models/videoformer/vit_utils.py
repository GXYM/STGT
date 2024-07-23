"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on https://github.com/facebookresearch/TimeSformer
"""

# Copyright 2023 somoszhang
# Various utility functions
import sys
import warnings
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from itertools import repeat
import collections.abc as container_abcs
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from einops import rearrange
# from torch_geometric.nn import GraphConv
# from torch_geometric.utils import degree



DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)



def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert self.head_dim * nhead == d_model, "Embedding size must be divisible by number of heads."

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        batch_size, seq_len, _ = q.size()
        q = self.q_linear(x)
        k = self.k_linear(y)
        v = self.v_linear(y)

        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_linear(out)

        return out


class GraphAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, graph_mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Reshape inputs to perform self-attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply graph mask if provided
        if graph_mask is not None:
            # print(graph_mask.shape)
            # 将graph_mask的形状调整为与scores相同的形状
            graph_mask = graph_mask.unsqueeze(1)  # 在第二个维度上添加一个维度
            scores = scores*graph_mask
            scores = scores.masked_fill(graph_mask == 0, float('-inf'))
        
        # Attention probabilities
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Attention output
        attn_output = torch.matmul(attn_probs, V)
        
        # Reshape and transpose attention output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # Linear projection
        attn_output = self.W_O(attn_output)
        
        return attn_output, None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, embed_dim, n_head, layers=1, T=8):
        super().__init__()    

        self.time_embed = nn.Parameter(torch.zeros([T, embed_dim]))
        trunc_normal_(self.time_embed, std=0.02)

        self.resblocks = \
        nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=n_head) for _ in range(layers)])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            time_embed = self.time_embed.unsqueeze(0).transpose(1,2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed

        # x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T) # not do so
        return x

    def forward(self, x): # b, t, -1
        
        ori_x = x
        B, T, _,  C  = x.size()
        x = self.temporal_encoding(x.flatten(0, 1), T, B)
        x = x.permute(1, 0, 2).contiguous()
        x = self.resblocks(x) # t,b, -1
        x = x.permute(1, 0, 2).contiguous().view(B, T, 1, C)
        x = x.type(ori_x.dtype) + ori_x  
        
        return x.mean(dim=1, keepdim=False)
        
        # return x.view(B, T, 1, C)
       

class CrossFramelGraphFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, sim_th: float, graph_mask: torch.Tensor = None, attn_mask: torch.Tensor = None, droppath = 0.,):
        super().__init__()

        self.proj_fc = nn.Linear(d_model, d_model)
        # Residual Attention for golabl feature
        self.message_attn = ResidualAttentionBlock(d_model=d_model, n_head=n_head)
        # graph attention for local feature
        self.graphattn = GraphAttention(d_model, n_head)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.g_mask = graph_mask
        self.sim_th =  sim_th


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def build_graph(self, x: torch.Tensor, t=None):    
      
        # b, t, l, d = x.size()
        # x = x.view(b, t*l, d)
        # calculate similarity matrix and build graph
        x_vectors = F.normalize(x.detach(), dim=-1) 
        # (b, t*l, t*l)
        # print(self.sim_th)
        sim_matrix =  (x_vectors@x_vectors.transpose(-1, -2))
        adj_mat = torch.where(sim_matrix < self.sim_th, 0.0, sim_matrix)
        # spatial temporal graph matrix mask
        if t !=1:
            graph_mask = adj_mat*self.g_mask.to(x.device)
        else:
            graph_mask = adj_mat
        # print(graph_mask.shape)

        return graph_mask.type(x.dtype)

    def forward(self, x):
        b, t, l, d = x.size()
        x = self.proj_fc(x.view(-1, l, d))
        x = x.view(b, t, l, d) # b,t,l,d

        glob_token = x[:, :, 0, :].permute(1,0,2).contiguous()  # t,b,d
        glob_token = self.message_attn(glob_token) # t,b,d
        glob_token = self.message_attn(glob_token) # t,b,d
        glob_token = glob_token.view(1, t, b, d)  # 1,t,b,d

        
        local_token = x[:, :, 1:, :].contiguous() # b,t,l-1,d
        local_token = local_token.view(b, -1, d)  # b,t*(l-1),d
        graph_mask = self.build_graph(local_token, t=t)
        # local_token, attn_probs = self.graphattn(local_token, graph_mask=graph_mask) # b,t*(l-1),d
        local_token, _ = self.graphattn(local_token, graph_mask=graph_mask) # b,t*(l-1),d
        local_token = local_token.view(b, t, l-1, d).permute(2, 1, 0, 3).contiguous()  # l-1,t,b,d
       
        x = torch.cat([glob_token, local_token], dim=0) # l,t,b,d
        x = x.view(l, -1, d) # l,t*b,d
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.view(l, t, b, d).permute(2, 1, 0, 3).contiguous()  # b, t, l, d
        # print(x.shape)
        # print(x[:, :, l-1:, :].contiguous().shape)
        # x[:, :, l-1:, :]  = self.gmhra_mit(x[:, :, l-1:, :].contiguous())

        return x


class CrossGraphTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, T: int, sim_th: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        self.patch_num = 16
        graph_mask = self.gen_graph_mask(T, self.patch_num*self.patch_num)
        self.resblocks = nn.Sequential(*[CrossFramelGraphFormerBlock(width, heads, sim_th, graph_mask, attn_mask, droppath[i]) for i in range(layers)])

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter((width ** -0.5) * torch.randn(width, width))
        
        self.time_embed = nn.Parameter(torch.zeros([T, width]))

        self.pos_2D_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(width, self.patch_num, cls_token=True)).float()
        ).requires_grad_(False)
        
        trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.pos_2D_embed, std=.02)
        self.apply(self._init_weights)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def gen_graph_mask(T, N):
        # print(sim_th)
        g_mask = torch.zeros((T, T, N*N), dtype=torch.int8)  # 创建一个(N*T, N*T)的零张量
        for i in range(T):
            sp = max(0, i-1)
            ep = min(T-1, i+1)
            g_mask[i, sp:ep+1, : ] = torch.ones((ep+1-sp, N*N),dtype=torch.int8)
        
        g_mask = g_mask.view(T, T, N, N)
        g_mask = g_mask.permute(0, 2, 1, 3).contiguous()
        g_mask = g_mask.view(T*N, T*N)
        return g_mask 

    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            # print(self.time_embed.shape)
            time_embed = self.time_embed.unsqueeze(0).transpose(1,2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed

        x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T) # not do so
        return x
       
    def forward(self, x: torch.Tensor):
        
        b, t, l, d = x.shape # (b, t, l, c)
        # print(x.shape)
        x = self.temporal_encoding(x.flatten(0, 1), t, b)
        # print(x.shape)
        x = x + self.pos_2D_embed.unsqueeze(0)
        x = x.view(b, t, l, d)
        if not self.use_checkpoint:
            x =  self.resblocks(x)
        else:
            x =  checkpoint_sequential(self.resblocks, 3, x)
        
        if t !=1:
            # b, t, l, d
            global_tok = x[:, :, 0:1, :]
            local_tok  = x[:, :, 1:, :].view(b, t, self.patch_num, self.patch_num, d)
            local_x, _ = local_tok.max(2)
            local_y, _ = local_tok.max(3)
            x = torch.cat([global_tok, local_x, local_y],dim=2)
        else:
            x = x.mean(dim=1, keepdim=False)
        # x = x.view(b, -1, d)
        # print(x.shape)

        # x = x + self.ln_post(x) @ self.proj

        x = self.ln_post(x.view(b, -1, d))
        x = x @ self.proj

        return x
        

class FrameTimesGraphFormer(nn.Module):
    def __init__(self, T: int, N: int, width: int, layers: int, sim_th=0.88):
        super().__init__()
        
        # print(sim_th)
        self.sim_th = sim_th
        g_mask = torch.zeros(T, T, N*N).to(torch.int8)  # 创建一个(N*T, N*T)的零张量
        for i in range(T):
            sp = max(0, i-1)
            ep = min(T-1, i+1)
            g_mask[i, sp:ep+1, : ] = torch.ones((ep+1-sp, N)).to(torch.int8)
        
        g_mask = g_mask.view(T, T, N, N)
        g_mask = g_mask.permute(0, 2, 1, 3).contiguous()
        self.g_mask = g_mask.view(T*N, T*N)
            
    def forward(self, x: torch.Tensor):    
      
        b, t, l, d = x.size()
        x = x.view(b, t*l, d)
        # calculate similarity matrix and build graph
        x_vectors = F.normalize(x.detach(), dim=-1) 
        # (b, t*l, t*l)
        sim_matrix =  (x_vectors@x_vectors.transpose(-1, -2))
        adj_mat = torch.where(sim_matrix > self.sim_th, 1.0, 0.)
        # spatial temporal graph matrix mask
        graph_mask = self.g_mask*adj_mat


        return x



