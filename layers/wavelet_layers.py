import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class HierarchicalWaveletAttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_patches = args.num_scales // args.patch_len
        self.norm1 = nn.LayerNorm(self.args.inner_dim, eps=1e-5, bias=True)
        self.norm2 = nn.LayerNorm(self.args.inner_dim, eps=1e-5, bias=True)
        self.norm3 = nn.LayerNorm(self.args.inner_dim, eps=1e-5, bias=True)

        # Attention blocks
        self.temporal_attention  = nn.MultiheadAttention(embed_dim=self.args.inner_dim,\
            num_heads=self.args.num_heads, bias=True, batch_first=True, dropout=self.args.dropout)
        self.scale_attention = nn.MultiheadAttention(embed_dim=self.args.inner_dim,\
            num_heads=self.args.num_heads, bias=True, batch_first=True, dropout=self.args.dropout)
                
        self.linear1 = nn.Linear(self.args.inner_dim, self.args.mlp_dim, bias=True)
        self.dropout = nn.Dropout(self.args.dropout)
        self.dropout1 = nn.Dropout(self.args.dropout)
        self.dropout2 = nn.Dropout(self.args.dropout)
        self.dropout3 = nn.Dropout(self.args.dropout)
        self.linear2 = nn.Linear(self.args.mlp_dim, self.args.inner_dim, bias=True)
        self.activation = F.relu
    
    def forward(self, src, attn_mask=None, key_padding_mask=None):
        # print(f"src: {src.shape}")
        # print(f"intra mask: {attn_mask.shape}")
        # print(f"inter mask: {self.inter_causal_mask.shape}")
        B, S, F, E = src.shape
        x = rearrange(src, "b s f e -> (b f) s e")
        # print(f"intra: {x.shape}")
        x = x + self.temporal_block(self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = rearrange(x, "(b f) s e -> (b s) f e", b=B, f = self.num_patches)   # No mask needed since attention is carried across scales at same time step
        # print(f"inter: {x.shape}")
        x = x + self.scale_block(self.norm2(x), attn_mask=None, key_padding_mask=key_padding_mask)
        x = rearrange(x, "(b s) f e -> b s f e", b=B, s=self.args.context_window * self.args.num_features_per_metric)        
        # print(f"ff: {x.shape}")
        x = x + self._ff_block(self.norm3(x))
        return x

    def temporal_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.temporal_attention(x, x, x, attn_mask=attn_mask,\
            key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def scale_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.scale_attention(x, x, x, attn_mask=attn_mask,\
            key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class HierarchicalWaveletEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        return output
    
    
class NewWaveletAttentionLayer(nn.Module):
    def __init__(self, args, num_scale_patches, num_time_patches):
        super().__init__()
        self.args = args
        self.num_scale_patches = num_scale_patches
        self.num_time_patches = num_time_patches
        self.norm1 = nn.LayerNorm(self.args.inner_dim, eps=1e-5, bias=True)
        self.norm2 = nn.LayerNorm(self.args.inner_dim, eps=1e-5, bias=True)
        self.norm3 = nn.LayerNorm(self.args.inner_dim, eps=1e-5, bias=True)

        # Attention blocks
        self.temporal_attention  = nn.MultiheadAttention(embed_dim=self.args.inner_dim,\
            num_heads=self.args.num_heads, bias=True, batch_first=True, dropout=self.args.dropout)
        self.scale_attention = nn.MultiheadAttention(embed_dim=self.args.inner_dim,\
            num_heads=self.args.num_heads, bias=True, batch_first=True, dropout=self.args.dropout)
        self.feature_attention = nn.MultiheadAttention(embed_dim=self.args.inner_dim,\
            num_heads=self.args.num_heads, bias=True, batch_first=True, dropout=self.args.dropout)
                
        self.linear1 = nn.Linear(self.args.inner_dim, self.args.mlp_dim, bias=True)
        self.dropout = nn.Dropout(self.args.dropout)
        self.dropout1 = nn.Dropout(self.args.dropout)
        self.dropout2 = nn.Dropout(self.args.dropout)
        self.dropout3 = nn.Dropout(self.args.dropout)
        self.linear2 = nn.Linear(self.args.mlp_dim, self.args.inner_dim, bias=True)
        self.activation = F.relu
    
    def forward(self, src, attn_mask=None, key_padding_mask=None):
        B, Q, P, V, E = src.shape
        x = rearrange(src, "b q p v e -> (b q v) p e")
        # print(f"scale: {x.shape}")
        x = x + self.scale_block(self.norm2(x), attn_mask=None, key_padding_mask=key_padding_mask)
        x = rearrange(x, "(b q v) p e -> (b p v) q e", b=B, q=self.num_time_patches, v=self.args.num_features)
        # print(f"time: {x.shape}")
        x = x + self.temporal_block(self.norm1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = rearrange(x, "(b p v) q e -> (b q p) v e", b=B,  q=self.num_time_patches, p=self.num_scale_patches)  
        # print(f"feature: {x.shape}")
        x = x + self.feature_block(self.norm1(x), attn_mask=None, key_padding_mask=key_padding_mask)
        x = rearrange(x, "(b q p) v e -> b q p v e",  b=B,  q=self.num_time_patches, p=self.num_scale_patches)
        # print(f"ff: {x.shape}")
        x = x + self._ff_block(self.norm3(x))
        return x

    def temporal_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.temporal_attention(x, x, x, attn_mask=attn_mask,\
            key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def scale_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.scale_attention(x, x, x, attn_mask=attn_mask,\
            key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)
    
    def feature_block(self, x, attn_mask=None, key_padding_mask=None):
        x = self.feature_attention(x, x, x, attn_mask=attn_mask,\
            key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class NewWaveletEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        return output