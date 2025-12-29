import math

import numpy as np
import torch
import torch.nn as nn

import ptwt

from einops import rearrange

from layers.wavelet_layers import HierarchicalWaveletAttentionLayer, HierarchicalWaveletEncoder, NewWaveletAttentionLayer, NewWaveletEncoder

class NARX_Wavelet_Transformer(nn.Module):
    def __init__(self, args):
        super(NARX_Wavelet_Transformer, self).__init__()
        self.args = args
        self.context_window = args.context_window

        self.num_patches = int(self.args.num_scales / self.args.patch_len)
        self.norm = nn.LayerNorm((self.args.patch_len, self.args.num_features))

        self.feature_projection_tgt = nn.Linear(1, self.args.embedding_dim)
        self.feature_projection_src = nn.Linear(args.num_features * args.num_scales, self.args.embedding_dim)
        
        self.position_encoding()

        self.transformer = nn.Transformer(
            d_model = args.embedding_dim,
            nhead=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.mlp_dim,
            dropout=args.dropout,
            batch_first=True
        )
        
        self.wavelet = "cmor1.5-1.0"
        if self.args.sample_type == "geometric":
            self.widths = np.geomspace(self.args.min_dilation, self.args.max_dilation, num=args.num_scales)
        else:
            self.widths = np.linspace(self.args.min_dilation, self.args.max_dilation, num=args.num_scales)

        self._init_projection_weights()
        self.apply(self.initialize_tsformer_weights)
    
    def position_encoding(self,):        
        pe = torch.zeros(self.args.max_len * self.args.num_features_per_metric, self.args.embedding_dim)
        position = torch.arange(0, self.args.max_len * self.args.num_features_per_metric, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.embedding_dim, 2).float() * (-math.log(10000.0) / self.args.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    
    def _init_projection_weights(self):
        """Special initialization for projection layers"""
        nn.init.xavier_uniform_(self.feature_projection_src.weight)
        nn.init.zeros_(self.feature_projection_src.bias)
        nn.init.xavier_uniform_(self.feature_projection_tgt.weight) 
        nn.init.zeros_(self.feature_projection_tgt.bias)

    
    @staticmethod
    def initialize_tsformer_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
               src_key_padding_mask, memory_key_padding_mask, tgt_key_padding_mask):
        tgt = tgt.unsqueeze(-1)
        
        # print(f"gaze_x: {gaze_x.shape}")
        gaze_x = rearrange(gaze_x, "b c n -> b (c n)")
        gaze_y = rearrange(gaze_y, "b c n -> b (c n)")
        gaze_vel = rearrange(gaze_vel, "b c n -> b (c n)")
        gaze_acc = rearrange(gaze_acc, "b c n -> b (c n)")
        pupil = rearrange(pupil, "b c n -> b (c n)")
        stimulus = rearrange(stimulus, "b c n -> b (c n)")

        cwt_x, _ = ptwt.cwt(gaze_x, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_y, _ = ptwt.cwt(gaze_y, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_vel, _ = ptwt.cwt(gaze_vel, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_acc, _ = ptwt.cwt(gaze_acc, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_pup, _ = ptwt.cwt(pupil, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_sti, _ = ptwt.cwt(stimulus, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)

        cwt_x = torch.abs(cwt_x).float()
        cwt_y = torch.abs(cwt_y).float()
        cwt_vel = torch.abs(cwt_vel).float()
        cwt_acc = torch.abs(cwt_acc).float()
        cwt_pup = torch.abs(cwt_pup).float()
        cwt_sti = torch.abs(cwt_sti).float()

        cwt_x = rearrange(cwt_x, "n b s -> b s n").unsqueeze(-1)
        cwt_y = rearrange(cwt_y, "n b s -> b s n").unsqueeze(-1)
        cwt_vel = rearrange(cwt_vel, "n b s -> b s n").unsqueeze(-1)
        cwt_acc = rearrange(cwt_acc,"n b s -> b s n").unsqueeze(-1)
        cwt_pup = rearrange(cwt_pup, "n b s -> b s n").unsqueeze(-1)
        cwt_sti = rearrange(cwt_sti, "n b s -> b s n").unsqueeze(-1)

        encoder_input = torch.cat([cwt_x, cwt_y, cwt_vel, cwt_acc, cwt_pup, cwt_sti], dim=-1)

        encoder_input = rearrange(encoder_input, "b s (p l) v -> b s p l v", p=self.num_patches, l = self.args.patch_len)
        encoder_input = self.norm(encoder_input)
        encoder_input = rearrange(encoder_input, "b s p l v -> b s (p l v)")

        encoder_input = self.feature_projection_src(encoder_input)

        tgt_proj = self.feature_projection_tgt(tgt)

        # print(gaze_x_proj.shape, gaze_y_proj.shape, pupil_proj.shape, tgt_proj.shape) # (batch, seq_len, embed_dim)
        # print(combined_input.shape, encoder_input.shape)                              # (batch, seq_len, 6 * embed_dim), (batch, seq_len, embed_dim)

        encoder_input = encoder_input * math.sqrt(self.args.embedding_dim) + self.pe[:, idx*self.args.num_features_per_metric :\
                                                                      (idx+self.context_window)*self.args.num_features_per_metric, :]
        decoder_input = tgt_proj * math.sqrt(self.args.embedding_dim) + self.pe[:, idx : idx+self.context_window, :]
                
        # print(src_input.shape, tgt_input.shape)                                       # (batch, seq_len, embed_dim)
        # print(tgt_input.shape)                                                        # (batch, seq_len, embed_dim)

        output = self.transformer(
            src=encoder_input, tgt=decoder_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # print(output.shape)                           # [batch, seq_len, embedding_dim]
        return output

    
class Wavelet_Transformer(nn.Module):
    def __init__(self, fc, transformer):
        super().__init__()
        self.fc = fc
        self.transformer = transformer

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask,
                memory_key_padding_mask,
                tgt_key_padding_mask):
                
        return self.fc(self.transformer(idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask
                        )).squeeze(-1)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HIERARCHICAL WAVELET TRANSFORMER    

class NARX_Hierarchical_Wavelet_Transformer(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_patches = args.num_scales // args.patch_len
        self.feature_projection_src = nn.Linear(args.patch_len * self.args.num_features, self.args.inner_dim)
        
        self.wavelet = "cmor1.5-1.0"
        if self.args.sample_type == "geometric":
            self.widths = np.geomspace(self.args.min_dilation, self.args.max_dilation, num=args.num_scales)
        else:
            self.widths = np.linspace(self.args.min_dilation, self.args.max_dilation, num=args.num_scales)

        encoder_layer = HierarchicalWaveletAttentionLayer(self.args)
        self.encoder = HierarchicalWaveletEncoder(encoder_layer=encoder_layer, num_layers=self.args.num_encoder_layers)
        
        self.feature_projection_tgt = nn.Linear(1, args.embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embedding_dim, nhead=args.num_heads, dim_feedforward=args.mlp_dim, dropout=args.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=args.num_decoder_layers)

        self.to_embedding = nn.Linear(self.num_patches * args.inner_dim, args.embedding_dim)

        self._init_projection_weights()
        self.apply(self.initialize_tsformer_weights)
        self._create_time_pe()
        self._create_scale_pe()
        self.position_encoding()

    def _init_projection_weights(self):
        """Special initialization for projection layers"""
        nn.init.xavier_uniform_(self.feature_projection_src.weight)
        nn.init.zeros_(self.feature_projection_src.bias)
        nn.init.xavier_uniform_(self.feature_projection_tgt.weight) 
        nn.init.zeros_(self.feature_projection_tgt.bias)
    
    @staticmethod
    def initialize_tsformer_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
    
    def _create_time_pe(self):
        pe = torch.zeros(self.args.max_len * self.args.num_features_per_metric, self.args.inner_dim)
        pos = torch.arange(0, self.args.max_len * self.args.num_features_per_metric, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.args.inner_dim, 2).float() * (-math.log(10000.0)/self.args.inner_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe_time', pe)  # (1, S, E)

    def _create_scale_pe(self):
        pe = torch.zeros(self.num_patches, self.args.inner_dim)
        pos = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.args.inner_dim, 2).float() * (-math.log(10000.0)/self.args.inner_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe_scale', pe)  # (1, J, E)

    def position_encoding(self,):        
        pe = torch.zeros(self.args.max_len, self.args.embedding_dim)
        position = torch.arange(0, self.args.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.embedding_dim, 2).float() * (-math.log(10000.0) / self.args.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
               src_key_padding_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        tgt = tgt.unsqueeze(-1)
        
        # print(f"gaze_x: {gaze_x.shape}")
        gaze_x = rearrange(gaze_x, "b c n -> b (c n)")
        gaze_y = rearrange(gaze_y, "b c n -> b (c n)")
        gaze_vel = rearrange(gaze_vel, "b c n -> b (c n)")
        gaze_acc = rearrange(gaze_acc, "b c n -> b (c n)")
        pupil = rearrange(pupil, "b c n -> b (c n)")
        stimulus = rearrange(stimulus, "b c n -> b (c n)")

        cwt_x, _ = ptwt.cwt(gaze_x, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_y, _ = ptwt.cwt(gaze_y, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_vel, _ = ptwt.cwt(gaze_vel, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_acc, _ = ptwt.cwt(gaze_acc, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_pup, _ = ptwt.cwt(pupil, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_sti, _ = ptwt.cwt(stimulus, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)

        cwt_x = torch.abs(cwt_x).float()
        cwt_y = torch.abs(cwt_y).float()
        cwt_vel = torch.abs(cwt_vel).float()
        cwt_acc = torch.abs(cwt_acc).float()
        cwt_pup = torch.abs(cwt_pup).float()
        cwt_sti = torch.abs(cwt_sti).float()
        # print(f"cwt_x: {cwt_x.shape}")

        cwt_x = rearrange(cwt_x, "n b s -> b s n").unsqueeze(-1)
        cwt_y = rearrange(cwt_y, "n b s -> b s n").unsqueeze(-1)
        cwt_vel = rearrange(cwt_vel, "n b s -> b s n").unsqueeze(-1)
        cwt_acc = rearrange(cwt_acc,"n b s -> b s n").unsqueeze(-1)
        cwt_pup = rearrange(cwt_pup, "n b s -> b s n").unsqueeze(-1)
        cwt_sti = rearrange(cwt_sti, "n b s -> b s n").unsqueeze(-1)
        # print(f"cwt_x: {cwt_x.shape}")

        encoder_input = torch.cat([cwt_x, cwt_y, cwt_vel, cwt_acc, cwt_pup, cwt_sti], dim=-1) # (b s n v)
        encoder_input = rearrange(encoder_input, "b s (p l) v -> b s p (l v)", l = self.args.patch_len)
        encoder_input = self.feature_projection_src(encoder_input)        
        # print(f"encoder_input: {encoder_input.shape}")

        encoder_input = encoder_input * math.sqrt(self.args.inner_dim)\
                        + self.pe_time[:, idx*self.args.num_features_per_metric :\
                            (idx+self.args.context_window)*self.args.num_features_per_metric].unsqueeze(2)\
                        + self.pe_scale.unsqueeze(1)

        memory = self.encoder(encoder_input, src_mask = src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = rearrange(memory, "b s f e -> b s (f e)")
        memory = self.to_embedding(memory)

        # print(f"memory: {memory.shape}")
        
        dec_inp_proj = self.feature_projection_tgt(tgt)
        tgt_input = dec_inp_proj * math.sqrt(self.args.embedding_dim) + self.pe[:, idx: idx + self.args.context_window, :]

        # print(f"tgt_inp: {tgt_input.shape}")
        output = self.decoder(memory=memory,
                              tgt=tgt_input,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask
                            )

        # print(f"Transformer output: {output.shape}")  # (batch, context_window, embedding_dim)
        return output


class Hierarchical_Wavelet_Transformer(nn.Module):
    def __init__(self, fc, transformer):
        super().__init__()
        self.fc = fc
        self.transformer = transformer

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask,
                memory_key_padding_mask,
                tgt_key_padding_mask):
                
        return self.fc(self.transformer(idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask
                        )).squeeze(-1)
    


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# NEW WAVELET TRANSFORMER    

class NARX_New_Wavelet_Transformer(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_scale_patches = args.num_scales // args.scale_patch_len
        self.num_time_patches = int((args.context_window * args.metric_window_duration / args.feature_window_duration) / args.time_patch_len)
        self.feature_projection_src = nn.Linear(args.time_patch_len * args.scale_patch_len, self.args.inner_dim)
        
        self.wavelet = "cmor1.5-1.0"
        if self.args.sample_type == "geometric":
            self.widths = np.geomspace(self.args.min_dilation, self.args.max_dilation, num=args.num_scales)
        else:
            self.widths = np.linspace(self.args.min_dilation, self.args.max_dilation, num=args.num_scales)

        encoder_layer = NewWaveletAttentionLayer(self.args, self.num_scale_patches, self.num_time_patches)
        self.encoder = NewWaveletEncoder(encoder_layer=encoder_layer, num_layers=self.args.num_encoder_layers)
        
        self.feature_projection_tgt = nn.Linear(1, args.embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embedding_dim, nhead=args.num_heads, dim_feedforward=args.mlp_dim, dropout=args.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=args.num_decoder_layers)

        self.to_embedding = nn.Linear(self.num_scale_patches * args.num_features * args.inner_dim, args.embedding_dim)

        self._init_projection_weights()
        self.apply(self.initialize_tsformer_weights)
        self._create_time_pe()
        self._create_scale_pe()
        self.position_encoding()

    def _init_projection_weights(self):
        """Special initialization for projection layers"""
        nn.init.xavier_uniform_(self.feature_projection_src.weight)
        nn.init.zeros_(self.feature_projection_src.bias)
        nn.init.xavier_uniform_(self.feature_projection_tgt.weight) 
        nn.init.zeros_(self.feature_projection_tgt.bias)
    
    @staticmethod
    def initialize_tsformer_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
    
    def _create_time_pe(self):
        pe = torch.zeros(self.args.max_len * self.num_time_patches, self.args.inner_dim)
        pos = torch.arange(0, self.args.max_len * self.num_time_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.args.inner_dim, 2).float() * (-math.log(10000.0)/self.args.inner_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe_time', pe)  # (1, S, E)

    def _create_scale_pe(self):
        pe = torch.zeros(self.num_scale_patches, self.args.inner_dim)
        pos = torch.arange(0, self.num_scale_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.args.inner_dim, 2).float() * (-math.log(10000.0)/self.args.inner_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe_scale', pe)  # (1, J, E)

    def position_encoding(self,):        
        pe = torch.zeros(self.args.max_len, self.args.embedding_dim)
        position = torch.arange(0, self.args.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.embedding_dim, 2).float() * (-math.log(10000.0) / self.args.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
               src_key_padding_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        tgt = tgt.unsqueeze(-1)
        
        # print(f"gaze_x: {gaze_x.shape}")
        gaze_x = rearrange(gaze_x, "b c n -> b (c n)")
        gaze_y = rearrange(gaze_y, "b c n -> b (c n)")
        gaze_vel = rearrange(gaze_vel, "b c n -> b (c n)")
        gaze_acc = rearrange(gaze_acc, "b c n -> b (c n)")
        pupil = rearrange(pupil, "b c n -> b (c n)")
        stimulus = rearrange(stimulus, "b c n -> b (c n)")

        cwt_x, _ = ptwt.cwt(gaze_x, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_y, _ = ptwt.cwt(gaze_y, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_vel, _ = ptwt.cwt(gaze_vel, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_acc, _ = ptwt.cwt(gaze_acc, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_pup, _ = ptwt.cwt(pupil, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)
        cwt_sti, _ = ptwt.cwt(stimulus, self.widths, self.wavelet, sampling_period=self.args.feature_shift_duration)

        cwt_x = torch.abs(cwt_x).float()
        cwt_y = torch.abs(cwt_y).float()
        cwt_vel = torch.abs(cwt_vel).float()
        cwt_acc = torch.abs(cwt_acc).float()
        cwt_pup = torch.abs(cwt_pup).float()
        cwt_sti = torch.abs(cwt_sti).float()
        # print(f"cwt_x: {cwt_x.shape}")

        cwt_x = rearrange(cwt_x, "n b s -> b s n").unsqueeze(-1)
        cwt_y = rearrange(cwt_y, "n b s -> b s n").unsqueeze(-1)
        cwt_vel = rearrange(cwt_vel, "n b s -> b s n").unsqueeze(-1)
        cwt_acc = rearrange(cwt_acc,"n b s -> b s n").unsqueeze(-1)
        cwt_pup = rearrange(cwt_pup, "n b s -> b s n").unsqueeze(-1)
        cwt_sti = rearrange(cwt_sti, "n b s -> b s n").unsqueeze(-1)
        # print(f"cwt_x: {cwt_x.shape}")

        encoder_input = torch.cat([cwt_x, cwt_y, cwt_vel, cwt_acc, cwt_pup, cwt_sti], dim=-1) # (b s n v)
        encoder_input = rearrange(encoder_input, "b (q t) (p s) v -> b q p v (t s) ", t = self.args.time_patch_len, s=self.args.scale_patch_len)
        # print(f"encoder_vector: {encoder_input.shape}")
        encoder_input = self.feature_projection_src(encoder_input)   
        # print(self.num_time_patches, self.num_scale_patches)     
        # print(f"encoder_input: {encoder_input.shape}")
        # print(f"time pe: {self.pe_time[:, idx*self.num_time_patches :(idx+1)*self.num_time_patches].unsqueeze(2).unsqueeze(3).shape}")
        # print(f"scale pe: {self.pe_scale.unsqueeze(1).unsqueeze(3).shape}")

        encoder_input = encoder_input * math.sqrt(self.args.inner_dim)\
                        + self.pe_time[:, idx*self.num_time_patches :\
                            (idx+1)*self.num_time_patches].unsqueeze(2).unsqueeze(3)\
                        + self.pe_scale.unsqueeze(1).unsqueeze(3)

        memory = self.encoder(encoder_input, src_mask = src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = rearrange(memory, "b q p v e -> b q (p v e)")
        memory = self.to_embedding(memory)

        # print(f"memory: {memory.shape}")
        
        dec_inp_proj = self.feature_projection_tgt(tgt)
        tgt_input = dec_inp_proj * math.sqrt(self.args.embedding_dim) + self.pe[:, idx: idx + self.args.context_window, :]

        # print(f"tgt_inp: {tgt_input.shape}")
        output = self.decoder(memory=memory,
                              tgt=tgt_input,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask
                            )

        # print(f"Transformer output: {output.shape}")  # (batch, context_window, embedding_dim)
        return output


class New_Wavelet_Transformer(nn.Module):
    def __init__(self, fc, transformer):
        super().__init__()
        self.fc = fc
        self.transformer = transformer

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask,
                memory_key_padding_mask,
                tgt_key_padding_mask):
                
        return self.fc(self.transformer(idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask
                        )).squeeze(-1)