# import math

# import numpy as np

# import torch
# import torch.nn as nn

# import ptwt

# from einops import rearrange

# from torch.nn import Transformer

# class FCHead(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.fc = nn.Sequential(
#                     nn.LayerNorm(args.embedding_dim),
#                     nn.Linear(args.embedding_dim, args.embedding_dim),
#                     nn.LeakyReLU(0.2),
#                     nn.LayerNorm(args.embedding_dim),
#                     nn.Dropout(0.2),
#                     nn.Linear(args.embedding_dim, args.embedding_dim * 2),
#                     nn.LeakyReLU(0.2),
#                     nn.LayerNorm(args.embedding_dim * 2),
#                     nn.Linear(args.embedding_dim * 2, 1)
#                 )
#         # self.quantiles = quantiles
#         # self.fc = nn.Linear(embedding_dim, len(self.quantiles))
#         self.initialize_fc_weights()

#     def initialize_fc_weights(self):
#         for layer in self.fc:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

#     def forward(self, x):
#         return self.fc(x)

# class Transformer(nn.Module):
#     def __init__(self, args):
#         super(Transformer, self).__init__()
#         self.args = args
#         self.feature_projection_tgt = nn.Linear(1, self.args.embedding_dim)
#         if self.args.type == "wavelet":
#             self.feature_projection_src = nn.Linear(args.num_features * args.num_scales, args.embedding_dim)
#         else:
#             self.feature_projection_src = nn.Linear(args.num_features, args.embedding_dim)            
        
#         self.position_encoding()

#         if self.args.type == "wavelet":
#             self.norm1 = nn.LayerNorm((args.seq_len, args.patch_len))
#             self.norm2 = nn.LayerNorm((args.seq_len, args.patch_len))
            
#         else:
#             self.norm1 = nn.LayerNorm(args.patch_len)
#             self.norm2 = nn.LayerNorm(args.patch_len)

#         self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer,
#                                                  num_layers=args.num_layers, d_model=)
#         self.transformer = nn.Transformer(
#             d_model=args.embedding_dim,
#             nhead=args.num_heads,
#             num_encoder_layers=args.num_layers,
#             num_decoder_layers=args.num_layers,
#             dim_feedforward=args.embedding_dim,
#             dropout=0,
#             batch_first=True
#         )
        
#         self._init_projection_weights()
#         self.apply(self.initialize_tsformer_weights)
#         # self.create_causal_mask(args.seq_len)
    
#     # def create_causal_mask(self, sz):
#     #     self.attn_mask = Transformer.generate_square_subsequent_mask(sz).bool()   
    
#     def fft(self, feature):    
#         X = torch.fft.fft(feature)
#         mag = torch.abs(X) / self.args.seq_len
#         # Frequency axis
#         # freqs = torch.fft.fftfreq(args.seq_len, d=1/args.sampling_rate)
#         return mag
    
    
#     def cwt(self, feature):
#         widths = torch.Tensor(np.linspace(1, 128, num=self.args.num_scales))
#         wavelet = "cmor1.5-1.0"
#         cwt_coeffs, _ = ptwt.cwt(feature, widths, wavelet, sampling_period=1/self.args.sfreq)
#         return rearrange(torch.abs(cwt_coeffs).float(), "n b s -> b s n")


#     def position_encoding(self,):        
#         pe = torch.zeros(self.args.seq_len, self.args.embedding_dim)
#         position = torch.arange(0, self.args.seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.args.embedding_dim, 2).float() * (-math.log(10000.0) / self.args.embedding_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

    
#     def _init_projection_weights(self):
#         """Special initialization for projection layers"""
#         nn.init.xavier_uniform_(self.feature_projection_src.weight)
#         nn.init.zeros_(self.feature_projection_src.bias)
#         nn.init.xavier_uniform_(self.feature_projection_tgt.weight) 
#         nn.init.zeros_(self.feature_projection_tgt.bias)

    
#     @staticmethod
#     def initialize_tsformer_weights(module):
#         if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.weight, 1.0)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, feature1, feature2, target):
#         if self.args.type == "wavelet":
#             feature1 = self.cwt(feature1)
#             feature2 = self.cwt(feature2)
#             feature1 = rearrange(feature1, "b s (p l) -> b p s l", l=self.args.patch_len)
#             feature2 = rearrange(feature2, "b s (p l) -> b p s l", l=self.args.patch_len)
#             feature1 = self.norm1(feature1)
#             feature2 = self.norm2(feature2)
#             feature1 = rearrange(feature1, "b p s l -> b s (p l)")
#             feature2 = rearrange(feature2, "b p s l -> b s (p l)")

#         elif self.args.type == "freq":
#             feature1 = self.fft(feature1)
#             feature2 = self.fft(feature2)
#             feature1 = rearrange(feature1, "b (p l) -> b p l", l=self.args.patch_len)
#             feature2 = rearrange(feature2, "b (p l) -> b p l", l=self.args.patch_len)
#             feature1 = self.norm1(feature1)
#             feature2 = self.norm2(feature2)
#             feature1 = rearrange(feature1, "b p l -> b (p l)")
#             feature2 = rearrange(feature2, "b p l -> b (p l)")        
#             feature1 = feature1.unsqueeze(-1)
#             feature2 = feature2.unsqueeze(-1)
        
#         elif self.args.type == "time":    
#             feature1 = feature1.unsqueeze(-1)
#             feature2 = feature2.unsqueeze(-1)


#         target = target.unsqueeze(-1)

#         encoder_input = torch.cat([feature1, feature2], dim=-1)
#         encoder_input = self.feature_projection_src(encoder_input)

#         tgt_proj = self.feature_projection_tgt(target)


#         encoder_input = encoder_input * math.sqrt(self.args.embedding_dim) + self.pe[:, :self.args.seq_len, :]
#         decoder_input = tgt_proj * math.sqrt(self.args.embedding_dim) + self.pe[:, :self.args.seq_len, :]
                
#         # print(src_input.shape, tgt_input.shape)                                       # (batch, seq_len, embed_dim)
#         # print(tgt_input.shape)                                                        # (batch, seq_len, embed_dim)

#         output = self.transformer(
#             src=encoder_input, tgt=decoder_input,
#             # src_mask=self.attn_mask,
#             # memory_mask=self.attn_mask,
#             # tgt_mask=self.attn_mask
#         )
#         # print(output.shape)                           # [batch, seq_len, embedding_dim]
#         return output

    
# class Vanilla_Model(nn.Module):
#     def __init__(self, fc, transformer):
#         super().__init__()
#         self.fc = fc
#         self.transformer = transformer

#     def forward(self, feature1, feature2, target):
                
#         return self.fc(self.transformer(feature1, feature2, target)).squeeze(-1)


import math
import numpy as np
import torch
import torch.nn as nn
import ptwt
from einops import rearrange

class FCHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.LayerNorm(args.embedding_dim),
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(args.embedding_dim),
            nn.Dropout(0.2),
            nn.Linear(args.embedding_dim, args.embedding_dim * 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(args.embedding_dim * 2),
            nn.Linear(args.embedding_dim * 2, 1)
        )
        self.initialize_fc_weights()

    def initialize_fc_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        if self.args.type == "wavelet":
            self.feature_projection_src = nn.Linear(args.num_features * args.num_scales, args.embedding_dim)
        else:
            self.feature_projection_src = nn.Linear(args.num_features, args.embedding_dim)
        
        self.position_encoding()

        if self.args.type == "wavelet":
            self.norm1 = nn.LayerNorm((args.seq_len, args.patch_len))
            self.norm2 = nn.LayerNorm((args.seq_len, args.patch_len))
        else:
            self.norm1 = nn.LayerNorm(args.patch_len)
            self.norm2 = nn.LayerNorm(args.patch_len)

        # Define TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.embedding_dim,
            nhead=args.num_heads,
            dim_feedforward=args.embedding_dim,
            dropout=0,
            batch_first=True
        )
        # Stack layers into TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_layers
        )
        
        self._init_projection_weights()
        self.apply(self.initialize_tsformer_weights)

    def fft(self, feature):    
        X = torch.fft.fft(feature)
        mag = torch.abs(X) / self.args.seq_len
        return mag
    
    def cwt(self, feature):
        widths = torch.Tensor(np.linspace(1, 128, num=self.args.num_scales))
        wavelet = "cmor1.5-1.0"
        cwt_coeffs, _ = ptwt.cwt(feature, widths, wavelet, sampling_period=1/self.args.sfreq)
        return rearrange(torch.abs(cwt_coeffs).float(), "n b s -> b s n")

    def position_encoding(self):        
        pe = torch.zeros(self.args.seq_len, self.args.embedding_dim)
        position = torch.arange(0, self.args.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.embedding_dim, 2).float() * (-math.log(10000.0) / self.args.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def _init_projection_weights(self):
        """Special initialization for projection layers"""
        nn.init.xavier_uniform_(self.feature_projection_src.weight)
        nn.init.zeros_(self.feature_projection_src.bias)

    @staticmethod
    def initialize_tsformer_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, feature1, feature2):
        if self.args.type == "wavelet":
            feature1 = self.cwt(feature1)
            feature2 = self.cwt(feature2)
            feature1 = rearrange(feature1, "b s (p l) -> b p s l", l=self.args.patch_len)
            feature2 = rearrange(feature2, "b s (p l) -> b p s l", l=self.args.patch_len)
            feature1 = self.norm1(feature1)
            feature2 = self.norm2(feature2)
            feature1 = rearrange(feature1, "b p s l -> b s (p l)")
            feature2 = rearrange(feature2, "b p s l -> b s (p l)")

        elif self.args.type == "freq":
            feature1 = self.fft(feature1)
            feature2 = self.fft(feature2)
            feature1 = rearrange(feature1, "b (p l) -> b p l", l=self.args.patch_len)
            feature2 = rearrange(feature2, "b (p l) -> b p l", l=self.args.patch_len)
            feature1 = self.norm1(feature1)
            feature2 = self.norm2(feature2)
            feature1 = rearrange(feature1, "b p l -> b (p l)")
            feature2 = rearrange(feature2, "b p l -> b (p l)")        
            feature1 = feature1.unsqueeze(-1)
            feature2 = feature2.unsqueeze(-1)
        
        elif self.args.type == "time":    
            feature1 = feature1.unsqueeze(-1)
            feature2 = feature2.unsqueeze(-1)

        encoder_input = torch.cat([feature1, feature2], dim=-1)
        encoder_input = self.feature_projection_src(encoder_input)

        encoder_input = encoder_input * math.sqrt(self.args.embedding_dim) + self.pe[:, :self.args.seq_len, :]

        # Pass through the encoder only
        output = self.transformer_encoder(encoder_input)
        
        return output

class Vanilla_Model(nn.Module):
    def __init__(self, fc, transformer):
        super().__init__()
        self.fc = fc
        self.transformer = transformer

    def forward(self, feature1, feature2):
        transformer_output = self.transformer(feature1, feature2)
        return self.fc(transformer_output).squeeze(-1)