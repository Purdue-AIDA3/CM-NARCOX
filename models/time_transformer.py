import math

import torch
import torch.nn as nn

from einops import rearrange

# from layers.time_layers import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder

# class Transformer(nn.Module):
#     def __init__(self, d_model, feature_dim, num_heads, num_encoder_layers, num_decoder_layers, mlp_dim, max_len, context, dropout=0.1):
#         super(Transformer, self).__init__()
#         print(f"feature_dim: {feature_dim}")
#         self.output_proj_dim = d_model 
#         self.context = context
#         self.feature_projection_gaze_x = nn.Linear(feature_dim, d_model)
#         self.feature_projection_gaze_y = nn.Linear(feature_dim, d_model)
#         self.feature_projection_gaze_vel = nn.Linear(feature_dim, d_model)
#         self.feature_projection_gaze_acc = nn.Linear(feature_dim, d_model)
#         self.feature_projection_pupil = nn.Linear(feature_dim, d_model)
#         self.feature_projection_stimulus = nn.Linear(feature_dim, d_model)
#         self.feature_projection_tgt = nn.Linear(1, self.output_proj_dim)
#         self.combined_projection = nn.Linear(d_model * 6, self.output_proj_dim)
#         self.max_len = max_len
#         self.position_encoding()

#         encoder_layer = TransformerEncoderLayer(
#             d_model, num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
#         )
#         decoder_layer = TransformerDecoderLayer(
#             d_model, num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
#         )
#         self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
#         self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
#         self._init_projection_weights()
#         self.apply(self.initialize_tsformer_weights)
    
#     def position_encoding(self,):        
#         pe = torch.zeros(self.max_len, self.output_proj_dim)
#         position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.output_proj_dim, 2).float() * (-math.log(10000.0) / self.output_proj_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

    
#     def _init_projection_weights(self):
#         """Special initialization for projection layers"""
#         nn.init.xavier_uniform_(self.feature_projection_gaze_x.weight)
#         nn.init.zeros_(self.feature_projection_gaze_x.bias)
#         nn.init.xavier_uniform_(self.feature_projection_gaze_y.weight)
#         nn.init.zeros_(self.feature_projection_gaze_y.bias)
#         nn.init.xavier_uniform_(self.feature_projection_gaze_vel.weight)
#         nn.init.zeros_(self.feature_projection_gaze_vel.bias)
#         nn.init.xavier_uniform_(self.feature_projection_gaze_acc.weight)
#         nn.init.zeros_(self.feature_projection_gaze_acc.bias)
#         nn.init.xavier_uniform_(self.feature_projection_pupil.weight)
#         nn.init.zeros_(self.feature_projection_pupil.bias)
#         nn.init.xavier_uniform_(self.feature_projection_stimulus.weight)
#         nn.init.zeros_(self.feature_projection_stimulus.bias)
        
#         nn.init.xavier_uniform_(self.feature_projection_tgt.weight) 
#         nn.init.zeros_(self.feature_projection_tgt.bias)
#         # Initialize the combined projection
#         nn.init.xavier_uniform_(self.combined_projection.weight)
#         nn.init.zeros_(self.combined_projection.bias)
    
#     @staticmethod
#     def initialize_tsformer_weights(module):
#         if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.weight, 1.0)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, 
#                 src_key_padding_mask, memory_key_padding_mask, tgt_key_padding_mask):
#         if tgt.dim() == 2:
#             tgt = tgt.unsqueeze(-1)
#         gaze_x_proj  = self.feature_projection_gaze_x(gaze_x)
#         gaze_y_proj  = self.feature_projection_gaze_y(gaze_y)
#         gaze_vel_proj  = self.feature_projection_gaze_vel(gaze_vel)
#         gaze_acc_proj  = self.feature_projection_gaze_acc(gaze_acc)
#         pupil_proj  = self.feature_projection_pupil(pupil)
#         stimulus_proj  = self.feature_projection_stimulus(stimulus)
#         tgt_proj = self.feature_projection_tgt(tgt)
#         combined_input = torch.cat([gaze_x_proj, gaze_y_proj, gaze_vel_proj, gaze_acc_proj, pupil_proj, stimulus_proj], dim=-1)
#         encoder_input = self.combined_projection(combined_input)
#         src_input = encoder_input * math.sqrt(self.output_proj_dim) + self.pe[:, idx : idx + self.context, :]
#         tgt_input = tgt_proj * math.sqrt(self.output_proj_dim) + self.pe[:, idx : idx + self.context:]

#         memory = self.encoder(
#             src_input,
#             src_mask=src_mask,
#             src_key_padding_mask=src_key_padding_mask
#         )
#         output = self.decoder(
#             tgt_input, memory,
#             tgt_mask=tgt_mask,
#             memory_mask=None,
#             tgt_key_padding_mask=tgt_key_padding_mask,
#             memory_key_padding_mask=memory_key_padding_mask
#         )

#         return output

class NARX_Time_Transformer(nn.Module):
    def __init__(self, args):
        super(NARX_Time_Transformer, self).__init__()
        self.args = args
        self.context_window = args.context_window
        self.feature_projection_tgt = nn.Linear(1, self.args.embedding_dim)
        self.feature_projection_src = nn.Linear(args.num_features, self.args.embedding_dim)
        
        self.position_encoding()

        self.transformer = nn.Transformer(
            d_model = self.args.embedding_dim,
            nhead=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.mlp_dim,
            dropout=args.dropout,
            batch_first=True
        )
        
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
        
        # print(f"gaze_x: {gaze_x.shape}")
        gaze_x = rearrange(gaze_x, "b c n -> b (c n)")
        gaze_y = rearrange(gaze_y, "b c n -> b (c n)")
        gaze_vel = rearrange(gaze_vel, "b c n -> b (c n)")
        gaze_acc = rearrange(gaze_acc, "b c n -> b (c n)")
        pupil = rearrange(pupil, "b c n -> b (c n)")
        stimulus = rearrange(stimulus, "b c n -> b (c n)")
        
        gaze_x = gaze_x.unsqueeze(-1)
        gaze_y = gaze_y.unsqueeze(-1)
        gaze_vel = gaze_vel.unsqueeze(-1)
        gaze_acc = gaze_acc.unsqueeze(-1)
        pupil = pupil.unsqueeze(-1)
        stimulus = stimulus.unsqueeze(-1)
        tgt = tgt.unsqueeze(-1)

        encoder_input = torch.cat([gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus], dim=-1)
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

    
class Time_Transformer(nn.Module):
    def __init__(self, fc, transformer):
        super().__init__()
        self.fc = fc
        self.transformer = transformer

    # def fc_forward(self, x):
    #     fc_out = self.fc(x)
    #     return fc_out
    
    # def transformer_forward(self, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
    #             src_key_padding_mask,
    #             memory_key_padding_mask,
    #             tgt_key_padding_mask):
        
    #     output = self.transformer(gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
    #                            src_key_padding_mask=src_key_padding_mask,
    #                            memory_key_padding_mask=memory_key_padding_mask,
    #                            tgt_key_padding_mask=tgt_key_padding_mask
    #                     )
        
    #     return output

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask, tgt_mask, memory_mask,
                src_key_padding_mask,
                memory_key_padding_mask,
                tgt_key_padding_mask):
                
        return self.fc(self.transformer(idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask
                        )).squeeze(-1)