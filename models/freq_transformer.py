import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from layers.freq_layers import Trans_C

class Fredformer_backbone(nn.Module):
    def __init__(self, cf_dim:int,cf_depth :int,cf_heads:int,cf_mlp:int,cf_head_dim:int,cf_drop:float,mlp_drop:float,c_in:int,
                 context_window:int, target_window:int, patch_len:int, stride:int,  d_model:int, individual = False, feature_wise=False, **kwargs):
        
        super().__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len)/stride + 1)
        if feature_wise:                                                     # (modification)
            self.norm1 = nn.LayerNorm((patch_len))
            self.norm2 = nn.LayerNorm((patch_len))
        else:                                                                # fredformer
            self.norm1 = nn.LayerNorm((c_in, patch_len))
            self.norm2 = nn.LayerNorm((c_in, patch_len))
            
        #print("depth=",cf_depth)
        # Backbone 
        self.fre_transformer = Trans_C(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp,
                                       dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 ,
                                       horizon = self.horizon*2, d_model=d_model*2)
        
        
        # Head
        self.head_nf_f  = d_model * 2 * patch_num #self.horizon * patch_num#patch_len * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=mlp_drop)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=mlp_drop)
        
        self.ircom = nn.Linear(self.targetwindow*2,self.targetwindow)

        #break up R&I:
        self.get_r = nn.Linear(d_model*2,d_model*2)
        self.get_i = nn.Linear(d_model*2,d_model*2)
        self.output1 = nn.Linear(target_window,target_window)


        #ablation
        self.input = nn.Linear(c_in,patch_len*2)
        self.outpt = nn.Linear(d_model*2,c_in)
        self.abfinal = nn.Linear(patch_len*patch_num,target_window)

    def forward(self, z):
        z = torch.fft.fft(z)                                                                        # z: [bs x nvars x seq_len]
        z1 = z.real
        z2 = z.imag
        

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]                                                                 

        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)


        # model shape
        batch_size = z1.shape[0]
        patch_num  = z1.shape[1]
        c_in       = z1.shape[2]
        patch_len  = z1.shape[3]
        
        #proposed
        z1 = torch.reshape(z1, (batch_size*patch_num,c_in,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num,c_in,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]

        ## MISSING??
        z1 = self.norm1(z1)
        z2 = self.norm2(z2)

        z = self.fre_transformer(torch.cat((z1,z2),-1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)
        

        z1 = torch.reshape(z1, (batch_size,patch_num,c_in,z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size,patch_num,c_in,z2.shape[-1]))
        

        z1 = z1.permute(0,2,1,3)                                                                    # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0,2,1,3)

        z1 = self.head_f1(z1)                                                                    # z: [bs x nvars x target_window] 
        z2 = self.head_f2(z2)                                                                    # z: [bs x nvars x target_window]
        
        z = torch.fft.ifft(torch.complex(z1.float(),z2.float()))
        zr = z.real                                              
        zi = z.imag
        z = self.ircom(torch.cat((zr,zi),-1))

        return z

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears1 = nn.ModuleList()
            #self.linears2 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                #self.linears2.append(nn.Linear(target_window, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)                    # z: [bs x target_window]
                #z = self.linears2[i](z)                    # z: [target_window x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
            #x = self.linear1(x)
            #x = self.linear2(x) + x
            #x = self.dropout(x)
        return x


class NARX_Freq_Transformer(nn.Module):    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features_per_metric = int(args.metric_window_duration / args.feature_window_duration)
        self.context = int(args.context_window * self.num_features_per_metric)

        self.fredformer = Fredformer_backbone(cf_dim=args.cf_dim, cf_depth=args.num_encoder_layers, cf_heads=args.num_heads, cf_head_dim=args.cf_head_dim,
                                 cf_mlp=args.mlp_dim, context_window=self.context, target_window=self.context, patch_len=args.cf_patch_len, feature_wise=args.feature_wise,
                                 stride=args.stride, d_model=args.embedding_dim, cf_drop=args.cf_drop, mlp_drop=args.mlp_drop, c_in=args.num_features)
        
        self.to_embedding = nn.Linear(args.num_features, args.embedding_dim)
        self.dec_proj = nn.Linear(1, args.embedding_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embedding_dim, nhead=args.num_heads, dim_feedforward=args.mlp_dim, dropout=self.args.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=args.num_decoder_layers)

        self.position_encoding()
    
    def position_encoding(self,):        
        pe = torch.zeros(self.args.max_len * self.args.num_features_per_metric, self.args.embedding_dim)
        position = torch.arange(0, self.args.max_len * self.args.num_features_per_metric, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.embedding_dim, 2).float() * (-math.log(10000.0) / self.args.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, memory_mask, tgt_mask, **kwargs):
        tgt = tgt.unsqueeze(-1) 
        
        # print(f"gaze_x: {gaze_x.shape}")                # (batch, context_window, num_feat_per_metric)
        z = torch.stack([gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus], dim=1)
        # print(f"NARX input z: {z.shape}")               # (batch, num_feat, context_window, num_feat_per_metric)
        z = rearrange(z, 'b v c n -> b v (c n)')    

        # print(f"Fredformer input z: {z.shape}")         # (batch, num_feat, context_window*num_feat_per_metric)
        out = self.fredformer(z)
        # print(f"Fredformer output: {out.shape}")        # (batch, num_feat, context_window*num_feat_per_metric)

        out = rearrange(out, 'b v c -> b c v')
        # print(f"Fredformer output reshaped: {out.shape}")    # (batch, context_window*num_feat_per_metric, num_feat)

        memory = self.to_embedding(out)
        dec_inp_proj = self.dec_proj(tgt)

        # print(f"memory: {memory.shape}, dec_inp: {dec_inp_proj.shape}") # (batch, context_window, embedding_dim), (batch, context_window, embedding_dim)

        src_input = memory * math.sqrt(self.args.embedding_dim) + self.pe[:, idx*self.args.num_features_per_metric:\
                                                                          (idx + self.args.context_window)*self.args.num_features_per_metric, :]
        tgt_input = dec_inp_proj * math.sqrt(self.args.embedding_dim) + self.pe[:, idx: idx + self.args.context_window, :]

        # print(f"tgt_inp: {tgt_input.shape}, src_input: {src_input.shape}") # (batch, context_window, embedding_dim), (batch, context_window, embedding_dim)
        output = self.decoder(memory=src_input,
                              tgt=tgt_input,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask
                            )

        # print(f"Transformer output: {output.shape}")  # (batch, context_window, embedding_dim)
        return output


class Freq_Transformer(nn.Module):
    def __init__(self, fc, transformer):
        super().__init__()
        self.fc = fc
        self.transformer = transformer

    def fc_forward(self, x):
        fc_out = self.fc(x)
        return fc_out
    
    def transformer_forward(self, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, memory_mask, tgt_mask, **kwargs):        
        output = self.transformer(gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, memory_mask, tgt_mask)        
        return output

    def forward(self, idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, memory_mask, tgt_mask, **kwargs):        
        return self.fc(self.transformer(idx, gaze_x, gaze_y, gaze_vel, gaze_acc, pupil, stimulus, tgt, memory_mask, tgt_mask)).squeeze(-1)