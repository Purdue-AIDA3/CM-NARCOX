import os
import numpy as np
import logging
import argparse

## PyTorch
import torch
import torch.optim.lr_scheduler

## model
from models.time_transformer import Time_Transformer, NARX_Time_Transformer
from models.freq_transformer import Freq_Transformer, NARX_Freq_Transformer
from models.wavelet_transformer import Wavelet_Transformer, NARX_Wavelet_Transformer, Hierarchical_Wavelet_Transformer, NARX_Hierarchical_Wavelet_Transformer, NARX_New_Wavelet_Transformer, New_Wavelet_Transformer
from models.feedforward_head import FCHead
from exp.exp_main import ExpMain
from utils.logger import begin_logging
from utils.utils import set_seeds

## wandb
import wandb
wandb.login()

logger = logging.getLogger(__name__)

torch.manual_seed(8000)
np.random.seed(8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AAAI 2026 - Modeling NARX time series using "Model"')
    # basic config
    parser.add_argument("--seed", type=int, default=8000, help="Random seed for reproducibility")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", type=str, default="time", choices=["time", "freq", "wavelet", "hierarchical", "new"])
    parser.add_argument('--data_path', type=str, default="../../../datasets/EEGEyeNet/raw_dots/") # path to dots data from EEGEyeNet
    parser.add_argument("--context_window", type=int, default=8) # time steps to look back on 
    # if this is 8, that is essentially using 8 previous values of TAR and 8 * metric_window_duration / feature_window_duration
    # (default: 8 * 4 / 0.2 = 160) values of features
    parser.add_argument("--num_workers", type=int, default=5) # number of workers to work on the dataloader
    
    # Arguments for data preprocessing
    parser.add_argument("--sfreq", type=int, default=500) # sampling frequency of EEGEyeNet
    parser.add_argument("--metric_window_duration", type=float, default=4.0) # window length used to calculate TAR
    parser.add_argument("--metric_shift_duration", type=float, default=4.0) # stride between windows that calculate TAR, metric_shift_duration < metric_window_duration means overlap
    parser.add_argument("--feature_window_duration", type=float, default=0.2) # window length used for features
    parser.add_argument("--feature_shift_duration", type=float, default=0.2) # stride between windows used for features, feature_shift_duration < feature_window_duration means overlap
    # the number of samples used is equal to duration * sfreq
    parser.add_argument('--theta_channels', nargs='+', type=int, # Channels from EEG used to calculate theta power
                       default=[10, 23, 123, 32, 121, 21, 8])
    
    parser.add_argument('--alpha_channels', nargs='+', type=int, # Channels from EEG used to calculate alpha power
                       default=[57, 95, 61, 51, 91, 69, 74, 82])

    # Arguments for hyperparameters
    parser.add_argument("--batch_size", type=int, default=4) # batch size to be used
    parser.add_argument("--learning_rate", type=float, default=0.0001) # learning rate for AdamW optimizer
    parser.add_argument("--epochs", type=int, default=500) # number of epochs to run
    
    # Arguments common to all models
    parser.add_argument("--num_encoder_layers", type=int, default=2) # number of encoder layers for transformer
    parser.add_argument("--num_decoder_layers", type=int, default=2) # number of decoder layers for transformer
    parser.add_argument("--embedding_dim", type=int, default=64) # embedding dimension for transformer
    parser.add_argument("--num_heads", type=int, default=8) # number of heads for multi-head attention
    parser.add_argument("--mlp_dim", type=int, default=128) # dimension of MLP in transformer

    # Arguments for time model
    parser.add_argument("--dropout", type=float, default=0.0) # dropout value to use for transformer

    # Arguments for freq model    
    parser.add_argument('--cf_dim',         type=int, default=48)               # feature dimension
    parser.add_argument('--cf_drop',        type=float, default=0.0)            # dropout
    parser.add_argument('--cf_head_dim',    type=int, default=32)               # dimension for single head
    parser.add_argument('--mlp_drop',       type=float, default=0.0)            # output mlp dropout
    parser.add_argument('--cf_patch_len',   type=int, default=16)               # patch length
    parser.add_argument('--stride',         type=int, default=8)                # stride between patches
    parser.add_argument('--feature_wise',   action='store_true')                # feature wise and patch wise normalization vs just patch wise normalization

    # Arguments for wavelet model and hierarchical wavelet model
    parser.add_argument('--patch_len',   type=int, default=10)                      # patch length
    parser.add_argument('--num_scales',      type=int, default=100)                 # number of scales to use for mother wavelet
    parser.add_argument('--min_dilation',    type=float, default=0.1)               # minimum dilation factor for mother wavelet
    parser.add_argument('--max_dilation',    type=float, default=128)               # maximum dilation factor for mother wavelet
    parser.add_argument('--sample_type',     type=str, default="geometric", choices=["geometric", "linear"]) # type of sampling - if geometric, lower frequencies are densely sampled

    # Arguments for new wavelet model
    parser.add_argument('--scale_patch_len', type=int, default=10)
    parser.add_argument('--time_patch_len', type=int, default=10)

    # Arguments for hierarchical wavelet and new wavelet model
    parser.add_argument('--inner_dim', type=int, default=8)                           # inner dimension for encoder

    args = parser.parse_args()
    set_seeds(args.seed)
    
    assert int(np.ceil(args.metric_window_duration / args.feature_window_duration))\
                    == int(args.metric_window_duration / args.feature_window_duration),\
            "Window duration for features and metrics should be compatible"
    
    assert (args.metric_window_duration / args.feature_window_duration / args.time_patch_len).is_integer() \

    args.max_len = int(156182 / args.sfreq / args.metric_window_duration)          # to use as max length for all inputs
    args.num_features = 6                                                          # gaze_x, gaze_y, gaze_vel, gaze_acc, pupil_diameter, stimulus
    args.num_features_per_metric = int(args.metric_window_duration / args.feature_window_duration)

    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    begin_logging(args)

    ############    CREATE COMPLETE MODEL    ############
    if args.model == "time":
        transformer = NARX_Time_Transformer(args=args)        
        fc = FCHead(embedding_dim=args.embedding_dim)        
        model = Time_Transformer(transformer=transformer, fc=fc)

    elif args.model == "freq": # change this to our novel model later
        transformer = NARX_Freq_Transformer(args=args)        
        fc = FCHead(embedding_dim=args.embedding_dim)
        model = Freq_Transformer(transformer=transformer, fc=fc)

    elif args.model == "wavelet": # change this to our novel model later
        transformer = NARX_Wavelet_Transformer(args=args)        
        fc = FCHead(embedding_dim=args.embedding_dim)
        model = Wavelet_Transformer(transformer=transformer, fc=fc)

    elif args.model == "hierarchical":
        transformer = NARX_Hierarchical_Wavelet_Transformer(args=args)
        fc = FCHead(embedding_dim=args.embedding_dim)
        model = Hierarchical_Wavelet_Transformer(transformer=transformer, fc=fc)
    elif args.model == "new":
        transformer = NARX_New_Wavelet_Transformer(args=args)
        fc = FCHead(embedding_dim=args.embedding_dim)
        model = New_Wavelet_Transformer(transformer=transformer, fc=fc)

    wandb.watch(model, log_freq=100)

    logger.info("Model Architecture")
    logger.info(f"{model}\n")

    logger.info("Starting Training\n")

    trainer = ExpMain(args, model)
    trainer.train()

    wandb.finish()