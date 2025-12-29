import os
import logging

import numpy as np
import wandb

def setup_logging(log_filename):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )    
    return logging.getLogger(__name__)

def begin_logging(args):
    project_name = "NARCOX"
    common_path = f"./{project_name}/{args.model}"
    dataset_details = f"c{args.context_window}_mw{args.metric_window_duration}_ms{args.metric_shift_duration}_fw{args.feature_window_duration}_fs{args.feature_shift_duration}"
    

    if args.model == "time":
        # Saving models and logs
        hyper_params = f"emb_dim_{args.embedding_dim}_batch_{args.batch_size}_lr_{args.learning_rate}_enc_l_{args.num_encoder_layers}_dec_l_{args.num_decoder_layers}_h_{args.num_heads}_mlp_dim_{args.mlp_dim}_drop_{args.dropout}"
        args.models_dir = f'{common_path}/models/{dataset_details}/{hyper_params}'
        logs_dir = f'{common_path}/logs/{dataset_details}/'    
        args.plots_dir_train = f'{common_path}/plots/{dataset_details}/{hyper_params}/train/'
        args.plots_dir_val = f'{common_path}/plots/{dataset_details}/{hyper_params}/val/'
        args.plots_dir_test = f'{common_path}/plots/{dataset_details}/{hyper_params}/test/'

        if not os.path.isdir(args.models_dir):
            os.makedirs(args.models_dir)    
        if not os.path.isdir(args.plots_dir_train):
            os.makedirs(args.plots_dir_train)    
        if not os.path.isdir(args.plots_dir_val):
            os.makedirs(args.plots_dir_val)    
        if not os.path.isdir(args.plots_dir_test):
            os.makedirs(args.plots_dir_test)    
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        filename = f"{logs_dir}/{hyper_params}.log"
        logger = setup_logging(filename)

        wandb.init(
            project=project_name,  # set your W&B project name
            config={
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "embedding_dim": args.embedding_dim,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "num_heads": args.num_heads,
                "context_window": args.context_window,
                "metric_window_duration": args.metric_window_duration,
                "metric_shift_duration": args.metric_shift_duration,
                "feature_window_duration": args.feature_window_duration,
                "feature_shift_duration": args.feature_shift_duration
            },
            tags=[f"{args.model}_{dataset_details}"],
            name=f"{hyper_params}"
        )

        logger.info("-------------------------------------")
        logger.info("Preprocessing Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Context Window: {args.context_window}")
        logger.info(f"Metric Window Duration: {args.metric_window_duration}")
        logger.info(f"Metric Shift Duration: {args.metric_shift_duration}")
        logger.info(f"Feature Window Duration: {args.feature_window_duration}")
        logger.info(f"Feature Shift Duration: {args.feature_shift_duration}")
        logger.info(f"Theta Channels: {args.theta_channels}")
        logger.info(f"Alpha Channels: {args.alpha_channels}")
        logger.info("-------------------------------------")
        logger.info("Model Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Embedding Dim: {args.embedding_dim}")
        logger.info(f"MLP Dim: {args.mlp_dim}")
        logger.info(f"Encoder Layers: {args.num_encoder_layers}")
        logger.info(f"Decoder Layers: {args.num_decoder_layers}")
        logger.info(f"Spatial Heads: {args.num_heads}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info("-------------------------------------")
        logger.info("Optimization Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info("-------------------------------------\n")
    
    elif args.model == "freq":
        # Saving models and logs
        hyper_params = f"cf_dim_{args.cf_dim}_cf_depth_{args.num_encoder_layers}_cf_heads_{args.num_heads}_cf_head_dim_{args.cf_head_dim}_cf_mlp_{args.mlp_dim}_plen_{args.cf_patch_len}_stride_{args.stride}_emb_dim_{args.embedding_dim}_batch_{args.batch_size}_lr_{args.learning_rate}_dec_l_{args.num_decoder_layers}_mlp_dim_{args.mlp_dim}_fw_{args.feature_wise}_cf_drop_{args.cf_drop}_mlp_drop_{args.mlp_drop}_decdrop_{args.dropout}"
        args.models_dir = f'{common_path}/models/{dataset_details}/{hyper_params}'
        logs_dir = f'{common_path}/logs/{dataset_details}/'    
        args.plots_dir_train = f'{common_path}/plots/{dataset_details}/{hyper_params}/train/'
        args.plots_dir_val = f'{common_path}/plots/{dataset_details}/{hyper_params}/val/'
        args.plots_dir_test = f'{common_path}/plots/{dataset_details}/{hyper_params}/test/'

        if not os.path.isdir(args.models_dir):
            os.makedirs(args.models_dir)    
        if not os.path.isdir(args.plots_dir_train):
            os.makedirs(args.plots_dir_train)    
        if not os.path.isdir(args.plots_dir_val):
            os.makedirs(args.plots_dir_val)    
        if not os.path.isdir(args.plots_dir_test):
            os.makedirs(args.plots_dir_test)    
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        filename = f"{logs_dir}/{hyper_params}.log"
        logger = setup_logging(filename)

        wandb.init(
            project=project_name,  # set your W&B project name
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "embedding_dim": args.embedding_dim,
                "cf_dim": args.cf_dim,
                "cf_mlp": args.mlp_dim,
                "cf_depth": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "num_heads": args.num_heads,
                "head_dim": args.cf_head_dim,
                "patch_len": args.cf_patch_len,
                "stride": args.stride,
                "feature_wise": args.feature_wise,
                "cf_drop": args.cf_drop,
                "mlp_drop": args.mlp_drop,
                "dec_drop": args.dropout,
                "context_window": args.context_window,
                "metric_window_duration": args.metric_window_duration,
                "metric_shift_duration": args.metric_shift_duration,
                "feature_window_duration": args.feature_window_duration,
                "feature_shift_duration": args.feature_shift_duration
            },
            tags=[f"{args.model}_{dataset_details}"],
            name=f"{hyper_params}"
        )

        logger.info("-------------------------------------")
        logger.info("Preprocessing Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Context Window: {args.context_window}")
        logger.info(f"Metric Window Duration: {args.metric_window_duration}")
        logger.info(f"Metric Shift Duration: {args.metric_shift_duration}")
        logger.info(f"Feature Window Duration: {args.feature_window_duration}")
        logger.info(f"Feature Shift Duration: {args.feature_shift_duration}")
        logger.info(f"Theta Channels: {args.theta_channels}")
        logger.info(f"Alpha Channels: {args.alpha_channels}")
        logger.info("-------------------------------------")
        logger.info("Model Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"embedding_dim: {args.embedding_dim}")
        logger.info(f"cf_dim: {args.cf_dim}")
        logger.info(f"cf_mlp: {args.mlp_dim}")
        logger.info(f"cf_depth: {args.num_encoder_layers}")
        logger.info(f"cf_heads: {args.num_heads}")
        logger.info(f"cf_head_dim: {args.cf_head_dim}")
        logger.info(f"patch_len: {args.cf_patch_len}")
        logger.info(f"stride: {args.stride}")
        logger.info(f"Decoder Layers: {args.num_decoder_layers}")
        logger.info(f"feature_wise: {args.feature_wise}")
        logger.info(f"cf_drop: {args.cf_drop}")
        logger.info(f"mlp_drop: {args.mlp_drop}")
        logger.info(f"dec_drop: {args.dropout}")
        logger.info("-------------------------------------")
        logger.info("Optimization Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info("-------------------------------------\n")
    
    elif args.model == "wavelet":
        
        assert int(np.ceil(args.num_scales / args.patch_len))\
                        == int(args.num_scales / args.patch_len),\
                "Wavelet number of scales and patch lengths should be compatible"
        # Saving models and logs
        hyper_params = f"cmor1.5-1.0_nscales_{args.num_scales}_min_dil_{args.min_dilation}_max_dil_{args.max_dilation}_type_{args.sample_type}_patch_len_{args.patch_len}_emb_dim_{args.embedding_dim}_batch_{args.batch_size}_lr_{args.learning_rate}_enc_l_{args.num_encoder_layers}_dec_l_{args.num_decoder_layers}_h_{args.num_heads}_mlp_dim_{args.mlp_dim}_drop_{args.dropout}"
        args.models_dir = f'{common_path}/models/{dataset_details}/{hyper_params}'
        logs_dir = f'{common_path}/logs/{dataset_details}/'    
        args.plots_dir_train = f'{common_path}/plots/{dataset_details}/{hyper_params}/train/'
        args.plots_dir_val = f'{common_path}/plots/{dataset_details}/{hyper_params}/val/'
        args.plots_dir_test = f'{common_path}/plots/{dataset_details}/{hyper_params}/test/'

        if not os.path.isdir(args.models_dir):
            os.makedirs(args.models_dir)    
        if not os.path.isdir(args.plots_dir_train):
            os.makedirs(args.plots_dir_train)    
        if not os.path.isdir(args.plots_dir_val):
            os.makedirs(args.plots_dir_val)    
        if not os.path.isdir(args.plots_dir_test):
            os.makedirs(args.plots_dir_test)    
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        filename = f"{logs_dir}/{hyper_params}.log"
        logger = setup_logging(filename)

        wandb.init(
            project=project_name,  # set your W&B project name
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "wavelet": "cmor1.5-1.0",
                "num_scales": args.num_scales,
                "min_dilation": args.min_dilation,
                "max_dilation": args.max_dilation,
                "sample_type": args.sample_type,
                "patch_len": args.patch_len,
                "embedding_dim": args.embedding_dim,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "num_heads": args.num_heads,
                "context_window": args.context_window,
                "metric_window_duration": args.metric_window_duration,
                "metric_shift_duration": args.metric_shift_duration,
                "feature_window_duration": args.feature_window_duration,
                "feature_shift_duration": args.feature_shift_duration
            },
            tags=[f"{args.model}_{dataset_details}"],
            name=f"{hyper_params}"
        )

        logger.info("-------------------------------------")
        logger.info("Preprocessing Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Context Window: {args.context_window}")
        logger.info(f"Metric Window Duration: {args.metric_window_duration}")
        logger.info(f"Metric Shift Duration: {args.metric_shift_duration}")
        logger.info(f"Feature Window Duration: {args.feature_window_duration}")
        logger.info(f"Feature Shift Duration: {args.feature_shift_duration}")
        logger.info(f"Theta Channels: {args.theta_channels}")
        logger.info(f"Alpha Channels: {args.alpha_channels}")
        logger.info("-------------------------------------")
        logger.info("Model Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"wavelet: cmor1.5-1.0"),
        logger.info(f"num_scales: {args.num_scales}"),
        logger.info(f"min_dilation: {args.min_dilation}"),
        logger.info(f"max_dilation: {args.max_dilation}"),
        logger.info(f"sample_type: {args.sample_type}"),
        logger.info(f"patch_len: {args.patch_len}"),
        logger.info(f"Embedding Dim: {args.embedding_dim}")
        logger.info(f"MLP Dim: {args.mlp_dim}")
        logger.info(f"Encoder Layers: {args.num_encoder_layers}")
        logger.info(f"Decoder Layers: {args.num_decoder_layers}")
        logger.info(f"Spatial Heads: {args.num_heads}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info("-------------------------------------")
        logger.info("Optimization Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info("-------------------------------------\n")

    elif args.model == "hierarchical":
        
        assert int(np.ceil(args.num_scales / args.patch_len))\
                        == int(args.num_scales / args.patch_len),\
                "Wavelet number of scales and patch lengths should be compatible"

        # Saving models and logs
        hyper_params = f"hier_cmor1.5-1.0_nscales_{args.num_scales}_min_dil_{args.min_dilation}_max_dil_{args.max_dilation}_type_{args.sample_type}_inner_dim_{args.inner_dim}_plen_{args.patch_len}_emb_dim_{args.embedding_dim}_batch_{args.batch_size}_lr_{args.learning_rate}_enc_l_{args.num_encoder_layers}_dec_l_{args.num_decoder_layers}_h_{args.num_heads}_mlp_dim_{args.mlp_dim}_drop_{args.dropout}"
        args.models_dir = f'{common_path}/models/{dataset_details}/{hyper_params}'
        logs_dir = f'{common_path}/logs/{dataset_details}/'    
        args.plots_dir_train = f'{common_path}/plots/{dataset_details}/{hyper_params}/train/'
        args.plots_dir_val = f'{common_path}/plots/{dataset_details}/{hyper_params}/val/'
        args.plots_dir_test = f'{common_path}/plots/{dataset_details}/{hyper_params}/test/'

        if not os.path.isdir(args.models_dir):
            os.makedirs(args.models_dir)    
        if not os.path.isdir(args.plots_dir_train):
            os.makedirs(args.plots_dir_train)    
        if not os.path.isdir(args.plots_dir_val):
            os.makedirs(args.plots_dir_val)    
        if not os.path.isdir(args.plots_dir_test):
            os.makedirs(args.plots_dir_test)    
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        filename = f"{logs_dir}/{hyper_params}.log"
        logger = setup_logging(filename)

        wandb.init(
            project=project_name,  # set your W&B project name
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "wavelet": "cmor1.5-1.0",
                "num_scales": args.num_scales,
                "min_dilation": args.min_dilation,
                "max_dilation": args.max_dilation,
                "sample_type": args.sample_type,
                "inner_dim": args.inner_dim,
                "patch_len": args.patch_len,
                "embedding_dim": args.embedding_dim,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "num_heads": args.num_heads,
                "context_window": args.context_window,
                "metric_window_duration": args.metric_window_duration,
                "metric_shift_duration": args.metric_shift_duration,
                "feature_window_duration": args.feature_window_duration,
                "feature_shift_duration": args.feature_shift_duration
            },
            tags=[f"{args.model}_{dataset_details}"],
            name=f"{hyper_params}"
        )

        logger.info("-------------------------------------")
        logger.info("Preprocessing Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Context Window: {args.context_window}")
        logger.info(f"Metric Window Duration: {args.metric_window_duration}")
        logger.info(f"Metric Shift Duration: {args.metric_shift_duration}")
        logger.info(f"Feature Window Duration: {args.feature_window_duration}")
        logger.info(f"Feature Shift Duration: {args.feature_shift_duration}")
        logger.info(f"Theta Channels: {args.theta_channels}")
        logger.info(f"Alpha Channels: {args.alpha_channels}")
        logger.info("-------------------------------------")
        logger.info("Model Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"wavelet: cmor1.5-1.0"),
        logger.info(f"num_scales: {args.num_scales}"),
        logger.info(f"min_dilation: {args.min_dilation}")
        logger.info(f"max_dilation: {args.max_dilation}")
        logger.info(f"sample_type: {args.sample_type}")
        logger.info(f"patch_len: {args.patch_len}")
        logger.info(f"Embedding Dim: {args.embedding_dim}")
        logger.info(f"Inner Dim: {args.inner_dim}")
        logger.info(f"MLP Dim: {args.mlp_dim}")
        logger.info(f"Encoder Layers: {args.num_encoder_layers}")
        logger.info(f"Decoder Layers: {args.num_decoder_layers}")
        logger.info(f"Spatial Heads: {args.num_heads}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info("-------------------------------------")
        logger.info("Optimization Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info("-------------------------------------\n")


    elif args.model == "new":
        
        assert int(np.ceil(args.num_scales / args.scale_patch_len))\
                        == int(args.num_scales / args.scale_patch_len),\
                "Wavelet number of scales and patch lengths should be compatible"
        
        assert int(np.ceil(args.context_window * args.metric_window_duration / args.feature_window_duration / args.time_patch_len))\
                        == int(args.context_window * args.metric_window_duration / args.feature_window_duration / args.time_patch_len),\
                "Wavelet time sequence length and patch length should be compatible"

        # Saving models and logs
        hyper_params = f"new_cmor1.5-1.0_nscales_{args.num_scales}_min_dil_{args.min_dilation}_max_dil_{args.max_dilation}_type_{args.sample_type}_inner_dim_{args.inner_dim}_tplen_{args.time_patch_len}_splen_{args.scale_patch_len}_emb_dim_{args.embedding_dim}_batch_{args.batch_size}_lr_{args.learning_rate}_enc_l_{args.num_encoder_layers}_dec_l_{args.num_decoder_layers}_h_{args.num_heads}_mlp_dim_{args.mlp_dim}_drop_{args.dropout}"
        args.models_dir = f'{common_path}/models/{dataset_details}/{hyper_params}'
        logs_dir = f'{common_path}/logs/{dataset_details}/'    
        args.plots_dir_train = f'{common_path}/plots/{dataset_details}/{hyper_params}/train/'
        args.plots_dir_val = f'{common_path}/plots/{dataset_details}/{hyper_params}/val/'
        args.plots_dir_test = f'{common_path}/plots/{dataset_details}/{hyper_params}/test/'

        if not os.path.isdir(args.models_dir):
            os.makedirs(args.models_dir)    
        if not os.path.isdir(args.plots_dir_train):
            os.makedirs(args.plots_dir_train)    
        if not os.path.isdir(args.plots_dir_val):
            os.makedirs(args.plots_dir_val)    
        if not os.path.isdir(args.plots_dir_test):
            os.makedirs(args.plots_dir_test)    
        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)

        filename = f"{logs_dir}/{hyper_params}.log"
        logger = setup_logging(filename)

        wandb.init(
            project=project_name,  # set your W&B project name
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "wavelet": "cmor1.5-1.0",
                "num_scales": args.num_scales,
                "min_dilation": args.min_dilation,
                "max_dilation": args.max_dilation,
                "sample_type": args.sample_type,
                "inner_dim": args.inner_dim,
                "scale_patch_len": args.scale_patch_len,
                "time_patch_len": args.time_patch_len,
                "embedding_dim": args.embedding_dim,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "num_heads": args.num_heads,
                "context_window": args.context_window,
                "metric_window_duration": args.metric_window_duration,
                "metric_shift_duration": args.metric_shift_duration,
                "feature_window_duration": args.feature_window_duration,
                "feature_shift_duration": args.feature_shift_duration
            },
            tags=[f"{args.model}_{dataset_details}"],
            name=f"{hyper_params}"
        )

        logger.info("-------------------------------------")
        logger.info("Preprocessing Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Context Window: {args.context_window}")
        logger.info(f"Metric Window Duration: {args.metric_window_duration}")
        logger.info(f"Metric Shift Duration: {args.metric_shift_duration}")
        logger.info(f"Feature Window Duration: {args.feature_window_duration}")
        logger.info(f"Feature Shift Duration: {args.feature_shift_duration}")
        logger.info(f"Theta Channels: {args.theta_channels}")
        logger.info(f"Alpha Channels: {args.alpha_channels}")
        logger.info("-------------------------------------")
        logger.info("Model Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"wavelet: cmor1.5-1.0"),
        logger.info(f"num_scales: {args.num_scales}"),
        logger.info(f"min_dilation: {args.min_dilation}")
        logger.info(f"max_dilation: {args.max_dilation}")
        logger.info(f"sample_type: {args.sample_type}")
        logger.info(f"time patch_len: {args.time_patch_len}")
        logger.info(f"scale patch_len: {args.scale_patch_len}")
        logger.info(f"Embedding Dim: {args.embedding_dim}")
        logger.info(f"Inner Dim: {args.inner_dim}")
        logger.info(f"MLP Dim: {args.mlp_dim}")
        logger.info(f"Encoder Layers: {args.num_encoder_layers}")
        logger.info(f"Decoder Layers: {args.num_decoder_layers}")
        logger.info(f"Spatial Heads: {args.num_heads}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info("-------------------------------------")
        logger.info("Optimization Hyperparameters")
        logger.info("-------------------------------------")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning Rate: {args.learning_rate}")
        logger.info("-------------------------------------\n")

