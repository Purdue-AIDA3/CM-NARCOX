import os
import torch
import torch.nn as nn

from generate_data import generate_dataset
from vanilla_model import Vanilla_Model, FCHead, Transformer
from processing import plot_signal, plot_spectrum, plot_scaleogram

def set_seed(seed=42):
    """Set seeds for reproducibility across all random number generators."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

class Args:
    sfreq = 100
    embedding_dim = 32
    seq_len = 1000
    num_heads = 8
    num_layers = 2
    num_scales = 100
    num_features = 2
    patch_len = 20
    type = "time"
    epochs = 1000
    learning_rate = 0.0001
    seed = 1

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        # Compute mean and std for normalization (per sample, across sequence)
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + 1e-8  # Avoid division by zero

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor):
        return x * self.std + self.mean

class Predict:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=0.01
        )
        self.criterion = nn.MSELoss()
        # Initialize normalizers
        self.feature1_normalizer = Normalizer()
        self.feature2_normalizer = Normalizer()
        self.target_normalizer = Normalizer()

    def plot(self, true_norm, preds_norm, epoch):
        # Denormalize using target's normalizer for plotting in original scale
        true = self.target_normalizer.denormalize(true_norm).detach().cpu()[0]
        preds = self.target_normalizer.denormalize(preds_norm).detach().cpu()[0]
        
        plot_signal(self.args, true, preds, epoch)
        plot_spectrum(self.args, true, preds, epoch)
        plot_scaleogram(self.args, true, preds, epoch)

    def train(self, feature1, feature2, target):
        # Fit normalizers on original data
        self.feature1_normalizer.fit(feature1)
        self.feature2_normalizer.fit(feature2)
        self.target_normalizer.fit(target)
        
        # Normalize inputs and target
        feature1_norm = self.feature1_normalizer.normalize(feature1)
        feature2_norm = self.feature2_normalizer.normalize(feature2)
        target_norm = self.target_normalizer.normalize(target)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(feature1_norm, feature2_norm).squeeze(-1)
            loss = self.criterion(predictions, target_norm)
            
            loss.backward()
            self.optimizer.step()
            
        print(f'Epoch [{epoch+1}/{self.args.epochs}], Loss: {loss.item():.4f}')
        self.plot(target_norm, predictions, epoch)

if __name__ == "__main__":
    args = Args()
    
    # Set seed for reproducibility - MUST be called first!
    set_seed(args.seed)
    
    print(args.type)
    if args.type == "wavelet":
        args.patch_len = 5
    
    # Generate data
    feature1, feature2, target, time = generate_dataset(
        sampling_rate=args.sfreq, 
        duration=int(args.seq_len / args.sfreq)
    )
    
    # Initialize model
    fc = FCHead(args=args)
    transformer = Transformer(args=args)
    model = Vanilla_Model(fc=fc, transformer=transformer)
    
    # Convert to tensors (batch size 1)
    feature1 = torch.Tensor(feature1).unsqueeze(0)
    feature2 = torch.Tensor(feature2).unsqueeze(0)
    target = torch.Tensor(target).unsqueeze(0)
    
    # Train
    trainer = Predict(args, model)
    trainer.train(feature1, feature2, target)
