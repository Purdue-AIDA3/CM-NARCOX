import math
import random

## numpy
import numpy as np

## PyTorch
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Transformer

## statsmodels
from statsmodels.tsa.seasonal import STL

## matplotlib
import matplotlib.pyplot as plt
 
## MODEL UTILS
def generate_square_subsequent_mask(sz):
    return Transformer.generate_square_subsequent_mask(sz).bool()   

def create_causal_memory_mask(T, num_feat_per_metric, device):
    S = T * num_feat_per_metric    
    # Create decoder positions: [0, 1, 2, ..., T-1]
    decoder_positions = torch.arange(T).unsqueeze(1)  # Shape: [T, 1]    
    # Create encoder positions: [0, 1, 2, ..., S-1]
    encoder_positions = torch.arange(S).unsqueeze(0)  # Shape: [1, S]    
    # Calculate available encoder positions for each decoder position
    available_positions = (decoder_positions + 1) * num_feat_per_metric  # Shape: [T, 1]    
    # Create mask: True where encoder position >= available positions (mask out future)
    memory_mask = encoder_positions >= available_positions  # Shape: [T, S]    
    return memory_mask.to(device)

def create_causal_memory_mask_new(T, num_feat_per_metric, time_patch_len, device):
    stride = num_feat_per_metric // time_patch_len
    S = T * stride  # total encoder positions
    
    decoder_positions = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
    encoder_positions = torch.arange(S, device=device).unsqueeze(0)  # [1, S]

    # For each row t, cutoff index is (t+1)*stride
    cutoff = (decoder_positions + 1) * stride  # [T, 1]

    # True where encoder index >= cutoff  (mask out)
    memory_mask = encoder_positions >= cutoff  
    return memory_mask.to(device)


def padding_mask(seq_lengths, max_len, device=None):
    mask = torch.arange(max_len).expand(len(seq_lengths), max_len) >= seq_lengths.unsqueeze(1)
    return mask.bool().to(device)

def sliding_causal_mask(seq_len, window, device=None):
        """
        Builds an SxS boolean mask enforcing:
          - mask[i,j] = True if j>i (no future)
          - mask[i,j] = True if i-j>window (beyond lookback)
        Returns:
          torch.Tensor of shape [seq_len, seq_len], dtype=bool.
        """
        idx = torch.arange(seq_len, device=device)
        i = idx.view(seq_len, 1)
        j = idx.view(1, seq_len)
        return (j > i) | ((i - j) >= window)

def generate_batch_safe_mask(seq_len, window, pad_mask, num_heads, device):
    """
    Build a [B, S, S] boolean mask combining:
      - Sliding-window causal mask (j>i or i-j>=window)
      - Per-sample padding (pad_mask)
    Then unmask the diagonal so every query has at least itself unmasked.
    True = masked.
    """
    idx = torch.arange(seq_len, device=device)
    i, j = idx.view(seq_len,1), idx.view(1,seq_len)
    base = (j > i) | ((i - j) >= window)       # [S,S]
    combined = base.unsqueeze(0) | pad_mask.unsqueeze(1)  # [B,S,S]
    diag = torch.arange(seq_len, device=device)
    combined[:, diag, diag] = False            # allow self-attention
    return combined.repeat_interleave(num_heads, dim=0)  # [B*H,S,S]

def generate_batch_safe_cross_mask(seq_len, window, pad_mask, num_heads, device):
    idx = torch.arange(seq_len, device=device)
    i, j = idx.view(seq_len,1), idx.view(1,seq_len)
    base = (j > i) | ((i - j) >= window)
    combined = base.unsqueeze(0) | pad_mask.unsqueeze(1)
    diag = torch.arange(seq_len, device=device)
    combined[:, diag, diag] = False        
    return combined 


## OPTIMIZER AND ANALYSIS UTILS
def get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def analyze_spikes(y_pred, y_true, masks, spike_sensitivity=1.2, time_tolerance=1, magnitude_tolerance=0.1, use_magnitude_check=False):
    assert y_pred.shape == y_true.shape == masks.shape, f"Shapes must match but y_pred has shape {y_pred.shape}, y_true has shape {y_true.shape} and masks has shape {masks.shape}"
    assert time_tolerance >= 0, "time_tolerance must be non-negative"

    # time tolerance lets the model "match" spikes from t - time_tolerance to t + time_tolerance, even if a 1 to 1 spike is not found
    B, N = y_pred.shape
    masks = masks.bool() # Ensure boolean mask

    # Apply masks
    # Use float() for std calculation compatibility, handle potential NaNs if all masked
    y_pred_masked = torch.where(masks, y_pred, torch.tensor(torch.nan, device=y_pred.device))
    y_true_masked = torch.where(masks, y_true, torch.tensor(torch.nan, device=y_true.device))

    # Calculate differences between consecutive values (only where mask is False, ignoring padding)
    # Pad with zero difference at the start for the first difference
    pred_diff = torch.diff(y_pred, prepend=y_pred[:, :1]) * masks
    true_diff = torch.diff(y_true, prepend=y_true[:, :1]) * masks

    # Calculate standard deviations robustly on unmasked data
    pred_std_list = [torch.std(y_pred_masked[b][masks[b]]) if masks[b].any() else torch.tensor(0.0, device=y_pred.device) for b in range(B)]
    true_std_list = [torch.std(y_true_masked[b][masks[b]]) if masks[b].any() else torch.tensor(0.0, device=y_true.device) for b in range(B)]
    pred_std = torch.stack(pred_std_list).unsqueeze(1) # Shape (B, 1)
    true_std = torch.stack(true_std_list).unsqueeze(1) # Shape (B, 1)

    # Identify spikes based on positive difference exceeding threshold
    # Ensure threshold is non-zero to avoid division by zero or trivial spikes
    pred_spikes = (pred_diff > (pred_std * spike_sensitivity)) & (pred_std > 1e-6) & masks
    true_spikes = (true_diff > (true_std * spike_sensitivity)) & (true_std > 1e-6) & masks

    # --- Match Spikes with Temporal Tolerance ---
    true_positives = 0
    # Keep track of which predicted spikes have been matched
    pred_spikes_matched = torch.zeros_like(pred_spikes)

    for b in range(B):
        # Find indices of true spikes
        true_spike_indices = torch.where(true_spikes[b])[0]

        for t_true in true_spike_indices:
            # Define the search window around the true spike
            t_start = max(0, t_true - time_tolerance)
            t_end = min(N, t_true + time_tolerance + 1) # +1 because slice is exclusive of the end index

            # Find predicted spikes within the window that haven't been matched yet
            window_pred_indices = torch.where(pred_spikes[b, t_start:t_end] & ~pred_spikes_matched[b, t_start:t_end])[0]

            if len(window_pred_indices) > 0:
                # A match is found! Select the first available predicted spike in the window
                t_pred_relative = window_pred_indices[0]
                t_pred_absolute = t_start + t_pred_relative

                # Optional: Check magnitude similarity if requested
                magnitude_match = True # Assume match unless check fails
                if use_magnitude_check:
                    true_change = true_diff[b, t_true]
                    pred_change = pred_diff[b, t_pred_absolute]
                    # Avoid division by zero if true_change is very small
                    if abs(true_change) > 1e-6:
                         if abs(pred_change - true_change) > abs(magnitude_tolerance * true_change):
                             magnitude_match = False
                    elif abs(pred_change) > 1e-6: # True change is zero, pred change is not
                        magnitude_match = False

                if magnitude_match:
                    true_positives += 1
                    pred_spikes_matched[b, t_pred_absolute] = True # Mark this predicted spike as used

    # --- Calculate TP, FP, FN, TN ---
    TP = true_positives
    total_true_spikes = torch.sum(true_spikes).item()
    total_pred_spikes = torch.sum(pred_spikes).item()

    true_negatives_mask = ~true_spikes & ~pred_spikes & masks
    TN = torch.sum(true_negatives_mask).item()

    # FP = All predicted spikes - those that were matched (TP)
    FP = total_pred_spikes - TP
    # FN = All true spikes - those that were matched (TP)
    FN = total_true_spikes - TP

    return total_true_spikes, total_pred_spikes, TP, FP, FN, TN

def analyse_trend(y_pred, y_true, mask_tensor):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    mask_np = mask_tensor.detach().cpu().numpy().astype(bool)
    B, N = y_true_np.shape
    mse_list = []

    for i in range(B):
        valid_idx = mask_np[i]
        true_seq = y_true_np[i][valid_idx]
        pred_seq = y_pred_np[i][valid_idx]
        if len(true_seq) < 7 or len(pred_seq) < 7:
            # Not enough points for STL, use simple MSE
            mse_list.append(np.mean((true_seq - pred_seq) ** 2))
            continue
        # STL decomposition with period=2 (minimum allowed)
        stl_true = STL(true_seq, period=2, trend=15, robust=True).fit()
        stl_pred = STL(pred_seq, period=2, trend=15, robust=True).fit()
        trend_true = stl_true.trend
        trend_pred = stl_pred.trend
        mse = np.mean((trend_true - trend_pred) ** 2)
        mse_list.append(mse)
    return mse_list

# Set seeds for reproducibility
def set_seeds(seed=8000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize(epoch, save_path, preds, labels, seq_len, title):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for i in range(5):
        plt.figure(figsize=(14, 6))
        plt.plot(labels[i][1: seq_len + 1], label='Labels', marker='o')
        plt.plot(preds[i][1: seq_len + 1], label='Predictions', marker='x')
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("TAR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/{epoch}_{i}.png")
        plt.close()


# def visualize(epoch, save_path, preds, labels, seq_len, title):
#     # Denormalize (reverse log1p) while still as tensors
#     preds = torch.expm1(preds)
#     labels = torch.expm1(labels)
    
#     # Now detach, move to CPU, and convert to NumPy for plotting
#     preds = preds.detach().cpu().numpy()
#     labels = labels.detach().cpu().numpy()
    
#     for i in range(5):
#         plt.figure(figsize=(14, 6))
#         plt.plot(labels[i][1: seq_len + 1], linewidth=3)
#         plt.plot(preds[i][1: seq_len + 1], linewidth=3)
#         # plt.title(title)
#         # plt.xlabel("Time Step")
#         # plt.ylabel("TAR (Original Scale)")
#         # plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"{save_path}/{epoch}_{i}.png")
#         plt.close()