import os
import matplotlib.pyplot as plt

import numpy as np

import torch
import ptwt
import torch.nn.functional as F

def plot_signal(args, true, preds, epoch, save_dir='plots/time'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 6))
    plt.plot(true, label='Ground Truth', color='blue', linewidth=2)
    plt.plot(preds, label='Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'{args.type} - Epoch {epoch+1}')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()    
    save_path = os.path.join(save_dir, f'{args.type}_epoch_{epoch+1}.png')
    plt.savefig(save_path)
    plt.close()


def plot_spectrum(args, true, preds, epoch, save_dir="plots/freq"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n = true.shape[0]
    # Use rfft for real-valued signals
    true_freq = torch.fft.rfft(true)
    preds_freq = torch.fft.rfft(preds)
    true_mag = torch.abs(true_freq) / n
    preds_mag = torch.abs(preds_freq) / n
    freqs = torch.fft.rfftfreq(n, d=1 / args.sfreq)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs.numpy()[5:], true_mag.numpy()[5:], label='True Spectrum')
    plt.plot(freqs.numpy()[5:], preds_mag.numpy()[5:], label='Prediction Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title(f'{args.type} - Epoch {epoch+1}')
    # plt.legend()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{args.type}_epoch_{epoch+1}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Frequency Spectra MSE: {F.mse_loss(true_mag, preds_mag).item()}")


def plot_scaleogram(args, true, preds, epoch, save_dir="plots/wavelet"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n = true.shape[0]

    time = np.linspace(0, int(args.seq_len/args.sfreq), args.seq_len, endpoint=False)
    widths = torch.Tensor(np.geomspace(1, 128, num=args.num_scales))
    wavelet = "cmor1.5-1.0"
    
    cwt_true, freqs = ptwt.cwt(true, widths, wavelet, sampling_period=1/args.sfreq)
    cwt_preds, _ = ptwt.cwt(preds, widths, wavelet, sampling_period=1/args.sfreq)

    true_mag = torch.abs(cwt_true)
    preds_mag = torch.abs(cwt_preds)
    
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

    pcm2 = ax2.pcolormesh(time, freqs, preds_mag, shading='auto', vmin=0.0, vmax=15.0, cmap="plasma")
    # ax2.set_title(f'Predicted Scalogram')
    ax2.set_ylim(0, 50)  # Limit frequency range to 0-50 Hz
    ax2.tick_params(axis='both', which='major', labelsize=25)
    # ax2.set_xlabel('Time (s)');  ax2.set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(pcm2, ax=ax2)
    cbar.ax.tick_params(labelsize=25)

    # plt.title(f'{args.type} - Epoch {epoch+1}')
    save_path = os.path.join(save_dir, f'{args.type}_epoch_{epoch+1}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Wavelet Scaleogram MSE: {F.mse_loss(true_mag, preds_mag).item()}")
