import numpy as np
import logging

## PyTorch
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import glob
from scipy.signal import welch
from scipy.signal import savgol_filter
from pymatreader import read_mat

logger = logging.getLogger(__name__)

class Dataset_EEGEyeNet(Dataset):
    def __init__(self, args, flag):
        self.args = args
        self.sfreq = args.sfreq
        self.metric_window_duration = args.metric_window_duration
        self.metric_shift_duration = args.metric_shift_duration
        self.feature_window_duration = args.feature_window_duration
        self.feature_shift_duration = args.feature_shift_duration
        self.theta_channels = args.theta_channels
        self.alpha_channels = args.alpha_channels
        self.data_path = args.data_path
        self.flag = flag

        self._read_data()

    def _read_data(self, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42):
        """
        Build the sequence index for all possible sliding windows.
        Splits into train/val/test based on subject folders.
        """
        # List all subject folders (not files!)
        subject_folders = sorted([os.path.join(self.data_path, d) for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))])
        n_subjects = len(subject_folders)
        logger.info(f"Found {n_subjects} subjects.")

        # Compute split sizes
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        n_test = n_subjects - n_train - n_val

        if self.flag == 'train':
            selected_folders = subject_folders[:n_train]
        elif self.flag == 'val':
            selected_folders = subject_folders[n_train:n_train + n_val]
        elif self.flag == 'test':
            selected_folders = subject_folders[n_train + n_val:]
        else:
            raise ValueError(f"Unknown flag: {self.flag}")

        logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

        self.file_paths = []
        for folder in selected_folders:
           self.file_paths += sorted(glob.glob(os.path.join(folder, '*.mat')))   

        logger.info(f"{self.flag} subjects: {len(selected_folders)} files: {len(self.file_paths)}")      

        self.num_feat_per_metric = int(np.ceil(self.metric_window_duration / self.feature_window_duration)) 
        logger.info(f"Number of feature values per metric value: {self.num_feat_per_metric}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        d = sio.loadmat(self.file_paths[idx], simplify_cells=True)
        sEEG = d['sEEG']
        data = sEEG["data"]  # [num_features, n_samples]
        tar, avg_pupil, avg_velocity, avg_acc, avg_gaze_x, avg_gaze_y = self.get_metric_and_features(data,
                                                                    metric_window_duration=self.metric_window_duration,
                                                                    metric_shift_duration=self.metric_shift_duration,
                                                                    theta_channels=self.theta_channels,
                                                                    alpha_channels=self.alpha_channels,
                                                                    feature_window_duration=self.feature_window_duration,
                                                                    feature_shift_duration=self.feature_shift_duration,
                                                                )
        
        stimulus = self.get_stimulus_array(self.file_paths[idx],
                                           feature_window_duration=self.feature_window_duration,
                                           feature_shift_duration=self.feature_shift_duration,
                                           TAR = tar
                                        )
        

        # print(tar.shape, avg_pupil.shape, avg_velocity.shape, avg_acc.shape, avg_gaze_x.shape, avg_gaze_y.shape)
        return (tar, avg_pupil, avg_velocity, avg_acc, avg_gaze_x, avg_gaze_y, stimulus)

    def create_windows(self, data, window_duration, shift_duration):
        # Calculate the number of samples in each window and the shift samples
        channels, samples = data.shape
        window_samples = int(window_duration * self.sfreq)
        shift_samples = int(shift_duration * self.sfreq)
        # Create windows using a sliding window approach
        num_windows = (samples - window_samples) // shift_samples + 1
        windows = np.zeros((channels, num_windows, window_samples))

        for i in range(num_windows):
            start = i * shift_samples
            windows[:, i, :] = data[:, start:start + window_samples]
        
        return windows
    
    def compute_TAR_welch_absolute(self, eeg_data, frontal_channel_indices,
        parietal_channel_indices, nperseg=256):
        """
        Computes Theta alpha ratio (scaled Frontal Theta / Parietal Alpha) using
        specified channels for each band, with PSD estimated by Welch's method
        and using absolute powers.

        Parameters:
        - eeg_data (np.ndarray): EEG data of shape (n_channels, n_samples).
        - sfreq (float): Sampling frequency of the EEG data.
        - frontal_channel_indices (list or np.ndarray): Indices of channels to be used for Theta power.
        - parietal_channel_indices (list or np.ndarray): Indices of channels to be used for Alpha power.
        - nperseg (int): Length of each segment for Welch's method. Defaults to 256.
                        Adjust based on sfreq and desired frequency resolution.
                        Should be <= n_samples.

        Returns:
        - float: The computed TAR. Returns np.nan if inputs are invalid
                or if the denominator (alpha power) is zero.
        """
        bands = {
            'Theta': (4, 8),  # Hz
            'Alpha': (8, 12), # Hz
        }
        
        if not frontal_channel_indices or not parietal_channel_indices:
            logger.error("Error: Frontal or parietal channel indices list is empty.")
            return np.nan
        if eeg_data.ndim != 2:
            logger.error("Error: eeg_data must be 2D (n_channels, n_samples).")
            return np.nan

        n_channels, n_samples = eeg_data.shape

        if n_samples < nperseg:
            logger.error(f"Error: n_samples ({n_samples}) must be greater than or equal to nperseg ({nperseg}) for Welch's method.")
            return np.nan

        if max(frontal_channel_indices) >= n_channels or min(frontal_channel_indices) < 0:
            logger.error("Error: Invalid frontal channel index.")
            return np.nan
        if max(parietal_channel_indices) >= n_channels or min(parietal_channel_indices) < 0:
            logger.error("Error: Invalid parietal channel index.")
            return np.nan        

        # Calculate PSD using Welch's method for all channels
        # freqs will be the same for all channels if sfreq and nperseg are constant
        psd_all_channels = []

        if nperseg is None:
            nperseg = int(self.sfreq)
            
        freqs, psd_all_channels = welch(eeg_data, fs=self.sfreq, nperseg=nperseg,
                                    axis=1, scaling='density', window='hann')
        
        psd_all_channels = np.array(psd_all_channels)  # Shape: (n_channels, n_freq_bins)

        # Calculate absolute band powers for Theta and Alpha for each channel
        absolute_band_powers_per_channel = {} 
        df = freqs[1] - freqs[0]

        for band_name, (fmin, fmax) in bands.items():
            if band_name == 'Theta':
                freq_mask = (freqs >= fmin) & (freqs < fmax)
            else:
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                
            power_in_band = np.sum(psd_all_channels[:, freq_mask] * df, axis=1)
            absolute_band_powers_per_channel[band_name] = power_in_band

        # Sum absolute Theta power from specified FRONTAL channels
        frontal_indices_np = np.array(frontal_channel_indices, dtype=int)
        sum_absolute_frontal_theta = np.mean(absolute_band_powers_per_channel["Theta"][frontal_indices_np])

        # Sum absolute Alpha power from specified PARIETAL channels
        parietal_indices_np = np.array(parietal_channel_indices, dtype=int)
        sum_absolute_parietal_alpha = np.mean(absolute_band_powers_per_channel["Alpha"][parietal_indices_np])
            
        TAR = sum_absolute_frontal_theta / sum_absolute_parietal_alpha
            
        return TAR
    
    def compute_gaze_kinematics(self, eye_x, eye_y, window_time=0.1, polyorder=2):
        window_length = int(window_time * self.sfreq)
        window_length = window_length + 1 if window_length % 2 == 0 else window_length
        window_length = max(window_length, polyorder + 3)  # Ensure minimum size
        
        # Smooth coordinates using Savitzky-Golay filter
        x_smooth = savgol_filter(eye_x, window_length, polyorder)
        y_smooth = savgol_filter(eye_y, window_length, polyorder)
        
        # Compute velocity components (1st derivative)
        dx = savgol_filter(x_smooth, window_length, polyorder, deriv=1, delta=1.0/self.sfreq)
        dy = savgol_filter(y_smooth, window_length, polyorder, deriv=1, delta=1.0/self.sfreq)
        
        # Compute velocity magnitude
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Compute acceleration components (2nd derivative)
        ddx = savgol_filter(x_smooth, window_length, polyorder, deriv=2, delta=1.0/self.sfreq)
        ddy = savgol_filter(y_smooth, window_length, polyorder, deriv=2, delta=1.0/self.sfreq)
        
        # Compute acceleration magnitude
        acceleration = np.sqrt(ddx**2 + ddy**2)        
        return np.nanmean(velocity), np.nanmean(acceleration)
    
    def get_metric(self, data, metric_window_duration, metric_shift_duration, theta_channels, alpha_channels):
        windows = self.create_windows(data, metric_window_duration, metric_shift_duration)
        num_windows = windows.shape[1]
        TAR = np.zeros((num_windows))
        for win in range(num_windows):
            TAR[win] = self.compute_TAR_welch_absolute(windows[:, win, :], theta_channels, alpha_channels)
        return TAR
    
    def get_features(self, data, feature_window_duration, feature_shift_duration):        
        windows = self.create_windows(data, feature_window_duration, feature_shift_duration)
        num_windows = windows.shape[1]
        avg_pupil_diameter = np.zeros(num_windows)
        avg_gaze_x_location = np.zeros(num_windows)
        avg_gaze_y_location = np.zeros(num_windows)
        avg_gaze_speed = np.zeros(num_windows)
        avg_gaze_acceleration = np.zeros(num_windows)
        for win in range(num_windows):
            avg_pupil_diameter[win] = np.mean(windows[-1, win, :])
            avg_gaze_x_location[win] = np.mean(windows[-3, win, :])
            avg_gaze_y_location[win] = np.mean(windows[-2, win, :])
            v, a = self.compute_gaze_kinematics(windows[-3, win, :], windows[-2, win, :])
            avg_gaze_speed[win] = v
            avg_gaze_acceleration[win] = a
        
        return avg_pupil_diameter, avg_gaze_speed, avg_gaze_acceleration, avg_gaze_x_location, avg_gaze_y_location

    def get_metric_and_features(self, data, metric_window_duration,
                                metric_shift_duration, theta_channels, alpha_channels,
                                feature_window_duration, feature_shift_duration):
        TAR = self.get_metric(data, metric_window_duration, metric_shift_duration, theta_channels, alpha_channels)
        pupil_diameter, gaze_vel, gaze_acc, gaze_x, gaze_y = self.get_features(data, feature_window_duration, feature_shift_duration)

        pupil_diameter = pupil_diameter[:TAR.shape[0] * self.num_feat_per_metric].reshape((TAR.shape[0]), self.num_feat_per_metric)
        gaze_vel = gaze_vel[:TAR.shape[0] * self.num_feat_per_metric].reshape((TAR.shape[0]), self.num_feat_per_metric)
        gaze_acc = gaze_acc[:TAR.shape[0] * self.num_feat_per_metric].reshape((TAR.shape[0]), self.num_feat_per_metric)
        gaze_x = gaze_x[:TAR.shape[0] * self.num_feat_per_metric].reshape((TAR.shape[0]), self.num_feat_per_metric)
        gaze_y = gaze_y[:TAR.shape[0] * self.num_feat_per_metric].reshape((TAR.shape[0]), self.num_feat_per_metric)
        
        return TAR, pupil_diameter, gaze_vel, gaze_acc, gaze_x, gaze_y

    
    def load_eeg_events(self, mat_file):
        data = read_mat(mat_file)["sEEG"]
        events = data["event"]
        if isinstance(events, list):
            events = events[0]
        types = np.array(events["type"]).flatten()
        latencies = np.array(events["latency"]).flatten().astype(int)
        return data, types, latencies

    def get_stimulus(self, file):
        onset_queue = []
        data, types, latencies = self.load_eeg_events(file)
        stimulus_array = np.zeros(shape=(data["data"].shape[1],))

        for code_str, latency in zip(types[1:], latencies[1:]):
            code_str = code_str.strip()
            if code_str.isdigit():
                code = int(code_str)
                # Onset: 1–27 or 101–127
                if 1 <= code <= 27:
                    onset_queue.append((latency, code))
                elif 101 <= code <= 127:
                    onset_queue.append((latency, code - 100))
                # Offset: 41
                elif code == 41 and onset_queue:
                    onset_latency, dot_id = onset_queue.pop(0)  # FIFO: close oldest open onset
                    start = onset_latency
                    end = latency
                    stimulus_array[start:end] = dot_id
        return np.expand_dims(stimulus_array, axis=0)
    
    def get_stimulus_array(self, file, feature_window_duration, feature_shift_duration, TAR):
        stimulus_array = self.get_stimulus(file=file)
        # add dimension to stimulus_array so you can pass it to create_windows
        windows = self.create_windows(stimulus_array, feature_window_duration, feature_shift_duration)
        num_windows = windows.shape[1]
        stimulus = np.zeros(num_windows)
        for win in range(num_windows):
            values, counts = np.unique(windows[0, win, :], return_counts=True)
            stimulus[win] = values[np.argmax(counts)]
        
        stimulus = stimulus[:TAR.shape[0] * self.num_feat_per_metric].reshape((TAR.shape[0]), self.num_feat_per_metric)
        return stimulus

def create_custom_collate_fn(args):
    def custom_collate_fn(batch):
        pd = []
        gv = []
        ga = []
        gx = []
        gy = []
        t = []
        s = []
        masks = []
        seq_lengths = []

        for TAR, pupil_diameter, gaze_vel, gaze_acc, gaze_x, gaze_y, stimulus in batch:
            length_of_sample = TAR.shape[0]

            # Normalization
            # TAR = (TAR - np.min(TAR)) / (np.max(TAR) - np.min(TAR))
            pupil_diameter = (pupil_diameter - np.min(pupil_diameter)) / (np.max(pupil_diameter) - np.min(pupil_diameter))
            gaze_vel = (gaze_vel - np.min(gaze_vel)) / (np.max(gaze_vel) - np.min(gaze_vel))
            gaze_acc = (gaze_acc - np.min(gaze_acc)) / (np.max(gaze_acc) - np.min(gaze_acc))
            gaze_x = (gaze_x - np.min(gaze_x)) / (np.max(gaze_x) - np.min(gaze_x))
            gaze_y = (gaze_y - np.min(gaze_y)) / (np.max(gaze_y) - np.min(gaze_y))
            stimulus = stimulus / 27

            TAR = np.concatenate((np.array([-1.0]), TAR), axis=0)
            mask = [0] * args.context_window + [1] * length_of_sample
                
            pd.append(np.array(pupil_diameter)[:args.max_len])
            gv.append(np.array(gaze_vel)[:args.max_len])
            ga.append(np.array(gaze_acc)[:args.max_len])
            gx.append(np.array(gaze_x)[:args.max_len])
            gy.append(np.array(gaze_y)[:args.max_len])
            t.append(np.array(TAR)[:args.max_len])
            s.append(np.array(stimulus)[:args.max_len])
            masks.append(np.array(mask)[:args.max_len])
            seq_lengths.append(args.max_len)

        pd = np.stack(pd)
        gv = np.stack(gv)
        ga = np.stack(ga)
        gx = np.stack(gx)
        gy = np.stack(gy)
        t = np.stack(t)
        s = np.stack(s)
        masks = np.stack(masks)

        return {
            'pupil': torch.from_numpy(pd).float(),
            'gaze_vel': torch.from_numpy(gv).float(),
            'gaze_acc': torch.from_numpy(ga).float(),
            'gaze_x': torch.from_numpy(gx).float(),
            'gaze_y': torch.from_numpy(gy).float(),
            'stimulus': torch.from_numpy(s).float(),
            'labels': torch.from_numpy(t).float(),
            'loss_masks': torch.from_numpy(masks).bool(),
        }

    return custom_collate_fn


def create_custom_collate_fn_feat_norm(args):
    def custom_collate_fn(batch):
        pd = []
        gv = []
        ga = []
        gx = []
        gy = []
        t = []
        s = []
        masks = []
        seq_lengths = []

        for TAR, pupil_diameter, gaze_vel, gaze_acc, gaze_x, gaze_y, stimulus in batch:
            length_of_sample = TAR.shape[0]

            # Gaussian normalization for features (per sample)
            def gaussian_normalize(arr):
                mean = np.mean(arr)
                std = np.std(arr)
                if std == 0:
                    std = 1e-8  # Small epsilon for stability
                return (arr - mean) / std

            pupil_diameter = gaussian_normalize(pupil_diameter)
            gaze_vel = gaussian_normalize(gaze_vel)
            gaze_acc = gaussian_normalize(gaze_acc)
            gaze_x = gaussian_normalize(gaze_x)
            gaze_y = gaussian_normalize(gaze_y)
            stimulus = stimulus / 27  # Keeping your original scaling for stimulus

            TAR = np.concatenate((np.array([-1.0]), TAR), axis=0)  # Your original padding
            mask = [0] * args.context_window + [1] * length_of_sample

            pd.append(np.array(pupil_diameter)[:args.max_len])
            gv.append(np.array(gaze_vel)[:args.max_len])
            ga.append(np.array(gaze_acc)[:args.max_len])
            gx.append(np.array(gaze_x)[:args.max_len])
            gy.append(np.array(gaze_y)[:args.max_len])
            t.append(np.array(TAR)[:args.max_len])
            s.append(np.array(stimulus)[:args.max_len])
            masks.append(np.array(mask)[:args.max_len])
            seq_lengths.append(args.max_len)

        pd = np.stack(pd)
        gv = np.stack(gv)
        ga = np.stack(ga)
        gx = np.stack(gx)
        gy = np.stack(gy)
        t = np.stack(t)
        s = np.stack(s)
        masks = np.stack(masks)

        return {
            'pupil': torch.from_numpy(pd).float(),
            'gaze_vel': torch.from_numpy(gv).float(),
            'gaze_acc': torch.from_numpy(ga).float(),
            'gaze_x': torch.from_numpy(gx).float(),
            'gaze_y': torch.from_numpy(gy).float(),
            'stimulus': torch.from_numpy(s).float(),
            'labels': torch.from_numpy(t).float(),
            'loss_masks': torch.from_numpy(masks).bool(),
        }

    return custom_collate_fn



def create_custom_collate_fn_normalized(args):
    def custom_collate_fn(batch):
        pd = []
        gv = []
        ga = []
        gx = []
        gy = []
        t = []
        s = []
        masks = []
        seq_lengths = []

        for TAR, pupil_diameter, gaze_vel, gaze_acc, gaze_x, gaze_y, stimulus in batch:
            length_of_sample = TAR.shape[0]

            # Gaussian normalization for features (per sample)
            def gaussian_normalize(arr):
                mean = np.mean(arr)
                std = np.std(arr)
                if std == 0:
                    std = 1e-8  # Small epsilon for stability
                return (arr - mean) / std

            pupil_diameter = gaussian_normalize(pupil_diameter)
            gaze_vel = gaussian_normalize(gaze_vel)
            gaze_acc = gaussian_normalize(gaze_acc)
            gaze_x = gaussian_normalize(gaze_x)
            gaze_y = gaussian_normalize(gaze_y)
            stimulus = stimulus / 27  # Keeping original scaling for stimulus

            # Log normalization for target (TAR)
            TAR = np.log1p(TAR)  # Assumes TAR >= 0; adjust if needed

            TAR = np.concatenate((np.array([-1.0]), TAR), axis=0)  # original padding
            mask = [0] * args.context_window + [1] * length_of_sample

            pd.append(np.array(pupil_diameter)[:args.max_len])
            gv.append(np.array(gaze_vel)[:args.max_len])
            ga.append(np.array(gaze_acc)[:args.max_len])
            gx.append(np.array(gaze_x)[:args.max_len])
            gy.append(np.array(gaze_y)[:args.max_len])
            t.append(np.array(TAR)[:args.max_len])
            s.append(np.array(stimulus)[:args.max_len])
            masks.append(np.array(mask)[:args.max_len])
            seq_lengths.append(args.max_len)

        pd = np.stack(pd)
        gv = np.stack(gv)
        ga = np.stack(ga)
        gx = np.stack(gx)
        gy = np.stack(gy)
        t = np.stack(t)
        s = np.stack(s)
        masks = np.stack(masks)

        return {
            'pupil': torch.from_numpy(pd).float(),
            'gaze_vel': torch.from_numpy(gv).float(),
            'gaze_acc': torch.from_numpy(ga).float(),
            'gaze_x': torch.from_numpy(gx).float(),
            'gaze_y': torch.from_numpy(gy).float(),
            'stimulus': torch.from_numpy(s).float(),
            'labels': torch.from_numpy(t).float(),
            'loss_masks': torch.from_numpy(masks).bool(),
        }

    return custom_collate_fn