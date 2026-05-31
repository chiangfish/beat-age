"""
Visualize averaged Beat-age signals and input-gradient saliency maps for three predicted age groups.
Uses PREDICTED AGE (not true age) to group subjects based on extreme age_gap values.
Filters beats by length (68% CI) to avoid incorrectly segmented ECG beats.
"""
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import neurokit2 as nk
import pickle

from matplotlib import font_manager, rcParams
regular_path = "Helvetica-01.ttf"
bold_path = "Helvetica-Bold-02.ttf"
font_manager.fontManager.addfont(regular_path)
font_manager.fontManager.addfont(bold_path)
rcParams["font.family"] = "Helvetica"
rcParams["axes.unicode_minus"] = False

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from beat_age.models import Net1D

# Paths
PROCESSED_BEATS_DIR = "datasets/processed_beats_clinical"
SPLITS_CSV = "datasets/splits.csv"
PREDICTIONS_CSV = "results/clinical_predictions.csv"
MODEL_PATH = "ckpts/v1_best.pth"
OUTPUT_DIR = "results"
CACHE_FILE = "results/saliency_input_gradient_cache.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def gaussian_kernel_1d(sigma, device, dtype):
    """Create a normalized 1D Gaussian kernel."""
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def gaussian_smooth_per_lead(x, sigma=2.0):
    """
    Apply Gaussian smoothing to each ECG lead independently.

    Args:
        x: (1, C, L) tensor
        sigma: Gaussian sigma in samples
    """
    kernel = gaussian_kernel_1d(sigma=sigma, device=x.device, dtype=x.dtype)
    c = x.shape[1]
    kernel = kernel.repeat(c, 1, 1)
    pad = kernel.shape[-1] // 2
    return F.conv1d(x, kernel, padding=pad, groups=c)


def load_model(device):
    """Load trained model"""
    model = Net1D(
        in_channels=12, base_filters=24, ratio=1.0,
        filter_list=[24, 48, 96, 192], m_blocks_list=[2, 2, 2, 2],
        kernel_size=13, stride=1, groups_width=12,
        verbose=False, n_classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def collect_beats_with_predicted_ages(predictions_csv, processed_dir):
    """
    Collect all beats with their predicted ages and age_gaps from clinical predictions.

    Returns:
        all_beats: list of tensors
        all_pred_ages: list of predicted ages
        all_age_gaps: list of age gaps
        all_lengths: list of beat lengths
    """
    # Load predictions (clinical cohort)
    pred_df = pd.read_csv(predictions_csv)
    print(f"Loaded {len(pred_df)} predictions from clinical cohort")

    all_beats = []
    all_pred_ages = []
    all_age_gaps = []
    all_lengths = []

    print("Loading beats from processed files...")
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        fname = row['FileName']
        pred_age = row['pred_age']
        age_gap = row['age_gap']
        pt_path = os.path.join(processed_dir, f"{fname}.pt")

        if not os.path.exists(pt_path):
            continue

        try:
            data = torch.load(pt_path, map_location='cpu')
            beats = data['beats']  # List of tensors (12, L)

            for beat in beats:
                all_beats.append(beat)
                all_pred_ages.append(pred_age)
                all_age_gaps.append(age_gap)
                all_lengths.append(beat.shape[1])  # L dimension
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            continue

    return all_beats, all_pred_ages, all_age_gaps, all_lengths


def filter_by_length_ci(beats, pred_ages, age_gaps, lengths, ci=0.68):
    """Filter beats to keep only those within 68% CI of length distribution"""
    lengths_array = np.array(lengths)
    mean_len = np.mean(lengths_array)
    std_len = np.std(lengths_array)

    # 68% CI corresponds to ±1 standard deviation
    z_score = 1.0  # For 68% CI
    lower_bound = mean_len - z_score * std_len
    upper_bound = mean_len + z_score * std_len

    print(f"Length statistics: mean={mean_len:.2f}, std={std_len:.2f}")
    print(f"68% CI range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Filter beats
    filtered_beats = []
    filtered_pred_ages = []
    filtered_age_gaps = []
    for beat, pred_age, age_gap, length in zip(beats, pred_ages, age_gaps, lengths):
        if lower_bound <= length <= upper_bound:
            filtered_beats.append(beat)
            filtered_pred_ages.append(pred_age)
            filtered_age_gaps.append(age_gap)

    print(f"Filtered: {len(filtered_beats)}/{len(beats)} beats retained")
    return filtered_beats, filtered_pred_ages, filtered_age_gaps


def select_extreme_age_gap_groups(beats, pred_ages, age_gaps, sigma_threshold=2.0):
    """
    Select three groups based on extreme age_gap values.

    Note: age_gap = predicted_age - true_age (standard definition)
    - Positive age_gap: pred_age > true_age (Overestimate - predicts older)
    - Negative age_gap: pred_age < true_age (Underestimate - predicts younger)

    Returns:
        groups: dict with keys 'underestimate', 'correct', 'overestimate'
                each containing {'beats': list, 'pred_ages': list, 'age_gaps': list}
    """
    age_gaps_array = np.array(age_gaps)
    mean_gap = np.mean(age_gaps_array)
    std_gap = np.std(age_gaps_array)

    print(f"\nAge gap statistics:")
    print(f"  Mean: {mean_gap:.2f}")
    print(f"  Std: {std_gap:.2f}")
    print(f"  {sigma_threshold}-sigma thresholds: [{mean_gap - sigma_threshold*std_gap:.2f}, {mean_gap + sigma_threshold*std_gap:.2f}]")

    # Define thresholds based on 2-sigma
    underestimate_threshold = mean_gap - sigma_threshold * std_gap  # Negative (pred < true)
    overestimate_threshold = mean_gap + sigma_threshold * std_gap   # Positive (pred > true)
    medium_threshold = 0.5 * std_gap

    groups = {
        'underestimate': {'beats': [], 'pred_ages': [], 'age_gaps': []},  # age_gap < 0
        'correct': {'beats': [], 'pred_ages': [], 'age_gaps': []},        # age_gap ≈ 0
        'overestimate': {'beats': [], 'pred_ages': [], 'age_gaps': []}    # age_gap > 0
    }

    for beat, pred_age, age_gap in zip(beats, pred_ages, age_gaps):
        if age_gap < underestimate_threshold:
            # Negative age_gap: pred_age < true_age (underestimate)
            groups['underestimate']['beats'].append(beat)
            groups['underestimate']['pred_ages'].append(pred_age)
            groups['underestimate']['age_gaps'].append(age_gap)
        elif age_gap > overestimate_threshold:
            # Positive age_gap: pred_age > true_age (overestimate)
            groups['overestimate']['beats'].append(beat)
            groups['overestimate']['pred_ages'].append(pred_age)
            groups['overestimate']['age_gaps'].append(age_gap)
        elif abs(age_gap - mean_gap) < medium_threshold:
            # Close to mean: correct prediction
            groups['correct']['beats'].append(beat)
            groups['correct']['pred_ages'].append(pred_age)
            groups['correct']['age_gaps'].append(age_gap)

    print(f"\nGroup sizes:")
    print(f"  Underestimate (age_gap < {underestimate_threshold:.2f}): {len(groups['underestimate']['beats'])} beats")
    print(f"  Correct (|age_gap| < {medium_threshold:.2f}): {len(groups['correct']['beats'])} beats")
    print(f"  Overestimate (age_gap > {overestimate_threshold:.2f}): {len(groups['overestimate']['beats'])} beats")

    # Print average statistics for each group
    for group_name in ['underestimate', 'correct', 'overestimate']:
        if len(groups[group_name]['pred_ages']) > 0:
            avg_pred_age = np.mean(groups[group_name]['pred_ages'])
            avg_age_gap = np.mean(groups[group_name]['age_gaps'])
            print(f"  {group_name.capitalize()}: avg pred_age={avg_pred_age:.1f}, avg age_gap={avg_age_gap:.1f}")

    return groups


def detect_r_peak(beat, fs=360):
    """
    Detect R peak position in a single beat using NeuroKit2.

    Args:
        beat: (12, L) tensor, single beat ECG
        fs: sampling rate

    Returns:
        r_peak_idx: int, index of R peak in the beat
    """
    # Use lead II (index 1) for R peak detection as it typically has clear QRS
    lead_signal = beat[1].numpy() if isinstance(beat, torch.Tensor) else beat[1]

    try:
        # Use NeuroKit2 to detect R peaks
        signals, info = nk.ecg_process(lead_signal, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]

        if len(rpeaks) > 0:
            # Find the R peak closest to the center of the beat
            center = len(lead_signal) // 2
            r_peak_idx = min(rpeaks, key=lambda x: abs(x - center))
            return r_peak_idx
        else:
            # If no R peak detected, return center as fallback
            return len(lead_signal) // 2
    except:
        # If detection fails, return center as fallback
        return len(lead_signal) // 2


def align_beats_by_r_peak(beats, target_length=None):
    """
    Align all beats by their R peaks to the center.

    Args:
        beats: list of (12, L_i) tensors
        target_length: desired output length (if None, use median length)

    Returns:
        aligned_beats: (N, 12, target_length) tensor
    """
    if target_length is None:
        # Use median length as target
        lengths = [beat.shape[1] for beat in beats]
        target_length = int(np.median(lengths))

    center = target_length // 2
    aligned_beats = []

    print(f"  Aligning {len(beats)} beats by R peak to center position {center}...")

    for beat in tqdm(beats, desc="  R-peak alignment", leave=False):
        # Detect R peak position in this beat
        r_peak_idx = detect_r_peak(beat)

        # Calculate shift needed to align R peak to center
        shift = center - r_peak_idx

        # Create aligned beat with zero padding
        aligned_beat = torch.zeros(12, target_length)

        # Calculate source and target ranges
        if shift >= 0:
            # R peak is left of center, shift right
            src_start = 0
            src_end = min(beat.shape[1], target_length - shift)
            tgt_start = shift
            tgt_end = shift + (src_end - src_start)
        else:
            # R peak is right of center, shift left
            src_start = -shift
            src_end = min(beat.shape[1], target_length - shift)
            tgt_start = 0
            tgt_end = src_end - src_start

        # Copy the beat data to aligned position
        if src_end > src_start and tgt_end > tgt_start:
            aligned_beat[:, tgt_start:tgt_end] = beat[:, src_start:src_end]

        aligned_beats.append(aligned_beat)

    return torch.stack(aligned_beats)  # (N, 12, target_length)


def compute_averaged_signal(beats):
    """Compute averaged signal across all beats with R-peak alignment"""
    # Align all beats by R peak
    beats_tensor = align_beats_by_r_peak(beats)

    # Compute mean across all beats
    return beats_tensor.mean(dim=0)  # (12, target_length)


def compute_input_gradient_saliency(model, signal, device):
    """
    Compute input-gradient saliency map with Gaussian smoothing and interpolation.

    Args:
        model: trained model
        signal: (12, L) tensor
        device: torch device

    Returns:
        saliency: (12, L) numpy array in [0, 1]
    """
    model.eval()
    x = signal.unsqueeze(0).to(device).clone().detach().requires_grad_(True)  # (1, 12, L)

    output = model(x).squeeze()
    model.zero_grad()
    output.backward()

    # Raw input gradients (absolute value for attribution strength)
    grads = x.grad.detach().abs()  # (1, 12, L)

    # Gaussian smoothing to reduce high-frequency noise
    grads_smoothed = gaussian_smooth_per_lead(grads, sigma=2.0)

    # Interpolate saliency to exactly match input length
    grads_interp = F.interpolate(
        grads_smoothed, size=signal.shape[1], mode='linear', align_corners=False
    )

    saliency = grads_interp.squeeze(0).cpu().numpy()  # (12, L)

    # Per-lead normalization to [0, 1]
    saliency_min = saliency.min(axis=1, keepdims=True)
    saliency_max = saliency.max(axis=1, keepdims=True)
    denom = np.maximum(saliency_max - saliency_min, 1e-8)
    saliency = (saliency - saliency_min) / denom

    return saliency


def visualize_signals_with_saliency(group_labels, signals, saliency_maps, output_path):
    """
    Create visualization with 3 columns (predicted age groups) and 12 rows (leads).
    Each subplot overlays ECG signal with saliency heatmap.

    Args:
        group_labels: list of group labels ['Low Pred Age', 'Medium Pred Age', 'High Pred Age']
        signals: list of (12, L) numpy arrays
        saliency_maps: list of (12, L) numpy arrays
        output_path: path to save figure
    """
    lead_indices = list(range(12))
    lead_names = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(12, 3, figsize=(16, 28))

    for col_idx, (label, signal, saliency) in enumerate(zip(group_labels, signals, saliency_maps)):
        time_axis = np.arange(signal.shape[1]) / 360  # Convert to seconds (360 Hz)

        for row_idx, (lead_idx, lead_name) in enumerate(zip(lead_indices, lead_names)):
            ax = axes[row_idx, col_idx]

            # Plot saliency as background heatmap
            saliency_2d = saliency[lead_idx].reshape(1, -1)  # (1, L) for imshow
            extent = [time_axis[0], time_axis[-1], -110, 110]
            im = ax.imshow(saliency_2d, aspect='auto', cmap='hot', alpha=0.4,
                          extent=extent, interpolation='bilinear', origin='lower')

            # Plot ECG signal on top
            ax.plot(time_axis, signal[lead_idx], 'b-', linewidth=1.2, alpha=0.9)

            # Set title for top row
            if row_idx == 0:
                ax.set_title(label, fontsize=14, fontweight='bold')

            # Set ylabel for first column
            if col_idx == 0:
                ax.set_ylabel(f'{lead_name}\n(mV)', fontsize=11, fontweight='bold')

            # Set xlabel for bottom row
            if row_idx == 11:
                ax.set_xlabel('Time (s)', fontsize=10)

            # Grid and styling
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(time_axis[0], time_axis[-1])
            ax.set_ylim(-100, 100)

            # Remove x-tick labels for non-bottom rows
            if row_idx < 11:
                ax.set_xticklabels([])

            # Add colorbar for the last column of each row
            if col_idx == 2:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Input Gradient Saliency', fontsize=9)
                cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def visualize_overlaid_signals(group_labels, signals, saliency_maps, output_path):
    """
    Create overlay visualization with 12 leads in 4x3 layout.
    All three groups are overlaid in each subplot with different colors.
    Saliency is shown as background heatmap (averaged across three groups).

    Args:
        group_labels: list of group labels ['Low Pred Age', 'Medium Pred Age', 'High Pred Age']
        signals: list of (12, L) numpy arrays
        saliency_maps: list of (12, L) numpy arrays
        output_path: path to save figure
    """
    lead_indices = list(range(12))
    lead_names = ['Lead I', 'Lead II', 'Lead III', 'Lead aVR', 'Lead aVL', 'Lead aVF', 'Lead V1', 'Lead V2', 'Lead V3', 'Lead V4', 'Lead V5', 'Lead V6']

    # Colors for three groups
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    short_labels = ['Underestimate', 'Correct', 'Overestimate']

    # 4x3 layout for 12 leads
    fig, axes = plt.subplots(4, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Find max length and resample all signals to same length
    max_len = max(signal.shape[1] for signal in signals)

    # Resample all signals and saliency maps to max_len for proper averaging
    signals_resampled = []
    saliency_resampled = []
    for signal, saliency in zip(signals, saliency_maps):
        if signal.shape[1] != max_len:
            # Resample using interpolation
            signal_tensor = torch.from_numpy(signal).unsqueeze(0)  # (1, 12, L)
            signal_resampled = F.interpolate(signal_tensor, size=max_len, mode='linear', align_corners=False)
            signals_resampled.append(signal_resampled.squeeze(0).numpy())

            saliency_tensor = torch.from_numpy(saliency).unsqueeze(0)  # (1, 12, L)
            saliency_interp = F.interpolate(saliency_tensor, size=max_len, mode='linear', align_corners=False)
            saliency_resampled.append(saliency_interp.squeeze(0).numpy())
        else:
            signals_resampled.append(signal)
            saliency_resampled.append(saliency)

    # Average saliency across three groups for background
    saliency_avg = np.mean(saliency_resampled, axis=0)  # (12, max_len)

    for row_idx, (lead_idx, lead_name) in enumerate(zip(lead_indices, lead_names)):
        ax = axes[row_idx]

        # Time axis for max_len
        time_axis = np.arange(max_len) / 360  # Convert to seconds (360 Hz)

        # Plot averaged saliency as background heatmap
        saliency_2d = saliency_avg[lead_idx].reshape(1, -1)  # (1, L) for imshow
        extent = [time_axis[0], time_axis[-1], -110, 110]
        im = ax.imshow(saliency_2d, aspect='auto', cmap='hot', alpha=0.4,
                      extent=extent, interpolation='bilinear', origin='lower')

        # Plot all three groups' ECG signals overlaid
        for group_idx, (label, signal, color) in enumerate(zip(short_labels, signals_resampled, colors)):
            ax.plot(time_axis, signal[lead_idx], color=color, linewidth=1.5,
                   alpha=0.85, label=label)

        # Set ylabel
        ax.set_ylabel(f'{lead_name} (mV)', fontweight='bold')

        # Grid and styling
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(-110, 110)

        # Add legend only for first subplot
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=False)

        # Set xlabel for bottom row
        if row_idx >= 9:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])

        # Add colorbar to the right of the last subplot
        # if row_idx == 11:
        #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.ax.tick_params()

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path[:-4] + '.pdf', bbox_inches='tight')
    print(f"Saved overlay visualization to {output_path}")
    plt.close()


def save_cache(data, cache_file):
    """Save processed data to cache file"""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cache to {cache_file}")


def load_cache(cache_file):
    """Load processed data from cache file"""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded cache from {cache_file}")
        return data
    return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Try to load from cache
    cache_data = load_cache(CACHE_FILE)

    if cache_data is not None:
        print("\n=== Using cached data ===")
        signals = cache_data['signals']
        cams = cache_data['cams']
        group_labels = cache_data['group_labels']
        print(f"Loaded {len(signals)} groups from cache")
    else:
        print("\n=== Computing from scratch ===")

        # Load model
        print("Loading model...")
        model = load_model(device)

        # Collect all beats with predicted ages and age_gaps
        all_beats, all_pred_ages, all_age_gaps, all_lengths = collect_beats_with_predicted_ages(
            PREDICTIONS_CSV, PROCESSED_BEATS_DIR
        )
        print(f"Total beats collected: {len(all_beats)}")

        # Filter by length (68% CI)
        filtered_beats, filtered_pred_ages, filtered_age_gaps = filter_by_length_ci(
            all_beats, all_pred_ages, all_age_gaps, all_lengths, ci=0.68
        )

        # Select extreme age gap groups (2-sigma threshold)
        groups = select_extreme_age_gap_groups(
            filtered_beats, filtered_pred_ages, filtered_age_gaps, sigma_threshold=2.0
        )

        # Process three groups: underestimate, correct, overestimate
        group_names = ['underestimate', 'correct', 'overestimate']
        group_labels = [
            'Underestimate\n(Age gap < -10.39)',
            'Correct',
            'Overestimate\n(Age gap > 15.67)'
        ]
        signals = []
        cams = []

        for group_name, label in zip(group_names, group_labels):
            print(f"\nProcessing group: {label}")

            group_beats = groups[group_name]['beats']
            if len(group_beats) == 0:
                print(f"  Warning: No beats found for {group_name} group")
                continue

            # Compute averaged signal
            avg_signal = compute_averaged_signal(group_beats)
            print(f"  Averaged signal shape: {avg_signal.shape}")

            # Compute input-gradient saliency
            cam = compute_input_gradient_saliency(model, avg_signal, device)
            print(f"  Saliency shape: {cam.shape}")

            signals.append(avg_signal.numpy())
            cams.append(cam)

        # Save to cache
        if len(signals) == 3:
            cache_data = {
                'signals': signals,
                'cams': cams,
                'group_labels': group_labels
            }
            save_cache(cache_data, CACHE_FILE)

    # Generate visualizations
    if len(signals) == 3:
        # # Visualization 1: 12 leads × 3 groups (original)
        # output_path1 = os.path.join(OUTPUT_DIR, "fig_age_gradcam_overlay.png")
        # visualize_signals_with_saliency(group_labels, signals, cams, output_path1)

        # Visualization 2: 12 leads × 1 (overlaid in 4x3 layout)
        output_path2 = os.path.join(OUTPUT_DIR, "formal_fig7b_age_gradcam_overlaid.png")
        visualize_overlaid_signals(group_labels, signals, cams, output_path2)

        print(f"\n=== Visualization complete! ===")
        print(f"Generated 2 figures:")
        # print(f"  1. {output_path1}")
        print(f"  2. {output_path2}")
    else:
        print(f"\nError: Expected 3 groups, got {len(signals)}")


if __name__ == "__main__":
    main()
