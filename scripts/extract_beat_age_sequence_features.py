"""
Part III: Serial ECG Analysis
Analyzes beat-level age gap sequences for subjects with multiple ECG instances.
"""
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from beat_age.models import Net1D

from matplotlib import font_manager, rcParams
regular_path = "Helvetica-01.ttf"
bold_path = "Helvetica-Bold-02.ttf"
font_manager.fontManager.addfont(regular_path)
font_manager.fontManager.addfont(bold_path)
rcParams["font.family"] = "Helvetica"
rcParams["axes.unicode_minus"] = False

# Paths
CLINICAL_BEATS_DIR = "datasets/processed_beats_clinical"
MASTER_CSV = "datasets/raw_data/dataset.csv"
HEALTH_CSV = "datasets/proc_data/health_population_ecg_metadata.csv"
MODEL_PATH = "ckpts/v1_best.pth"
SURVIVAL_DATA = "results/survival_data.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

class ClinicalDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname, age = self.file_list[idx]
        pt_path = os.path.join(CLINICAL_BEATS_DIR, f"{fname}.pt")
        data = torch.load(pt_path, map_location='cpu')
        beats = data['beats']
        return beats, torch.tensor([age] * len(beats), dtype=torch.float32), fname


def collate_fn(batch):
    all_beats, all_labels, all_fnames = [], [], []
    for beats, labels, fname in batch:
        all_beats.extend(beats)
        all_labels.extend(labels)
        all_fnames.extend([fname] * len(beats))

    if not all_beats:
        return torch.tensor([]), torch.tensor([]), []

    beats_t = [b.permute(1, 0) for b in all_beats]
    padded = rnn_utils.pad_sequence(beats_t, batch_first=True, padding_value=0.0)
    x = padded.permute(0, 2, 1)
    y = torch.stack(all_labels) if isinstance(all_labels[0], torch.Tensor) else torch.tensor(all_labels, dtype=torch.float32)
    return x, y, all_fnames


def generate_beat_level_predictions():
    """Generate beat-level predictions for all clinical cohort subjects."""
    print("=" * 80)
    print("Step 1: Generating beat-level predictions")
    print("=" * 80)

    # Load clinical cohort metadata
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    health = pd.read_csv(HEALTH_CSV, low_memory=False)
    health_files = set(health['FileName'].tolist())

    clinical = master[~master['FileName'].isin(health_files)].copy()
    clinical = clinical.dropna(subset=['RecordAge'])

    # Filter to files that have processed beats
    clinical_pt_files = set(f.replace('.pt','') for f in os.listdir(CLINICAL_BEATS_DIR))
    clinical = clinical[clinical['FileName'].isin(clinical_pt_files)]

    print(f"Clinical cohort: {len(clinical)} subjects")

    file_list = [(row['FileName'], row['RecordAge']) for _, row in clinical.iterrows()]

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net1D(
        in_channels=12, base_filters=24, ratio=1.0,
        filter_list=[24, 48, 96, 192], m_blocks_list=[2, 2, 2, 2],
        kernel_size=13, stride=1, groups_width=12,
        verbose=False, n_classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference - save beat-level predictions
    dataset = ClinicalDataset(file_list)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    beat_results = []
    with torch.no_grad():
        for x, y, fnames in tqdm(loader, desc="Inference"):
            if len(x) == 0:
                continue
            x = x.to(device)
            preds = model(x).squeeze(1).cpu().numpy()
            ages = y.numpy()

            # Save each beat prediction
            for i, fname in enumerate(fnames):
                beat_results.append({
                    'FileName': fname,
                    'beat_idx': i,
                    'true_age': ages[i],
                    'pred_age': preds[i],
                    'age_gap': ages[i] - preds[i]  # age_gap = true_age - pred_age
                })

    # Save beat-level predictions
    df_beats = pd.DataFrame(beat_results)

    # Add eid
    df_beats = df_beats.merge(clinical[['FileName', 'eid']], on='FileName', how='left')

    # Reorder beat_idx within each FileName
    df_beats = df_beats.sort_values(['FileName', 'beat_idx']).reset_index(drop=True)
    df_beats['beat_idx'] = df_beats.groupby('FileName').cumcount()

    output_path = os.path.join(RESULTS_DIR, 'serial_1_beat_level_predictions.csv')
    df_beats.to_csv(output_path, index=False)
    print(f"Saved beat-level predictions to {output_path}")
    print(f"Total beats: {len(df_beats)}")

    return df_beats


def identify_serial_ecg_subjects(df_beats):
    """Identify subjects with multiple ECG instances."""
    print("\n" + "=" * 80)
    print("Step 2: Identifying serial ECG subjects")
    print("=" * 80)

    # Extract instance from FileName
    df_beats['instance'] = df_beats['FileName'].str.split('_').str[2].astype(int)

    # Count instances per eid
    instance_counts = df_beats.groupby('eid')['instance'].nunique()
    serial_eids = instance_counts[instance_counts > 1].index.tolist()

    print(f"Total subjects with multiple instances: {len(serial_eids)}")
    print(f"Max instances per subject: {instance_counts.max()}")

    # Filter to serial ECG subjects
    df_serial = df_beats[df_beats['eid'].isin(serial_eids)].copy()

    # Sort by eid, instance, beat_idx
    df_serial = df_serial.sort_values(['eid', 'instance', 'beat_idx']).reset_index(drop=True)

    output_path = os.path.join(RESULTS_DIR, 'serial_2_ecg_subjects.csv')
    df_serial.to_csv(output_path, index=False)
    print(f"Saved serial ECG data to {output_path}")

    return df_serial, serial_eids


def extract_single_segment_features(df_beats):
    """Extract features from single 10s ECG segments (per FileName)."""
    print("\n" + "=" * 80)
    print("Step 3: Extracting single segment features")
    print("=" * 80)

    features_list = []

    for fname, group in tqdm(df_beats.groupby('FileName'), desc="Extracting features"):
        age_gaps = group['age_gap'].values

        # Calculate age gap deltas (successive differences)
        age_gap_deltas = np.diff(age_gaps)

        # Extract features
        features = {
            'FileName': fname,
            'eid': group['eid'].iloc[0],
            'n_beats': len(age_gaps),
            'mean': np.mean(age_gaps),
            'variance': np.var(age_gaps),
            'min': np.min(age_gaps),
            'max': np.max(age_gaps),
            'range': np.max(age_gaps) - np.min(age_gaps),
            'skewness': stats.skew(age_gaps),
            'kurtosis': stats.kurtosis(age_gaps),
        }

        # RMSSD: Root Mean Square of Successive Differences
        if len(age_gap_deltas) > 0:
            features['rmssd'] = np.sqrt(np.mean(age_gap_deltas ** 2))
        else:
            features['rmssd'] = 0.0

        features_list.append(features)

    df_features = pd.DataFrame(features_list)

    output_path = os.path.join(RESULTS_DIR, 'serial_3_single_segment_features.csv')
    df_features.to_csv(output_path, index=False)
    print(f"Saved single segment features to {output_path}")

    return df_features


def merge_with_macce_outcome(df_features):
    """Merge features with MACCE outcome data."""
    print("\n" + "=" * 80)
    print("Step 4: Merging with MACCE outcome data")
    print("=" * 80)

    # Load survival data
    survival = pd.read_csv(SURVIVAL_DATA, low_memory=False)

    # Keep only MACCE-related columns
    survival_cols = ['FileName', 'eid', 'macce_event', 'macce_time', 'macce_exclude']
    survival_subset = survival[survival_cols].copy()

    # Merge
    df_merged = df_features.merge(survival_subset, on='FileName', how='left')

    # Filter out excluded subjects
    df_merged = df_merged[df_merged['macce_exclude'] == 0].copy()

    print(f"Total subjects after filtering: {len(df_merged)}")
    print(f"MACCE events: {df_merged['macce_event'].sum()}")
    print(f"MACCE controls: {(df_merged['macce_event'] == 0).sum()}")

    return df_merged


def plot_feature_histograms(df_features):
    """Plot histograms of all features."""
    print("\n" + "=" * 80)
    print("Step 5: Plotting feature histograms")
    print("=" * 80)

    feature_cols = ['mean', 'variance', 'min', 'max', 'range', 'rmssd', 'skewness', 'kurtosis']

    fig, axes = plt.subplots(8, 1, figsize=(8, 20))

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df_features[col].dropna()

        # Remove outliers using IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_filtered = data[(data >= lower_bound) & (data <= upper_bound)]

        ax.hist(data_filtered, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col} Distribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'serial_4_特征分布直方图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {output_path}")


def plot_correlation_heatmap(df_features):
    """Plot correlation heatmap between features."""
    print("\n" + "=" * 80)
    print("Step 6: Plotting correlation heatmap")
    print("=" * 80)

    feature_cols = ['mean', 'variance', 'min', 'max', 'range', 'rmssd', 'skewness', 'kurtosis']
    df_feat = df_features[feature_cols].dropna()

    # Calculate Pearson and Spearman correlations
    pearson_corr = df_feat.corr(method='pearson')
    spearman_corr = df_feat.corr(method='spearman')

    # Plot both
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pearson
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[0], square=True, cbar_kws={'shrink': 0.8})
    axes[0].set_title('Pearson Correlation', fontsize=14)

    # Spearman
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[1], square=True, cbar_kws={'shrink': 0.8})
    axes[1].set_title('Spearman Correlation', fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'serial_5_特征相关性热力图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")

    # Print high correlations
    print("\n高度相关的特征对 (|r| > 0.7):")
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            r = pearson_corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {feature_cols[i]} vs {feature_cols[j]}: r = {r:.3f}")


def plot_boxplots_by_macce(df_merged):
    """Plot boxplots of features by MACCE groups."""
    print("\n" + "=" * 80)
    print("Step 7: Plotting boxplots by MACCE groups")
    print("=" * 80)

    from lifelines import CoxPHFitter

    feature_cols = ['mean', 'variance', 'min', 'max', 'range', 'rmssd', 'skewness', 'kurtosis']

    fig, axes = plt.subplots(8, 1, figsize=(8, 20))

    pvalues = []
    hr_results = []

    for i, col in enumerate(feature_cols):
        ax = axes[i]

        # Prepare data
        case_data = df_merged[df_merged['macce_event'] == 1][col].dropna()
        control_data = df_merged[df_merged['macce_event'] == 0][col].dropna()

        # Plot boxplot without outliers (showfliers=False)
        data_to_plot = [control_data, case_data]
        bp = ax.boxplot(data_to_plot, labels=['Control', 'Case'], patch_artist=True, showfliers=False)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(col)
        ax.set_title(f'{col}')
        ax.grid(True, alpha=0.3, axis='y')

        # Calculate p-value (Mann-Whitney U test)
        if len(case_data) > 0 and len(control_data) > 0:
            stat, pval = stats.mannwhitneyu(case_data, control_data, alternative='two-sided')
            pvalues.append({'feature': col, 'p_value': pval})

            # Calculate HR using Cox regression
            # Prepare data for Cox regression
            cox_data = df_merged[[col, 'macce_event', 'macce_time']].dropna()
            if len(cox_data) > 10 and cox_data[col].std() > 0:
                try:
                    cph = CoxPHFitter()
                    cph.fit(cox_data, duration_col='macce_time', event_col='macce_event')
                    hr = cph.hazard_ratios_[col]
                    ci_lower = np.exp(cph.confidence_intervals_.loc[col, '95% lower-bound'])
                    ci_upper = np.exp(cph.confidence_intervals_.loc[col, '95% upper-bound'])
                    hr_results.append({'feature': col, 'HR': hr, 'CI_lower': ci_lower, 'CI_upper': ci_upper})

                    # Add HR and p-value to plot
                    hr_text = f'HR: {hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f})'
                    if pval < 0.001:
                        pval_text = 'p < 0.001'
                    else:
                        pval_text = f'p = {pval:.3f}'

                    text_str = f'{hr_text}\n{pval_text}'
                    ax.text(0.5, 0.97, text_str, transform=ax.transAxes,
                           ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    # If Cox regression fails, just show p-value
                    if pval < 0.001:
                        pval_text = 'p < 0.001'
                    else:
                        pval_text = f'p = {pval:.3f}'
                    ax.text(0.5, 0.95, pval_text, transform=ax.transAxes,
                           ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                # Just show p-value
                if pval < 0.001:
                    pval_text = 'p < 0.001'
                else:
                    pval_text = f'p = {pval:.3f}'
                ax.text(0.5, 0.95, pval_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'serial_6_MACCE分组箱线图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplots to {output_path}")

    # Save p-values and HR results
    df_pvalues = pd.DataFrame(pvalues)
    output_path = os.path.join(RESULTS_DIR, 'serial_7_MACCE分组特征差异p值.csv')
    df_pvalues.to_csv(output_path, index=False)
    print(f"Saved p-values to {output_path}")

    if hr_results:
        df_hr = pd.DataFrame(hr_results)
        output_path = os.path.join(RESULTS_DIR, 'serial_7_MACCE分组特征HR.csv')
        df_hr.to_csv(output_path, index=False)
        print(f"Saved HR results to {output_path}")

    return df_pvalues


def plot_single_segment_trajectories(df_beats, df_merged):
    """Plot age gap trajectories for single segments by MACCE groups."""
    print("\n" + "=" * 80)
    print("Step 8: Plotting single segment trajectories")
    print("=" * 80)

    # Merge beats with MACCE outcome
    df_beats_macce = df_beats.merge(df_merged[['FileName', 'macce_event']], on='FileName', how='inner')

    # Sample a subset for visualization (to avoid overcrowding)
    np.random.seed(42)
    case_files = df_beats_macce[df_beats_macce['macce_event'] == 1]['FileName'].unique()
    control_files = df_beats_macce[df_beats_macce['macce_event'] == 0]['FileName'].unique()

    n_sample = min(100, len(case_files), len(control_files))
    case_sample = np.random.choice(case_files, n_sample, replace=False)
    control_sample = np.random.choice(control_files, n_sample, replace=False)

    df_sample = df_beats_macce[df_beats_macce['FileName'].isin(np.concatenate([case_sample, control_sample]))].copy()

    # Plot age gap trajectories
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Age gap trajectories
    ax = axes[0]
    for fname, group in df_sample[df_sample['macce_event'] == 0].groupby('FileName'):
        ax.plot(group['beat_idx'], group['age_gap'], alpha=0.1, color='blue', linewidth=0.5)
    for fname, group in df_sample[df_sample['macce_event'] == 1].groupby('FileName'):
        ax.plot(group['beat_idx'], group['age_gap'], alpha=0.1, color='red', linewidth=0.5)

    # Add mean trajectories
    case_mean = df_sample[df_sample['macce_event'] == 1].groupby('beat_idx')['age_gap'].mean()
    control_mean = df_sample[df_sample['macce_event'] == 0].groupby('beat_idx')['age_gap'].mean()
    ax.plot(control_mean.index, control_mean.values, color='blue', linewidth=2, label='Control Mean')
    ax.plot(case_mean.index, case_mean.values, color='red', linewidth=2, label='Case Mean')

    ax.set_xlabel('Beat Index')
    ax.set_ylabel('Age Gap (years)')
    ax.set_title('Age Gap Trajectory (Single ECG)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Age gap delta trajectories
    ax = axes[1]
    for fname, group in df_sample[df_sample['macce_event'] == 0].groupby('FileName'):
        deltas = np.diff(group['age_gap'].values)
        ax.plot(range(len(deltas)), deltas, alpha=0.1, color='blue', linewidth=0.5)
    for fname, group in df_sample[df_sample['macce_event'] == 1].groupby('FileName'):
        deltas = np.diff(group['age_gap'].values)
        ax.plot(range(len(deltas)), deltas, alpha=0.1, color='red', linewidth=0.5)

    ax.set_xlabel('Beat Index')
    ax.set_ylabel('Age Gap Delta (years)')
    ax.set_title('Age Gap Delta Trajectory (Single ECG)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'serial_8_单次ECG轨迹图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved single segment trajectories to {output_path}")


def plot_multiple_followups_trajectories(df_serial, df_merged):
    """Plot complete age gap trajectories for subjects with multiple follow-ups."""
    print("\n" + "=" * 80)
    print("Step 9: Plotting multiple follow-ups trajectories")
    print("=" * 80)

    # Merge with MACCE outcome
    df_serial_macce = df_serial.merge(df_merged[['FileName', 'macce_event']], on='FileName', how='inner')

    # Get unique eids with multiple instances
    serial_eids = df_serial_macce['eid'].unique()

    # For each eid, determine MACCE status (use the first FileName's outcome)
    eid_macce = df_serial_macce.groupby('eid')['macce_event'].first()

    # Sample subjects for visualization
    np.random.seed(42)
    case_eids = eid_macce[eid_macce == 1].index.tolist()
    control_eids = eid_macce[eid_macce == 0].index.tolist()

    n_sample = min(50, len(case_eids), len(control_eids))
    case_sample = np.random.choice(case_eids, n_sample, replace=False)
    control_sample = np.random.choice(control_eids, n_sample, replace=False)

    df_sample = df_serial_macce[df_serial_macce['eid'].isin(np.concatenate([case_sample, control_sample]))].copy()

    # Create continuous beat index across all instances
    df_sample = df_sample.sort_values(['eid', 'instance', 'beat_idx']).reset_index(drop=True)
    df_sample['global_beat_idx'] = df_sample.groupby('eid').cumcount()

    # Plot age gap trajectories
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Age gap trajectories
    ax = axes[0]
    for eid in control_sample:
        group = df_sample[df_sample['eid'] == eid]
        ax.plot(group['global_beat_idx'], group['age_gap'], alpha=0.2, color='blue', linewidth=0.5)
    for eid in case_sample:
        group = df_sample[df_sample['eid'] == eid]
        ax.plot(group['global_beat_idx'], group['age_gap'], alpha=0.2, color='red', linewidth=0.5)

    # Add mean trajectories
    case_mean = df_sample[df_sample['eid'].isin(case_sample)].groupby('global_beat_idx')['age_gap'].mean()
    control_mean = df_sample[df_sample['eid'].isin(control_sample)].groupby('global_beat_idx')['age_gap'].mean()
    ax.plot(control_mean.index, control_mean.values, color='blue', linewidth=2, label='Control Mean')
    ax.plot(case_mean.index, case_mean.values, color='red', linewidth=2, label='Case Mean')

    ax.set_xlabel('Beat Index (Across Multiple Follow-ups)')
    ax.set_ylabel('Age Gap (years)')
    ax.set_title('Age Gap Trajectory (Multiple Follow-ups)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Age gap delta trajectories
    ax = axes[1]
    for eid in control_sample:
        group = df_sample[df_sample['eid'] == eid]
        deltas = np.diff(group['age_gap'].values)
        ax.plot(range(len(deltas)), deltas, alpha=0.2, color='blue', linewidth=0.5)
    for eid in case_sample:
        group = df_sample[df_sample['eid'] == eid]
        deltas = np.diff(group['age_gap'].values)
        ax.plot(range(len(deltas)), deltas, alpha=0.2, color='red', linewidth=0.5)

    ax.set_xlabel('Beat Index (Across Multiple Follow-ups)')
    ax.set_ylabel('Age Gap Delta (years)')
    ax.set_title('Age Gap Delta Trajectory (Multiple Follow-ups)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'serial_9_多次随访轨迹图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multiple follow-ups trajectories to {output_path}")


def main():
    """Main function to run all analyses."""
    print("\n" + "=" * 80)
    print("Part III: Serial ECG Analysis")
    print("=" * 80)

    # Step 1: Generate beat-level predictions
    df_beats = generate_beat_level_predictions()

    # Step 2: Identify serial ECG subjects
    df_serial, serial_eids = identify_serial_ecg_subjects(df_beats)

    # Step 3: Extract single segment features
    df_features = extract_single_segment_features(df_beats)

    # Step 4: Merge with MACCE outcome
    df_merged = merge_with_macce_outcome(df_features)

    # Step 5: Plot feature histograms
    plot_feature_histograms(df_features)

    # Step 6: Plot correlation heatmap
    plot_correlation_heatmap(df_features)

    # Step 7: Plot boxplots by MACCE groups
    df_pvalues = plot_boxplots_by_macce(df_merged)

    # Step 8: Plot single segment trajectories
    plot_single_segment_trajectories(df_beats, df_merged)

    # Step 9: Plot multiple follow-ups trajectories
    plot_multiple_followups_trajectories(df_serial, df_merged)

    print("\n" + "=" * 80)
    print("Part III Analysis Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - serial_1_beat_level_predictions.csv")
    print("  - serial_2_ecg_subjects.csv")
    print("  - serial_3_single_segment_features.csv")
    print("  - serial_4_特征分布直方图.png")
    print("  - serial_5_特征相关性热力图.png")
    print("  - serial_6_MACCE分组箱线图.png")
    print("  - serial_7_MACCE分组特征差异p值.csv")
    print("  - serial_8_单次ECG轨迹图.png")
    print("  - serial_9_多次随访轨迹图.png")


if __name__ == "__main__":
    main()

