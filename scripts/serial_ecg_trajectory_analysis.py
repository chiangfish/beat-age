"""
Part IV: Serial ECG Analysis - Experiments
1. Experiment 1: Define 3 groups based on age gap at two time points, draw KM curves for MACCE
2. Experiment 2: Train LSTM and MLP models, compare ROC curves
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Paths
SERIAL_ECG_CSV = "results/serial_2_ecg_subjects.csv"
SERIAL_FEATURES_CSV = "results/serial_3_single_segment_features.csv"
SURVIVAL_DATA = "results/survival_data.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Set plot style for ROC curves only
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def create_composite_outcome(survival_df):
    """
    Use MACCE outcome only (not composite of all 8 outcomes).
    """
    print("\nUsing MACCE outcome...")

    # Use MACCE outcome directly
    survival_df['composite_event'] = survival_df['macce_event']
    survival_df['composite_time'] = survival_df['macce_time']
    survival_df['composite_exclude'] = survival_df['macce_exclude']

    print(f"  Total MACCE events: {survival_df['composite_event'].sum()}/{len(survival_df)} ({100*survival_df['composite_event'].sum()/len(survival_df):.1f}%)")

    return survival_df


def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    """Calculate bootstrap confidence interval for AUC."""
    np.random.seed(42)
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    alpha = 1 - ci
    lower_idx = int(alpha / 2 * len(sorted_scores))
    upper_idx = int((1 - alpha / 2) * len(sorted_scores))

    return sorted_scores[lower_idx], sorted_scores[upper_idx]


def calculate_hr_with_ci(df, group1, group2):
    """Calculate HR with 95% CI using Cox regression."""
    # Prepare data for Cox regression
    df_cox = df[df['group'].isin([group1, group2])].copy()
    df_cox['group_binary'] = (df_cox['group'] == group1).astype(int)

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df_cox[['composite_time', 'composite_event', 'group_binary']],
            duration_col='composite_time', event_col='composite_event')

    # Extract HR and CI
    hr = np.exp(cph.params_['group_binary'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['group_binary', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['group_binary', '95% upper-bound'])

    return hr, ci_lower, ci_upper


def prepare_two_timepoint_data():
    """
    Prepare data for subjects with exactly 2 ECG instances.
    Calculate mean age gap for each instance per subject.
    """
    print("=" * 80)
    print("Preparing two-timepoint data")
    print("=" * 80)

    # Load serial ECG data
    df_serial = pd.read_csv(SERIAL_ECG_CSV)

    # Count instances per subject
    instance_counts = df_serial.groupby('eid')['instance'].nunique()
    print(f"Instance distribution:")
    print(instance_counts.value_counts().sort_index())

    # Filter to subjects with exactly 2 instances
    eids_with_2_instances = instance_counts[instance_counts == 2].index.tolist()
    df_two = df_serial[df_serial['eid'].isin(eids_with_2_instances)].copy()

    print(f"\nSubjects with exactly 2 instances: {len(eids_with_2_instances)}")
    print(f"Total ECG samples: {df_two['FileName'].nunique()}")

    # Calculate mean age gap per FileName (per ECG segment)
    segment_age_gaps = df_two.groupby(['eid', 'FileName', 'instance']).agg({
        'age_gap': 'mean',
        'true_age': 'first'
    }).reset_index()

    # Pivot to get age gaps at two time points
    pivot = segment_age_gaps.pivot_table(
        index='eid',
        columns='instance',
        values='age_gap',
        aggfunc='first'
    ).reset_index()

    # Get the two instances (should be sorted)
    instances = sorted(pivot.columns[1:])  # Skip 'eid' column
    print(f"Instances: {instances}")

    # Rename columns
    pivot.columns = ['eid', 'age_gap_t1', 'age_gap_t2']

    # Load MACCE outcome data (deduplicate by eid)
    survival = pd.read_csv(SURVIVAL_DATA)
    survival = survival.drop_duplicates(subset='eid', keep='first')

    # Use MACCE outcome
    survival = create_composite_outcome(survival)
    survival = survival[['eid', 'composite_event', 'composite_time', 'composite_exclude']]

    # Merge with survival data
    df_merged = pivot.merge(survival, on='eid', how='left')

    # Exclude subjects with missing outcome data or composite_exclude=1
    df_merged = df_merged[df_merged['composite_exclude'] == 0].copy()

    print(f"\nSubjects after merging with MACCE outcome data: {len(df_merged)}")
    print(f"MACCE events: {df_merged['composite_event'].sum()}")

    return df_merged


def define_four_groups(df_merged):
    """
    Define 3 groups based on age gap at two time points.
    G1: Both time points overestimate (age_gap > 0)
    G2: First underestimate/correct (age_gap <= 0), second overestimate (age_gap > 0)
    G3: Both underestimate/correct (age_gap <= 0)

    Note: Removed old G3 (First Over, Second Under/Correct) which had 0 events.
    """
    print("\n" + "=" * 80)
    print("Defining three groups")
    print("=" * 80)

    # Define groups
    conditions = [
        (df_merged['age_gap_t1'] > 0) & (df_merged['age_gap_t2'] > 0),  # G1
        (df_merged['age_gap_t1'] <= 0) & (df_merged['age_gap_t2'] > 0),  # G2
        (df_merged['age_gap_t1'] <= 0) & (df_merged['age_gap_t2'] <= 0),  # G3 (was G4)
    ]
    choices = ['G1', 'G2', 'G3']
    df_merged['group'] = np.select(conditions, choices, default='Unknown')

    # Print group statistics
    print("\nGroup distribution:")
    for group in ['G1', 'G2', 'G3']:
        n = (df_merged['group'] == group).sum()
        events = df_merged[df_merged['group'] == group]['composite_event'].sum()
        print(f"{group}: N={n}, Events={events}")

    return df_merged


def plot_km_curves_experiment1(df_merged):
    """
    Prepare data for KM curves and call R script for plotting.
    Two comparisons: G1 vs G3, G2 vs G3
    """
    print("\n" + "=" * 80)
    print("Preparing data for KM curves (Experiment 1)")
    print("=" * 80)

    import subprocess

    # Create temp directory for data exchange with R
    TEMP_DIR = "results/temp_serial_km_data"
    os.makedirs(TEMP_DIR, exist_ok=True)

    comparisons = [
        ('G1', 'G3', 'MACCE'),
        ('G2', 'G3', 'MACCE')
    ]

    # Prepare Cox results
    cox_results = []

    for group1, group2, outcome_label in comparisons:
        print(f"Processing {group1} vs {group2}...")

        # Prepare survival data for this comparison
        comparison_data = df_merged[df_merged['group'].isin([group1, group2])][
            ['eid', 'group', 'composite_time', 'composite_event']
        ].copy()
        comparison_data = comparison_data.dropna()

        # Rename columns for R
        comparison_data.columns = ['eid', 'group', 'time', 'event']

        # Save to CSV for R
        comparison_name = f'{group1}_vs_{group2}'
        comparison_data.to_csv(f'{TEMP_DIR}/{comparison_name}_data.csv', index=False)

        # Calculate Cox HR
        try:
            hr, ci_lower, ci_upper = calculate_hr_with_ci(df_merged, group1, group2)

            # Calculate p-value using log-rank test
            from lifelines.statistics import logrank_test
            df_g1 = df_merged[df_merged['group'] == group1]
            df_g2 = df_merged[df_merged['group'] == group2]
            results = logrank_test(
                df_g1['composite_time'], df_g2['composite_time'],
                df_g1['composite_event'], df_g2['composite_event']
            )
            p_value = results.p_value

            cox_results.append({
                'comparison': comparison_name,
                'hr': hr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value
            })
        except Exception as e:
            print(f"  Warning: Cox regression failed for {comparison_name}: {e}")

    # Save Cox results for downstream plotting or manuscript table generation.
    cox_df = pd.DataFrame(cox_results)
    cox_df.to_csv(f'{TEMP_DIR}/cox_results.csv', index=False)

    # Save outcome labels for R
    labels_df = pd.DataFrame([
        {'comparison': 'G1_vs_G3', 'label': 'MACCE'},
        {'comparison': 'G2_vs_G3', 'label': 'MACCE'}
    ])
    labels_df.to_csv(f'{TEMP_DIR}/outcome_labels.csv', index=False)

    print(f"\nData prepared. Saved to {TEMP_DIR}/")
    print(f"Cox regression results: {len(cox_results)} comparisons")

    print("Serial ECG KM input files and Cox summaries were saved.")


# ==================== Experiment 2: LSTM and MLP Models ====================

class LSTMModel(nn.Module):
    """LSTM model for age gap sequence classification."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.fc(h_n[-1])
        return out.squeeze()


class MLPModel(nn.Module):
    """MLP model for feature-based classification."""
    def __init__(self, input_size=8, hidden_sizes=[64, 32]):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


class SequenceDataset(Dataset):
    """Dataset for variable-length sequences."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]).unsqueeze(-1), torch.FloatTensor([self.labels[idx]])


def collate_fn_lstm(batch):
    """Collate function for variable-length sequences."""
    sequences, labels = zip(*batch)
    # Pad sequences
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded_seqs = torch.zeros(len(sequences), max_len, 1)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq), :] = seq
    labels = torch.cat(labels)
    return padded_seqs, labels

def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train a model and return the best validation AUC."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(1 - val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_auc


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def prepare_lstm_data_two_instances(df_serial, survival):
    """
    Prepare data for LSTM with two ECG instances.
    Concatenate age gap sequences from both instances.
    """
    print("\nPreparing LSTM data (two instances)...")

    # Deduplicate survival data by eid
    survival = survival.drop_duplicates(subset='eid', keep='first')

    # Use MACCE outcome
    survival = create_composite_outcome(survival)

    # Get subjects with exactly 2 instances
    instance_counts = df_serial.groupby('eid')['instance'].nunique()
    eids_with_2 = instance_counts[instance_counts == 2].index.tolist()
    df_two = df_serial[df_serial['eid'].isin(eids_with_2)].copy()

    # Merge with survival data
    df_two = df_two.merge(survival[['eid', 'composite_event', 'composite_exclude']], on='eid', how='left')
    df_two = df_two[df_two['composite_exclude'] == 0].copy()

    sequences = []
    labels = []

    for eid, group in df_two.groupby('eid'):
        # Sort by instance, then beat_idx
        group = group.sort_values(['instance', 'beat_idx'])
        seq = group['age_gap'].values.tolist()
        label = group['composite_event'].iloc[0]
        sequences.append(seq)
        labels.append(int(label))

    print(f"  Subjects: {len(sequences)}, Events: {sum(labels)}")
    return sequences, labels


def prepare_lstm_data_single_instance(df_serial, survival):
    """
    Prepare data for LSTM with single ECG instance.
    Use only the first instance for each subject.
    """
    print("\nPreparing LSTM data (single instance)...")

    # Deduplicate survival data by eid
    survival = survival.drop_duplicates(subset='eid', keep='first')

    # Use MACCE outcome
    survival = create_composite_outcome(survival)

    # Merge with survival data
    df = df_serial.merge(survival[['eid', 'composite_event', 'composite_exclude']], on='eid', how='left')
    df = df[df['composite_exclude'] == 0].copy()

    # Use only the first instance per subject
    first_instances = df.groupby('eid')['instance'].min().reset_index()
    first_instances.columns = ['eid', 'first_instance']
    df = df.merge(first_instances, on='eid')
    df = df[df['instance'] == df['first_instance']].copy()

    sequences = []
    labels = []

    for eid, group in df.groupby('eid'):
        group = group.sort_values('beat_idx')
        seq = group['age_gap'].values.tolist()
        label = group['composite_event'].iloc[0]
        sequences.append(seq)
        labels.append(int(label))

    print(f"  Subjects: {len(sequences)}, Events: {sum(labels)}")
    return sequences, labels


def prepare_mlp_data(df_serial, survival):
    """
    Prepare data for MLP with single segment features.
    """
    print("\nPreparing MLP data (single segment features)...")

    # Deduplicate survival data by eid
    survival = survival.drop_duplicates(subset='eid', keep='first')

    # Use MACCE outcome
    survival = create_composite_outcome(survival)

    # Load single segment features
    features_df = pd.read_csv(SERIAL_FEATURES_CSV)

    # Merge with survival data
    features_df = features_df.merge(survival[['eid', 'composite_event', 'composite_exclude']], on='eid', how='left')
    features_df = features_df[features_df['composite_exclude'] == 0].copy()

    # Get subjects with multiple instances (serial ECG set)
    instance_counts = df_serial.groupby('eid')['instance'].nunique()
    serial_eids = instance_counts[instance_counts > 1].index.tolist()

    # Filter to serial ECG subjects, use first instance
    features_df['instance'] = features_df['FileName'].str.split('_').str[2].astype(int)
    features_df = features_df[features_df['eid'].isin(serial_eids)].copy()

    # Use only first instance per subject
    first_instances = features_df.groupby('eid')['instance'].min().reset_index()
    first_instances.columns = ['eid', 'first_instance']
    features_df = features_df.merge(first_instances, on='eid')
    features_df = features_df[features_df['instance'] == features_df['first_instance']].copy()

    # Feature columns
    feature_cols = ['mean', 'variance', 'min', 'max', 'range', 'skewness', 'kurtosis', 'rmssd']
    X = features_df[feature_cols].values
    y = features_df['composite_event'].values.astype(int)

    print(f"  Subjects: {len(X)}, Events: {sum(y)}")
    return X, y


def run_experiment2(df_serial, survival):
    """
    Experiment 2: Train LSTM and MLP models, compare ROC curves.
    """
    print("\n" + "=" * 80)
    print("Experiment 2: LSTM and MLP Models")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    seqs_two, labels_two = prepare_lstm_data_two_instances(df_serial, survival)
    seqs_single, labels_single = prepare_lstm_data_single_instance(df_serial, survival)
    X_mlp, y_mlp = prepare_mlp_data(df_serial, survival)

    # Split data
    idx_two = list(range(len(seqs_two)))
    idx_single = list(range(len(seqs_single)))
    idx_mlp = list(range(len(X_mlp)))

    train_idx_two, test_idx_two = train_test_split(idx_two, test_size=0.2, random_state=42,
                                                    stratify=labels_two)
    train_idx_two, val_idx_two = train_test_split(train_idx_two, test_size=0.2, random_state=42,
                                                   stratify=[labels_two[i] for i in train_idx_two])

    train_idx_single, test_idx_single = train_test_split(idx_single, test_size=0.2, random_state=42,
                                                          stratify=labels_single)
    train_idx_single, val_idx_single = train_test_split(train_idx_single, test_size=0.2, random_state=42,
                                                         stratify=[labels_single[i] for i in train_idx_single])

    train_idx_mlp, test_idx_mlp = train_test_split(idx_mlp, test_size=0.2, random_state=42,
                                                    stratify=y_mlp)
    train_idx_mlp, val_idx_mlp = train_test_split(train_idx_mlp, test_size=0.2, random_state=42,
                                                   stratify=y_mlp[train_idx_mlp])

    # Create datasets
    def make_seq_dataset(seqs, labels, indices):
        return SequenceDataset([seqs[i] for i in indices], [labels[i] for i in indices])

    train_ds_two = make_seq_dataset(seqs_two, labels_two, train_idx_two)
    val_ds_two = make_seq_dataset(seqs_two, labels_two, val_idx_two)
    test_ds_two = make_seq_dataset(seqs_two, labels_two, test_idx_two)

    train_ds_single = make_seq_dataset(seqs_single, labels_single, train_idx_single)
    val_ds_single = make_seq_dataset(seqs_single, labels_single, val_idx_single)
    test_ds_single = make_seq_dataset(seqs_single, labels_single, test_idx_single)

    # Create data loaders
    train_loader_two = DataLoader(train_ds_two, batch_size=64, shuffle=True, collate_fn=collate_fn_lstm)
    val_loader_two = DataLoader(val_ds_two, batch_size=64, shuffle=False, collate_fn=collate_fn_lstm)
    test_loader_two = DataLoader(test_ds_two, batch_size=64, shuffle=False, collate_fn=collate_fn_lstm)

    train_loader_single = DataLoader(train_ds_single, batch_size=64, shuffle=True, collate_fn=collate_fn_lstm)
    val_loader_single = DataLoader(val_ds_single, batch_size=64, shuffle=False, collate_fn=collate_fn_lstm)
    test_loader_single = DataLoader(test_ds_single, batch_size=64, shuffle=False, collate_fn=collate_fn_lstm)

    # MLP data
    scaler = StandardScaler()
    X_train_mlp = scaler.fit_transform(X_mlp[train_idx_mlp])
    X_val_mlp = scaler.transform(X_mlp[val_idx_mlp])
    X_test_mlp = scaler.transform(X_mlp[test_idx_mlp])

    train_ds_mlp = TensorDataset(torch.FloatTensor(X_train_mlp), torch.FloatTensor(y_mlp[train_idx_mlp]))
    val_ds_mlp = TensorDataset(torch.FloatTensor(X_val_mlp), torch.FloatTensor(y_mlp[val_idx_mlp]))
    test_ds_mlp = TensorDataset(torch.FloatTensor(X_test_mlp), torch.FloatTensor(y_mlp[test_idx_mlp]))

    train_loader_mlp = DataLoader(train_ds_mlp, batch_size=64, shuffle=True)
    val_loader_mlp = DataLoader(val_ds_mlp, batch_size=64, shuffle=False)
    test_loader_mlp = DataLoader(test_ds_mlp, batch_size=64, shuffle=False)

    # Train models
    print("\nTraining LSTM (two instances)...")
    lstm_two = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
    lstm_two, auc_two_val = train_model(lstm_two, train_loader_two, val_loader_two, device, epochs=50)
    print(f"  Best val AUC: {auc_two_val:.4f}")

    print("\nTraining LSTM (single instance)...")
    lstm_single = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
    lstm_single, auc_single_val = train_model(lstm_single, train_loader_single, val_loader_single, device, epochs=50)
    print(f"  Best val AUC: {auc_single_val:.4f}")

    print("\nTraining MLP (single segment features)...")
    mlp = MLPModel(input_size=8, hidden_sizes=[64, 32]).to(device)
    mlp, auc_mlp_val = train_model(mlp, train_loader_mlp, val_loader_mlp, device, epochs=100)
    print(f"  Best val AUC: {auc_mlp_val:.4f}")

    # Evaluate on test set
    preds_two, labels_two_test = evaluate_model(lstm_two, test_loader_two, device)
    preds_single, labels_single_test = evaluate_model(lstm_single, test_loader_single, device)
    preds_mlp, labels_mlp_test = evaluate_model(mlp, test_loader_mlp, device)

    auc_two = roc_auc_score(labels_two_test, preds_two)
    auc_single = roc_auc_score(labels_single_test, preds_single)
    auc_mlp = roc_auc_score(labels_mlp_test, preds_mlp)

    # Calculate 95% CI for AUC
    ci_two_lower, ci_two_upper = bootstrap_auc_ci(labels_two_test, preds_two)
    ci_single_lower, ci_single_upper = bootstrap_auc_ci(labels_single_test, preds_single)
    ci_mlp_lower, ci_mlp_upper = bootstrap_auc_ci(labels_mlp_test, preds_mlp)

    print(f"\nTest AUC (95% CI):")
    print(f"  LSTM (two instances): {auc_two:.4f} ({ci_two_lower:.4f}-{ci_two_upper:.4f})")
    print(f"  LSTM (single instance): {auc_single:.4f} ({ci_single_lower:.4f}-{ci_single_upper:.4f})")
    print(f"  MLP (features): {auc_mlp:.4f} ({ci_mlp_lower:.4f}-{ci_mlp_upper:.4f})")

    # Plot ROC curves
    fpr_two, tpr_two, _ = roc_curve(labels_two_test, preds_two)
    fpr_single, tpr_single, _ = roc_curve(labels_single_test, preds_single)
    fpr_mlp, tpr_mlp, _ = roc_curve(labels_mlp_test, preds_mlp)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    ax.plot(fpr_two, tpr_two, 'b-', linewidth=2,
            label=f'LSTM (Two-Instance Sequence) AUC = {auc_two:.3f} ({ci_two_lower:.3f}-{ci_two_upper:.3f})')
    ax.plot(fpr_single, tpr_single, 'r-', linewidth=2,
            label=f'LSTM (Single-Instance Sequence) AUC = {auc_single:.3f} ({ci_single_lower:.3f}-{ci_single_upper:.3f})')
    ax.plot(fpr_mlp, tpr_mlp, 'g-', linewidth=2,
            label=f'MLP (Single-Instance Features) AUC = {auc_mlp:.3f} ({ci_mlp_lower:.3f}-{ci_mlp_upper:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('Experiment 2: ROC Curves for Composite Outcome Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'serial_11_实验2_ROC曲线.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved ROC curves to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Part IV: Serial ECG Analysis - Experiments")
    print("=" * 80)

    # Load survival data
    survival = pd.read_csv(SURVIVAL_DATA)

    # Load serial ECG data
    df_serial = pd.read_csv(SERIAL_ECG_CSV)

    # ==================== Experiment 1 ====================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Three Groups KM Analysis for MACCE")
    print("=" * 80)

    df_merged = prepare_two_timepoint_data()
    df_merged = define_four_groups(df_merged)
    plot_km_curves_experiment1(df_merged)

    # ==================== Experiment 2 ====================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: LSTM and MLP Models")
    print("=" * 80)

    run_experiment2(df_serial, survival)

    print("\n" + "=" * 80)
    print("Part IV Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
