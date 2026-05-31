#!/usr/bin/env python3
"""
External validation of Beat-age model on MIMIC-IV-ECG dataset.

This script:
1. For each patient, randomly selects 1 ECG recording
2. Excludes ECGs where patient age is not in range 20-89
3. Runs inference using the trained model
4. Calculates Pearson correlation coefficient and MAE
5. Updates Table_2_Performance_of_prediction_model.txt
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from datetime import datetime
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
MIMIC_ECG_DIR = os.environ.get("MIMIC_ECG_DIR", "datasets/raw_data/mimic-iv-ecg")
MIMIC_CORE_DIR = "datasets/raw_data/mimic/mimic-iv-3.1/hosp"
RECORD_LIST_PATH = os.path.join(MIMIC_ECG_DIR, "record_list.csv")
PATIENTS_PATH = os.path.join(MIMIC_CORE_DIR, "patients.csv.gz")
OUTPUT_DIR = "results"
MODEL_PATH = "ckpts/v1_best.pth"


def load_mimic_data():
    """Load MIMIC-IV-ECG record list and patient demographics."""
    print("Loading MIMIC-IV-ECG record list...")
    record_list = pd.read_csv(RECORD_LIST_PATH)
    print(f"Total ECG records: {len(record_list)}")

    print("Loading MIMIC-IV patient demographics...")
    patients = pd.read_csv(PATIENTS_PATH)
    print(f"Total patients: {len(patients)}")

    return record_list, patients


def calculate_age_at_ecg(record_list, patients):
    """Calculate patient age at the time of ECG recording."""
    # Merge record list with patient demographics
    merged = record_list.merge(patients, on='subject_id', how='inner')
    print(f"Records after merging with patient data: {len(merged)}")

    # Extract year from ecg_time
    merged['ecg_year'] = pd.to_datetime(merged['ecg_time']).dt.year

    # Calculate age at ECG time
    # age_at_ecg = anchor_age + (ecg_year - anchor_year)
    merged['age_at_ecg'] = merged['anchor_age'] + (merged['ecg_year'] - merged['anchor_year'])

    return merged


def create_mimic_hospital_mortality_set(merged_data):
    """
    Create MIMIC hospital mortality set:
    1. For each patient, randomly select 1 ECG
    2. Exclude ECGs where age is not in range 20-89
    """
    print("\nCreating MIMIC hospital mortality set...")

    # Step 1: For each patient, randomly select 1 ECG
    print("Step 1: Randomly selecting 1 ECG per patient...")
    selected_records = []

    for subject_id, group in merged_data.groupby('subject_id'):
        # Randomly select one record for this patient
        selected = group.sample(n=1, random_state=42)
        selected_records.append(selected)

    mimic_set = pd.concat(selected_records, ignore_index=True)
    print(f"After selecting 1 ECG per patient: {len(mimic_set)} records")

    # Step 2: Exclude ECGs where age is not in range 20-89
    print("Step 2: Filtering age range 20-89...")
    mimic_set = mimic_set[(mimic_set['age_at_ecg'] >= 20) & (mimic_set['age_at_ecg'] <= 89)]
    print(f"After age filtering: {len(mimic_set)} records")

    return mimic_set


def load_mimic_ecg_wfdb(record_path):
    """Load MIMIC ECG data from WFDB format."""
    import wfdb

    # Load the WFDB record
    record = wfdb.rdrecord(record_path)

    # Get the signal data (shape: (n_samples, n_leads))
    signal = record.p_signal  # Physical signal (in mV)

    # Transpose to (n_leads, n_samples) format
    signal = signal.T

    # Get sampling frequency
    fs = record.fs

    # IMPORTANT: Scale MIMIC data to match UKB data scale
    # MIMIC data is in mV with typical std ~0.22
    # UKB data has typical std ~17.5
    # Scaling factor is approximately 80x (17.5 / 0.22)
    SCALE_FACTOR = 80.0
    signal = signal * SCALE_FACTOR

    return signal, fs


def process_mimic_ecg(record_row):
    """Process a single MIMIC ECG record: load, segment, and return beats."""
    from beat_age.segmentation import segment

    # Construct the full path to the WFDB record
    record_path = os.path.join(MIMIC_ECG_DIR, record_row['path'])

    try:
        # Load WFDB data
        signal, fs = load_mimic_ecg_wfdb(record_path)

        # Ensure we have 12 leads
        if signal.shape[0] != 12:
            return None, f"Expected 12 leads, got {signal.shape[0]}"

        # Segment into beats
        rpeaks, segments = segment(signal, fs=int(fs))

        if len(segments) == 0:
            return None, "No beats detected"

        # Convert to tensors
        beat_tensors = [torch.tensor(seg, dtype=torch.float32) for seg in segments]

        return beat_tensors, None

    except Exception as e:
        return None, str(e)


def run_inference_on_mimic(mimic_set):
    """Run inference on MIMIC hospital mortality set."""
    from beat_age.models import Net1D
    import torch.nn.utils.rnn as rnn_utils

    print("\nRunning inference on MIMIC hospital mortality set...")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Net1D(
        in_channels=12, base_filters=24, ratio=1.0,
        filter_list=[24, 48, 96, 192], m_blocks_list=[2, 2, 2, 2],
        kernel_size=13, stride=1, groups_width=12,
        verbose=False, n_classes=1,
    )

    # Load trained model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    results = []
    failed_count = 0

    with torch.no_grad():
        for idx, row in tqdm(mimic_set.iterrows(), total=len(mimic_set), desc="Inference"):
            # Process ECG to get beats
            beats, error = process_mimic_ecg(row)

            if beats is None:
                failed_count += 1
                continue

            # Prepare beats for inference
            beats_t = [b.permute(1, 0) for b in beats]
            padded = rnn_utils.pad_sequence(beats_t, batch_first=True, padding_value=0.0)
            x = padded.permute(0, 2, 1).to(device)

            # Run inference
            preds = model(x).squeeze(1).cpu().numpy()

            # Aggregate predictions (mean of all beats)
            pred_age = preds.mean()
            true_age = row['age_at_ecg']

            results.append({
                'subject_id': row['subject_id'],
                'study_id': row['study_id'],
                'file_name': row['file_name'],
                'true_age': true_age,
                'pred_age': pred_age,
                'age_gap': pred_age - true_age
            })

    print(f"Inference completed. Success: {len(results)}, Failed: {failed_count}")

    return pd.DataFrame(results)


def calculate_metrics(results_df):
    """Calculate Pearson correlation coefficient and MAE."""
    # Remove NaN values
    results_clean = results_df.dropna(subset=['pred_age'])

    if len(results_clean) < len(results_df):
        print(f"Warning: Removed {len(results_df) - len(results_clean)} records with NaN predictions")

    true_ages = results_clean['true_age'].values
    pred_ages = results_clean['pred_age'].values

    # Pearson correlation coefficient
    pearson_corr, p_value = pearsonr(true_ages, pred_ages)

    # MAE (Mean Absolute Error)
    mae = np.abs(results_clean['age_gap']).mean()

    print(f"\nMetrics (n={len(results_clean)}):")
    print(f"Pearson correlation coefficient: {pearson_corr:.3f} (p={p_value:.2e})")
    print(f"MAE: {mae:.2f} years")

    return pearson_corr, mae


def update_table_2(pearson_corr, mae):
    """Update Table_2_Performance_of_prediction_model.txt with MIMIC results."""
    table_path = os.path.join(OUTPUT_DIR, "Table_2_Performance_of_prediction_model.txt")
    if not os.path.exists(table_path):
        print(f"\nSkipping Table 2 update because {table_path} does not exist.")
        return

    # Read the current table
    with open(table_path, 'r') as f:
        lines = f.readlines()

    # Find the line with "\\hline" before "\\end{tabular}"
    # Insert the new row before the last "\\hline"
    new_row = f"MIMIC hospital mortality set & {pearson_corr:.3f} & {mae:.2f} \\\\\n"

    # Find the position to insert (before the last \hline)
    insert_pos = None
    for i in range(len(lines) - 1, -1, -1):
        if "\\hline" in lines[i]:
            insert_pos = i
            break

    if insert_pos is not None:
        lines.insert(insert_pos, new_row)

        # Write back
        with open(table_path, 'w') as f:
            f.writelines(lines)

        print(f"\nUpdated {table_path}")
    else:
        print(f"Warning: Could not find insertion point in {table_path}")


def main():
    print("=" * 80)
    print("MIMIC-IV-ECG External Validation")
    print("=" * 80)

    # Step 1: Load MIMIC data
    record_list, patients = load_mimic_data()

    # Step 2: Calculate age at ECG time
    merged_data = calculate_age_at_ecg(record_list, patients)

    # Step 3: Create MIMIC hospital mortality set
    mimic_set = create_mimic_hospital_mortality_set(merged_data)

    # For testing, use a small subset first
    # TODO: Remove these lines for full validation
    # mimic_set = mimic_set.head(1000)
    # print(f"\n[TEST MODE] Using only {len(mimic_set)} records for testing")

    # Save the MIMIC hospital mortality set
    mimic_set_path = os.path.join(OUTPUT_DIR, "mimic_hospital_mortality_set.csv")
    mimic_set.to_csv(mimic_set_path, index=False)
    print(f"\nSaved MIMIC hospital mortality set to {mimic_set_path}")

    # Step 4: Run inference
    results_df = run_inference_on_mimic(mimic_set)

    # Save predictions
    predictions_path = os.path.join(OUTPUT_DIR, "mimic_predictions.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to {predictions_path}")

    # Step 5: Calculate metrics
    pearson_corr, mae = calculate_metrics(results_df)

    # Step 6: Update Table 2
    update_table_2(pearson_corr, mae)

    print("\n" + "=" * 80)
    print("External validation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()


