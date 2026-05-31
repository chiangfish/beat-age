"""
Inference script: Generate predictions for all clinical cohort subjects.
Outputs: predictions.csv with columns [FileName, eid, true_age, pred_age, age_gap]
"""
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

from beat_age.models import Net1D

# Paths
CLINICAL_BEATS_DIR = "datasets/processed_beats_clinical"
MASTER_CSV = "datasets/raw_data/dataset.csv"
HEALTH_CSV = "datasets/proc_data/health_population_ecg_metadata.csv"
MODEL_PATH = "ckpts/v1_best.pth"
OUTPUT_CSV = "results/clinical_predictions.csv"

os.makedirs("results", exist_ok=True)

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

def main():
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

    # Inference
    dataset = ClinicalDataset(file_list)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    results = []
    with torch.no_grad():
        for x, y, fnames in tqdm(loader, desc="Inference"):
            if len(x) == 0:
                continue
            x = x.to(device)
            preds = model(x).squeeze(1).cpu().numpy()
            ages = y.numpy()

            # Aggregate by FileName (mean of all beats)
            for fname in set(fnames):
                mask = np.array([f == fname for f in fnames])
                pred_age = preds[mask].mean()
                true_age = ages[mask][0]
                results.append({
                    'FileName': fname,
                    'true_age': true_age,
                    'pred_age': pred_age,
                    'age_gap': pred_age - true_age
                })

    # Save
    df_results = pd.DataFrame(results)
    df_results = df_results.merge(clinical[['FileName', 'eid']], on='FileName', how='left')
    df_results = df_results[['FileName', 'eid', 'true_age', 'pred_age', 'age_gap']]
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to {OUTPUT_CSV}")
    print(f"MAE: {np.abs(df_results['age_gap']).mean():.4f}")
    print(f"Pearson: {np.corrcoef(df_results['true_age'], df_results['pred_age'])[0,1]:.4f}")

if __name__ == "__main__":
    main()
