"""
Preprocess clinical cohort ECG files into .pt beat files.
Uses the same segment() logic as the healthy population preprocessing.
"""
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from beat_age.segmentation import segment

RAW_ECG_DIR = "datasets/raw_data/ecg_data_filtered"
OUTPUT_DIR = "datasets/processed_beats_clinical"
MASTER_CSV = "datasets/raw_data/dataset.csv"
HEALTH_CSV = "datasets/proc_data/health_population_ecg_metadata.csv"

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_one(args):
    file_name, age, fs = args
    out_path = os.path.join(OUTPUT_DIR, f"{file_name}.pt")
    if os.path.exists(out_path):
        return file_name, True, "skipped"

    npz_path = os.path.join(RAW_ECG_DIR, f"{file_name}.npz")
    if not os.path.exists(npz_path):
        return file_name, False, "npz not found"

    try:
        raw = np.load(npz_path)
        data = np.stack([raw[lead] for lead in LEAD_ORDER], axis=0)  # (12, L)

        rpeaks, segments = segment(data, fs=fs)

        if len(segments) == 0:
            return file_name, False, "no beats detected"

        beat_tensors = [torch.tensor(seg, dtype=torch.float32) for seg in segments]
        torch.save({'beats': beat_tensors, 'age': float(age), 'fs': fs}, out_path)
        return file_name, True, f"{len(segments)} beats"

    except Exception as e:
        return file_name, False, str(e)


def main():
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    health = pd.read_csv(HEALTH_CSV, low_memory=False)
    health_files = set(health['FileName'].tolist())

    clinical = master[~master['FileName'].isin(health_files)].copy()
    clinical = clinical.dropna(subset=['RecordAge'])
    print(f"Clinical cohort: {len(clinical)} files to process")

    tasks = [
        (row['FileName'], row['RecordAge'], int(row['SampleRate']))
        for _, row in clinical.iterrows()
    ]

    success, fail = 0, 0
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            fname, ok, msg = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    print(f"Done. Success: {success}, Failed: {fail}")


if __name__ == "__main__":
    main()
