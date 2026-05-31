"""
Generate train/val/test split CSV for the healthy population.
Split by eid (subject-level) with ratio 80:10:10, seed=42.
Age comes from master dataset.csv (RecordAge).
"""
import pandas as pd
import random

HEALTH_CSV = "datasets/proc_data/health_population_ecg_metadata.csv"
MASTER_CSV = "datasets/raw_data/dataset.csv"
OUTPUT_CSV = "datasets/splits.csv"

health = pd.read_csv(HEALTH_CSV, low_memory=False)
master = pd.read_csv(MASTER_CSV, low_memory=False)

# Merge to get RecordAge for health population
df = health[['FileName', 'eid']].merge(
    master[['FileName', 'RecordAge']], on='FileName', how='left'
)
df = df.dropna(subset=['RecordAge'])

unique_eids = df['eid'].unique().tolist()

random.seed(42)
random.shuffle(unique_eids)

n_total = len(unique_eids)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

eid_to_split = {}
for eid in unique_eids[:n_train]:
    eid_to_split[eid] = 'train'
for eid in unique_eids[n_train:n_train + n_val]:
    eid_to_split[eid] = 'val'
for eid in unique_eids[n_train + n_val:]:
    eid_to_split[eid] = 'test'

df['split'] = df['eid'].map(eid_to_split)

df[['FileName', 'eid', 'RecordAge', 'split']].to_csv(OUTPUT_CSV, index=False)

print(f"Total subjects: {n_total}")
for s in ['train', 'val', 'test']:
    sub = df[df['split'] == s]
    n_subj = sub['eid'].nunique()
    n_files = len(sub)
    print(f"  {s}: {n_subj} subjects, {n_files} files")
print(f"Saved to {OUTPUT_CSV}")
