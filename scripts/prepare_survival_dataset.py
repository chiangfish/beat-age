"""
Prepare survival analysis dataset by merging predictions with outcomes and covariates.
Output: survival_data.csv with all necessary columns for Cox regression and KM curves.
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
PRED_CSV = "results/clinical_predictions.csv"
MASTER_CSV = "datasets/raw_data/dataset.csv"
OUTPUT_CSV = "results/survival_data.csv"

# Load predictions
preds = pd.read_csv(PRED_CSV)
master = pd.read_csv(MASTER_CSV, low_memory=False)

# Reverse age_gap definition: new_age_gap = true_age - predicted_age (instead of predicted_age - true_age)
# This makes age_gap > 0 represent Overestimate (AI predicts lower, actual age is higher)
# and age_gap < 0 represent Underestimate (AI predicts higher, actual age is lower)
preds['age_gap'] = -preds['age_gap']

# Merge to get RecordTime (ECG date)
preds = preds.merge(master[['FileName', 'RecordTime']], on='FileName', how='left')
preds['RecordTime'] = pd.to_datetime(preds['RecordTime'])

# Load demographics
sex = pd.read_csv('datasets/raw_data/31.csv')  # 31-0.0: 0=Female, 1=Male
ethnicity = pd.read_csv('datasets/raw_data/21000.csv')  # 21000-0.0
bmi = pd.read_csv('datasets/raw_data/21001.csv')  # 21001-0.0
smoking = pd.read_csv('datasets/raw_data/20116.csv')  # 20116-0.0

# Merge demographics
df = preds.copy()
df = df.merge(sex[['eid', '31-0.0']].rename(columns={'31-0.0': 'sex'}), on='eid', how='left')
df = df.merge(ethnicity[['eid', '21000-0.0']].rename(columns={'21000-0.0': 'ethnicity'}), on='eid', how='left')
df = df.merge(bmi[['eid', '21001-0.0']].rename(columns={'21001-0.0': 'bmi'}), on='eid', how='left')
df = df.merge(smoking[['eid', '20116-0.0']].rename(columns={'20116-0.0': 'smoking'}), on='eid', how='left')

# Load SBP from category100011
cat100011 = pd.read_csv('datasets/raw_data/category100011.csv', low_memory=False)
df = df.merge(cat100011[['eid', '4080-0.0']].rename(columns={'4080-0.0': 'sbp'}), on='eid', how='left')

# Load cholesterol and HDL
chol_hdl = pd.read_csv('datasets/raw_data/6157_6177_30690.csv', low_memory=False)
df = df.merge(chol_hdl[['eid', '30690-0.0']].rename(columns={'30690-0.0': 'total_chol'}), on='eid', how='left')

hdl = pd.read_csv('datasets/raw_data/30760.csv', low_memory=False)
df = df.merge(hdl[['eid', '30760-0.0']].rename(columns={'30760-0.0': 'hdl'}), on='eid', how='left')

# Load antihypertensive medication (6157 for females, 6177 for males)
# 6157/6177: 1=cholesterol, 2=blood pressure, 3=insulin, -7=none
df = df.merge(chol_hdl[['eid', '6157-0.0', '6177-0.0']], on='eid', how='left')
df['antihypertensive'] = 0
df.loc[(df['sex'] == 0) & (df['6157-0.0'] == 2), 'antihypertensive'] = 1
df.loc[(df['sex'] == 1) & (df['6177-0.0'] == 2), 'antihypertensive'] = 1
df = df.drop(columns=['6157-0.0', '6177-0.0'])

# Load ICD codes for outcomes
c2404 = pd.read_csv('datasets/raw_data/C2404.csv', low_memory=False)  # Endocrine
c2409 = pd.read_csv('datasets/raw_data/C2409.csv', low_memory=False)  # Circulatory
c2414 = pd.read_csv('datasets/raw_data/C2414.csv', low_memory=False)  # Genitourinary

# Load MI and Stroke
mi = pd.read_csv('datasets/raw_data/MI.csv')
stroke = pd.read_csv('datasets/raw_data/Stroke.csv')

# Load COPD
copd = pd.read_csv('datasets/raw_data/42016.csv')

# Load death date
death = pd.read_csv('datasets/raw_data/40000.csv')

# Merge outcomes
df = df.merge(mi[['eid', '42000-0.0']].rename(columns={'42000-0.0': 'mi_date'}), on='eid', how='left')
df = df.merge(stroke[['eid', '42006-0.0']].rename(columns={'42006-0.0': 'stroke_date'}), on='eid', how='left')
df = df.merge(copd[['eid', '42016-0.0']].rename(columns={'42016-0.0': 'copd_date'}), on='eid', how='left')
df = df.merge(death[['eid', '40000-0.0']].rename(columns={'40000-0.0': 'death_date'}), on='eid', how='left')

# Merge ICD codes
# Diabetes: E10-E14 (130706, 130708, 130710, 130712, 130714)
diab_cols = ['130706-0.0', '130708-0.0', '130710-0.0', '130712-0.0', '130714-0.0']
df = df.merge(c2404[['eid'] + diab_cols], on='eid', how='left')
for col in diab_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['diabetes_date'] = df[diab_cols].min(axis=1)
df = df.drop(columns=diab_cols)

# Hypertension: Essential (131286) + Secondary (131294)
hyp_cols = ['131286-0.0', '131294-0.0']
df = df.merge(c2409[['eid'] + hyp_cols], on='eid', how='left')
for col in hyp_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['hypertension_date'] = df[hyp_cols].min(axis=1)
df = df.drop(columns=hyp_cols)

# CHD: I20-I25 (131296, 131298, 131300, 131302, 131304, 131306)
chd_cols = ['131296-0.0', '131298-0.0', '131300-0.0', '131302-0.0', '131304-0.0', '131306-0.0']
df = df.merge(c2409[['eid'] + chd_cols], on='eid', how='left')
for col in chd_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['chd_date'] = df[chd_cols].min(axis=1)
df = df.drop(columns=chd_cols)

# Heart failure: I50 (131354)
df = df.merge(c2409[['eid', '131354-0.0']].rename(columns={'131354-0.0': 'hf_date'}), on='eid', how='left')

# AF/AFL: I48 (131350)
df = df.merge(c2409[['eid', '131350-0.0']].rename(columns={'131350-0.0': 'af_date'}), on='eid', how='left')

# Dyslipidemia: E78 (130814)
df = df.merge(c2404[['eid', '130814-0.0']].rename(columns={'130814-0.0': 'dyslipidemia_date'}), on='eid', how='left')

# CKD: N18 (132032)
df = df.merge(c2414[['eid', '132032-0.0']].rename(columns={'132032-0.0': 'ckd_date'}), on='eid', how='left')

# Convert date columns to datetime
date_cols = ['mi_date', 'stroke_date', 'death_date', 'diabetes_date', 'hypertension_date',
             'chd_date', 'hf_date', 'af_date', 'copd_date', 'dyslipidemia_date', 'ckd_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Compute MACCE (composite: death, HF, MI, stroke) — use earliest date regardless of ECG timing
# (the per-outcome logic below will handle prior history exclusion)
df['macce_date'] = df[['death_date', 'hf_date', 'mi_date', 'stroke_date']].min(axis=1)

# Compute follow-up time and event indicators
# Follow-up end: earliest of (event date, death date, 2023-12-31)
end_date = pd.Timestamp('2023-12-31')

outcomes = ['macce', 'diabetes', 'hypertension', 'chd', 'hf', 'mi', 'stroke', 'af', 'copd', 'death']
for outcome in outcomes:
    date_col = f'{outcome}_date'
    event_date = df[date_col]
    # Exclude subjects with prior history (event before ECG)
    prior_history = event_date.notna() & (event_date <= df['RecordTime'])
    df[f'{outcome}_exclude'] = prior_history.astype(int)
    # Event: only if event occurred AFTER ECG recording
    df[f'{outcome}_event'] = (event_date.notna() & (event_date > df['RecordTime'])).astype(int)
    # Follow-up time in years from ECG date
    df[f'{outcome}_time'] = (event_date.where(df[f'{outcome}_event'] == 1, end_date) - df['RecordTime']).dt.days / 365.25
    # Safety clamp: negative times shouldn't exist after above logic, but clamp just in case
    df.loc[df[f'{outcome}_time'] < 0, f'{outcome}_time'] = 0

# Create age gap groups: Underestimate (<-9), Correct (-9 to 9), Overestimate (>9)
# Note: age_gap = true_age - predicted_age (REVERSED from original definition)
# age_gap > 9: true_age >> predicted_age → AI predicts much younger → biological age OVERESTIMATED → label as "Overestimate"
# age_gap < -9: true_age << predicted_age → AI predicts much older → biological age UNDERESTIMATED → label as "Underestimate"
df['age_gap_group'] = 'Correct'
df.loc[df['age_gap'] > 9, 'age_gap_group'] = 'Overestimate'  # age_gap > 9: Overestimate
df.loc[df['age_gap'] < -9, 'age_gap_group'] = 'Underestimate'  # age_gap < -9: Underestimate

# Binary covariates for Model 2
df['has_hypertension'] = (df['hypertension_date'] < df['RecordTime']).astype(int)
df['has_diabetes'] = (df['diabetes_date'] < df['RecordTime']).astype(int)
df['has_dyslipidemia'] = (df['dyslipidemia_date'] < df['RecordTime']).astype(int)
df['has_ckd'] = (df['ckd_date'] < df['RecordTime']).astype(int)

# Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved survival data to {OUTPUT_CSV}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nOutcome event counts (post-ECG only):")
for outcome in outcomes:
    n_event = df[f'{outcome}_event'].sum()
    n_exclude = df[f'{outcome}_exclude'].sum()
    print(f"  {outcome}: {n_event} events, {n_exclude} excluded (prior history)")
