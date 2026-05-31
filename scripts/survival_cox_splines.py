"""
Part II Survival Analysis: Cox regression, KM curves, and spline plots.
Generates all required figures and tables for Part II.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager, rcParams
regular_path = "Helvetica-01.ttf"
bold_path = "Helvetica-Bold-02.ttf"
font_manager.fontManager.addfont(regular_path)
font_manager.fontManager.addfont(bold_path)
rcParams["font.family"] = "Helvetica"
rcParams["axes.unicode_minus"] = False

# Paths
DATA_CSV = "results/survival_data.csv"
OUTPUT_DIR = "results"

# Load data
df = pd.read_csv(DATA_CSV)
df = df.dropna(subset=['true_age', 'age_gap', 'sex', 'ethnicity', 'bmi'])

# Calculate predicted age for categorical analysis
# Note: age_gap = true_age - predicted_age, so predicted_age = true_age - age_gap
df['predicted_age'] = df['true_age'] - df['age_gap']

# Outcomes
outcomes = ['macce', 'af', 'hypertension', 'chd', 'hf', 'mi', 'stroke', 'death']
outcome_labels = {
    'macce': 'MACCE',
    'af': 'Atrial Fibrillation',
    'hypertension': 'Hypertension',
    'chd': 'Coronary Heart Disease',
    'hf': 'Heart Failure',
    'mi': 'Myocardial Infarction',
    'stroke': 'Stroke',
    'death': 'All-Cause Mortality'
}

print(f"Total subjects: {len(df)}")
print(f"Age gap: mean={df['age_gap'].mean():.2f}, std={df['age_gap'].std():.2f}")
print(f"Age gap groups: {df['age_gap_group'].value_counts().to_dict()}")

# ===== Task 1: Continuous variable analysis with spline curves =====
print("\n=== Task 1: Spline curves (Model 1) ===")

# Model 1: age, sex, ethnicity, BMI
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

# Store all y-values to find global max
all_y_values = []

for idx, outcome in enumerate(outcomes):
    ax = axes[idx]

    # Prepare data — exclude subjects with prior history for this outcome
    data = df[df[f'{outcome}_exclude'] == 0][[f'{outcome}_time', f'{outcome}_event', 'age_gap', 'predicted_age', 'sex', 'ethnicity', 'bmi']].copy()
    data = data.dropna()

    if data[f'{outcome}_event'].sum() < 10:
        ax.text(0.5, 0.5, f'Insufficient events\n({data[f"{outcome}_event"].sum()})',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(outcome_labels[outcome])
        continue

    # Fit Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')

        # Get HR for age_gap at different values
        age_gap_range = np.linspace(-20, 20, 100)
        hrs = []
        ci_lower = []
        ci_upper = []

        for gap in age_gap_range:
            # Baseline: age_gap=0, mean values for others
            baseline = pd.DataFrame({
                'age_gap': [0],
                'predicted_age': [data['predicted_age'].mean()],
                'sex': [data['sex'].mode()[0]],
                'ethnicity': [data['ethnicity'].mode()[0]],
                'bmi': [data['bmi'].mean()]
            })

            # Test: age_gap=gap
            test = baseline.copy()
            test['age_gap'] = gap

            # Compute HR
            hr = np.exp(cph.predict_log_partial_hazard(test).values[0] -
                       cph.predict_log_partial_hazard(baseline).values[0])
            hrs.append(hr)

            # Approximate CI using coefficient SE
            coef = cph.params_['age_gap']
            se = cph.standard_errors_['age_gap']
            ci_lower.append(np.exp((coef - 1.96*se) * gap))
            ci_upper.append(np.exp((coef + 1.96*se) * gap))

        # Collect all y-values for global max calculation
        all_y_values.extend(hrs)
        all_y_values.extend(ci_upper)

        # Plot
        ax.plot(age_gap_range, hrs, 'b-', linewidth=2, label='HR')
        ax.fill_between(age_gap_range, ci_lower, ci_upper, alpha=0.2, color='b', label='95% CI')
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlim(-20, 20)
        ax.set_xlabel('Beat-age gap (years)')
        ax.set_ylabel(f'Adjusted HR (95% CI) for {outcome_labels[outcome]}')
        ax.grid(alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'Model failed:\n{str(e)[:50]}',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(outcome_labels[outcome])

# Set consistent y-axis limits for all subplots
if all_y_values:
    global_ymax = max(all_y_values)
    for ax in axes:
        ax.set_ylim(0, global_ymax)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/formal_fig3_spline_curves.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/formal_fig3_spline_curves.pdf', bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig_spline_curves.png")
plt.close()

# ===== Task 2: Continuous variable HR & p-values (3 models) =====
print("\n=== Task 2: Cox regression (continuous, 3 models) ===")

results_continuous = []

for outcome in outcomes:
    data = df[df[f'{outcome}_exclude'] == 0][[f'{outcome}_time', f'{outcome}_event', 'age_gap', 'predicted_age', 'sex', 'ethnicity', 'bmi',
               'smoking', 'has_hypertension', 'has_diabetes', 'has_dyslipidemia', 'has_ckd',
               'sbp', 'antihypertensive', 'total_chol', 'hdl']].copy()
    data = data.dropna(subset=[f'{outcome}_time', f'{outcome}_event', 'age_gap', 'predicted_age', 'sex', 'ethnicity', 'bmi'])

    if data[f'{outcome}_event'].sum() < 10:
        continue

    # Model 1: adjusted for predicted_age (not true_age to avoid multicollinearity)
    model1_data = data[['age_gap', 'predicted_age', 'sex', 'ethnicity', 'bmi', f'{outcome}_time', f'{outcome}_event']].dropna()
    cph1 = CoxPHFitter()
    try:
        cph1.fit(model1_data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')
        hr1 = np.exp(cph1.params_['age_gap'])
        ci1_lower = np.exp(cph1.confidence_intervals_.loc['age_gap', '95% lower-bound'])
        ci1_upper = np.exp(cph1.confidence_intervals_.loc['age_gap', '95% upper-bound'])
        p1 = cph1.summary.loc['age_gap', 'p']
    except:
        hr1, ci1_lower, ci1_upper, p1 = np.nan, np.nan, np.nan, np.nan

    # Model 2: adjusted for predicted_age + risk factors
    model2_data = data[['age_gap', 'predicted_age', 'sex', 'ethnicity', 'bmi', 'smoking',
                        'has_hypertension', 'has_diabetes', 'has_dyslipidemia', 'has_ckd',
                        f'{outcome}_time', f'{outcome}_event']].dropna()
    cph2 = CoxPHFitter()
    try:
        cph2.fit(model2_data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')
        hr2 = np.exp(cph2.params_['age_gap'])
        ci2_lower = np.exp(cph2.confidence_intervals_.loc['age_gap', '95% lower-bound'])
        ci2_upper = np.exp(cph2.confidence_intervals_.loc['age_gap', '95% upper-bound'])
        p2 = cph2.summary.loc['age_gap', 'p']
    except:
        hr2, ci2_lower, ci2_upper, p2 = np.nan, np.nan, np.nan, np.nan

    # Model 3: adjusted for predicted_age + clinical parameters
    model3_data = data[['age_gap', 'predicted_age', 'sex', 'sbp', 'antihypertensive', 'smoking',
                        'has_diabetes', 'total_chol', 'hdl',
                        f'{outcome}_time', f'{outcome}_event']].dropna()
    cph3 = CoxPHFitter()
    try:
        cph3.fit(model3_data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')
        hr3 = np.exp(cph3.params_['age_gap'])
        ci3_lower = np.exp(cph3.confidence_intervals_.loc['age_gap', '95% lower-bound'])
        ci3_upper = np.exp(cph3.confidence_intervals_.loc['age_gap', '95% upper-bound'])
        p3 = cph3.summary.loc['age_gap', 'p']
    except:
        hr3, ci3_lower, ci3_upper, p3 = np.nan, np.nan, np.nan, np.nan

    results_continuous.append({
        'Outcome': outcome_labels[outcome],
        'Model1_HR': f'{hr1:.3f} ({ci1_lower:.3f}-{ci1_upper:.3f})',
        'Model1_p': f'{p1:.2e}' if not np.isnan(p1) else 'NA',
        'Model2_HR': f'{hr2:.3f} ({ci2_lower:.3f}-{ci2_upper:.3f})',
        'Model2_p': f'{p2:.2e}' if not np.isnan(p2) else 'NA',
        'Model3_HR': f'{hr3:.3f} ({ci3_lower:.3f}-{ci3_upper:.3f})',
        'Model3_p': f'{p3:.2e}' if not np.isnan(p3) else 'NA',
    })

df_continuous = pd.DataFrame(results_continuous)
df_continuous.to_csv(f'{OUTPUT_DIR}/table_continuous_cox.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/table_continuous_cox.csv")
print(df_continuous.to_string(index=False))

# ===== Task 3: Categorical variable analysis (Underestimate vs Overestimate vs Correct) =====
print("\n=== Task 3: Cox regression (categorical, 3 models) ===")

results_categorical = []

for outcome in outcomes:
    data = df[df[f'{outcome}_exclude'] == 0][[f'{outcome}_time', f'{outcome}_event', 'age_gap_group', 'predicted_age', 'sex', 'ethnicity', 'bmi',
               'smoking', 'has_hypertension', 'has_diabetes', 'has_dyslipidemia', 'has_ckd',
               'sbp', 'antihypertensive', 'total_chol', 'hdl']].copy()
    data = data.dropna(subset=[f'{outcome}_time', f'{outcome}_event', 'age_gap_group', 'predicted_age', 'sex', 'ethnicity', 'bmi'])

    # Create dummy variables (reference: Correct)
    data['Underestimate'] = (data['age_gap_group'] == 'Underestimate').astype(int)
    data['Overestimate'] = (data['age_gap_group'] == 'Overestimate').astype(int)

    if data[f'{outcome}_event'].sum() < 10:
        continue

    for group in ['Underestimate', 'Overestimate']:
        # Model 1: adjusted for predicted_age (not true_age to avoid multicollinearity)
        model1_data = data[[group, 'predicted_age', 'sex', 'ethnicity', 'bmi', f'{outcome}_time', f'{outcome}_event']].dropna()
        cph1 = CoxPHFitter()
        try:
            cph1.fit(model1_data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')
            hr1 = np.exp(cph1.params_[group])
            ci1_lower = np.exp(cph1.confidence_intervals_.loc[group, '95% lower-bound'])
            ci1_upper = np.exp(cph1.confidence_intervals_.loc[group, '95% upper-bound'])
            p1 = cph1.summary.loc[group, 'p']
        except:
            hr1, ci1_lower, ci1_upper, p1 = np.nan, np.nan, np.nan, np.nan

        # Model 2: adjusted for predicted_age + risk factors
        model2_data = data[[group, 'predicted_age', 'sex', 'ethnicity', 'bmi', 'smoking',
                            'has_hypertension', 'has_diabetes', 'has_dyslipidemia', 'has_ckd',
                            f'{outcome}_time', f'{outcome}_event']].dropna()
        cph2 = CoxPHFitter()
        try:
            cph2.fit(model2_data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')
            hr2 = np.exp(cph2.params_[group])
            ci2_lower = np.exp(cph2.confidence_intervals_.loc[group, '95% lower-bound'])
            ci2_upper = np.exp(cph2.confidence_intervals_.loc[group, '95% upper-bound'])
            p2 = cph2.summary.loc[group, 'p']
        except:
            hr2, ci2_lower, ci2_upper, p2 = np.nan, np.nan, np.nan, np.nan

        # Model 3: adjusted for predicted_age + clinical parameters
        model3_data = data[[group, 'predicted_age', 'sex', 'sbp', 'antihypertensive', 'smoking',
                            'has_diabetes', 'total_chol', 'hdl',
                            f'{outcome}_time', f'{outcome}_event']].dropna()
        cph3 = CoxPHFitter()
        try:
            cph3.fit(model3_data, duration_col=f'{outcome}_time', event_col=f'{outcome}_event')
            hr3 = np.exp(cph3.params_[group])
            ci3_lower = np.exp(cph3.confidence_intervals_.loc[group, '95% lower-bound'])
            ci3_upper = np.exp(cph3.confidence_intervals_.loc[group, '95% upper-bound'])
            p3 = cph3.summary.loc[group, 'p']
        except:
            hr3, ci3_lower, ci3_upper, p3 = np.nan, np.nan, np.nan, np.nan

        results_categorical.append({
            'Outcome': outcome_labels[outcome],
            'Group': group,
            'Model1_HR': f'{hr1:.3f} ({ci1_lower:.3f}-{ci1_upper:.3f})',
            'Model1_p': f'{p1:.2e}' if not np.isnan(p1) else 'NA',
            'Model2_HR': f'{hr2:.3f} ({ci2_lower:.3f}-{ci2_upper:.3f})',
            'Model2_p': f'{p2:.2e}' if not np.isnan(p2) else 'NA',
            'Model3_HR': f'{hr3:.3f} ({ci3_lower:.3f}-{ci3_upper:.3f})',
            'Model3_p': f'{p3:.2e}' if not np.isnan(p3) else 'NA',
        })

df_categorical = pd.DataFrame(results_categorical)
df_categorical.to_csv(f'{OUTPUT_DIR}/table_categorical_cox.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/table_categorical_cox.csv")
print(df_categorical.to_string(index=False))

print("\n=== All Part II analyses completed ===")
