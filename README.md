# Beat-age

This repository contains the implementation of **Beat-age**, a deep learning model for estimating ECG-derived biological age from individual cardiac cycles. The model is based on a one-dimensional residual CNN (`Net1D`) and is trained on segmented 12-lead ECG beats from healthy participants in the UK Biobank.

Beat-age produces beat-level age predictions that can be averaged at the recording level. The resulting Beat-age gap, defined as predicted age minus chronological age, was used in the paper as a digital biomarker for cardiovascular risk stratification.

For a detailed explanation, please refer to the paper:

> AI-Derived Beat-age and Its Variability as Digital Biomarkers for Cardiovascular Risk Stratification

Model weights are released on https://huggingface.co/chiangfish/beat-age.

If you use this code, please cite:

```bibtex
@article{beatage2026,
  title = {Beat-Level Electrocardiographic Biological Age and Its Variability as Digital Biomarkers for Cardiovascular Risk Stratification},
  author = {Zirui Jiang, Guangkun Nie, Qinghao Zhao, and Shenda Hong},
  year = {2026}
}
```

## Requirements

The code is written in Python and was tested with Python 3.12. The main dependencies are:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
- [wfdb](https://wfdb.readthedocs.io/) for MIMIC-IV-ECG records
- [lifelines](https://lifelines.readthedocs.io/) for downstream survival analyses
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for figures

Install with conda:

```bash
conda env create -f environment.yml
conda activate beat-age
pip install -e .
```

or with pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Repository Structure

```text
src/beat_age/
  models.py          Net1D model definition
  datasets.py        PyTorch datasets and collate functions
  segmentation.py    ECG beat segmentation utilities

scripts/
  prepare_ukb_clinical_beats.py
  create_ukb_development_splits.py
  train_beat_age_model.py
  predict_ukb_clinical.py
  prepare_survival_dataset.py
  survival_cox_splines.py
  extract_beat_age_sequence_features.py
  serial_ecg_trajectory_analysis.py
  saliency_mapping.py
  mimic_beat_age_external_validation.py

ckpts/               Place downloaded model weights here
datasets/            Place controlled-access ECG and metadata files here
results/             Generated predictions, tables, and figures
configs/             Example environment/path configuration
```

## Data Preparation

This repository does not include UK Biobank or MIMIC-IV data. Arrange the UKB ECG data and metadata in the following layout:

```text
datasets/
  raw_data/
    dataset.csv
    ecg_data_filtered/*.npz
    31.csv
    21000.csv
    21001.csv
    20116.csv
    category100011.csv
    6157_6177_30690.csv
    30760.csv
    C2404.csv
    C2409.csv
    C2414.csv
    MI.csv
    Stroke.csv
    40000.csv
  proc_data/
    health_population_ecg_metadata.csv
```

Each ECG `.npz` file should contain the 12 leads:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

For MIMIC-IV-ECG external validation, set the WFDB root directory:

```bash
export MIMIC_ECG_DIR=/path/to/mimic-iv-ecg/1.0
```

## Steps: Train Beat-age

### 1. Segment ECG recordings into beats

The training code expects preprocessed beat tensors under `datasets/processed_beats/`. The clinical inference script prepares the clinical beat directory:

```bash
python scripts/prepare_ukb_clinical_beats.py
```

If preparing the Development Cohort beats, use the same segmentation format:

```python
{
    "beats": [Tensor(12, L), Tensor(12, L), ...],
    "age": float,
    "fs": int
}
```

saved as:

```text
datasets/processed_beats/<FileName>.pt
```

### 2. Create subject-level train/validation/test splits

```bash
python scripts/create_ukb_development_splits.py
```

This writes:

```text
datasets/splits.csv
```

The split is performed at the participant level so the training, validation, and test sets do not share participants.

### 3. Train the Beat-age model

```bash
python scripts/train_beat_age_model.py --device 0 --name v1
```

The script trains a beat-level `Net1D` regression model and saves the best checkpoint by validation MAE:

```text
ckpts/v1_best.pth
```

Main training settings:

- Input: segmented 12-lead ECG beats
- Target: chronological age at ECG recording
- Loss: mean squared error
- Optimizer: AdamW
- Scheduler: warmup plus cosine decay
- Batch construction: all beats from each recording are flattened and dynamically padded

## Inference

After training, run Beat-age prediction on the UKB Clinical Evaluation Cohort:

```bash
python scripts/predict_ukb_clinical.py
```

Output:

```text
results/clinical_predictions.csv
```

with columns:

```text
FileName, eid, true_age, pred_age, age_gap
```

where:

```text
age_gap = pred_age - true_age
```

## Pretrained Weights

Place downloaded weights in `ckpts/`. The expected Beat-age checkpoint filename is:

```text
ckpts/v1_best.pth
```

The repository intentionally excludes `.pth` files from git because checkpoints are large and will be distributed through Hugging Face.

## Additional Analysis Scripts

The repository keeps a small number of downstream scripts used in the main manuscript:

- `prepare_survival_dataset.py`: merge Beat-age predictions with clinical outcomes and covariates
- `survival_cox_splines.py`: Cox regression and restricted cubic spline analyses
- `extract_beat_age_sequence_features.py`: compute beat-level sequence features including RMSSD
- `serial_ecg_trajectory_analysis.py`: serial ECG trajectory analysis
- `saliency_mapping.py`: input-gradient saliency maps
- `mimic_beat_age_external_validation.py`: MIMIC-IV-ECG external validation

These scripts are secondary to the model training workflow and require the corresponding controlled-access datasets and generated prediction files.

## License

The code in this repository is released under the MIT License.