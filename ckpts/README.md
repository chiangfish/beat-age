# Checkpoints

Model checkpoints are not stored in this GitHub repository. Download the released weights from the Hugging Face repository associated with the paper and place them here.

Expected filenames used by the scripts:

- `v1_best.pth`: Beat-age beat-level Net1D model trained on the UKB Development Cohort.
- `ecg10s_net1d_64x1024_v1_best.pth`: recording-level ECG-age baseline model.
- `mimic_adapted.pth`: optional Beat-age model after MIMIC domain adaptation.
- `ecg10s_net1d_64x1024_mimic_finetuned.pth`: optional ECG-age baseline after MIMIC domain adaptation.
