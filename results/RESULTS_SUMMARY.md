# FediGAN-ViT — results summary (paper_fast, backbone vit_small_patch16_224)

Runs completed: 19; skipped for time: 0
Test set: 574 real images (patient-grouped, never seen in training, no synthetic images).

## Proposed model (mean ± SD over 3 seeds)

- accuracy: 92.45 ± 0.61 %
- balanced_accuracy: 92.47 ± 0.65 %
- precision_macro: 92.47 ± 0.61 %
- recall_macro: 92.47 ± 0.65 %
- f1_macro: 92.44 ± 0.61 %
- auc_macro: 98.95 ± 0.08 %
- Cohen's κ: 0.899; MCC: 0.900

## Tables
- `T01_dataset.csv`
- `T01b_duplicates.csv`
- `T02_partition.csv`
- `T03_hyperparameters.csv`
- `T04_main_results.csv`
- `T05_per_class.csv`
- `T06_ablation.csv`
- `T07_significance.csv`
- `T08_robustness.csv`
- `T09_gan_quality.csv`
- `T10_complexity.csv`
- `all_runs_long.csv`
- `convergence_history.csv`

## Figures
- `F03_preprocessing.png`
- `F04_partition_and_augmentation.png`
- `F05_cgan_losses.png`
- `F07_real_vs_synthetic_client0.png`
- `F07_real_vs_synthetic_client1.png`
- `F07_real_vs_synthetic_client2.png`
- `F07b_memorisation_client0.png`
- `F07b_memorisation_client1.png`
- `F07b_memorisation_client2.png`
- `F08_convergence.png`
- `F09_confusion_matrix.png`
- `F10_roc_pr.png`
- `F11_test_predictions.png`
- `F12_gradcam.png`
- `F13_ablation.png`

## Wording to use in the paper
- Report macro-averaged metrics (classes are balanced overall but not per client).
- Inception Score is computed on ImageNet classes and is weakly meaningful for fundus images; FID/KID are reported with a real-vs-real floor.
- Validation for model selection is a held-out server set (simulation); state this explicitly.
- No formal privacy mechanism (DP / secure aggregation) is used; describe the framework as 'data-local', not 'privacy-guaranteed'.