# FediGAN-ViT

Code and results for:

> **FediGAN-ViT: A Federated Conditional GAN and Vision Transformer Framework for Data-Local Retinal Disease Classification with Visual Explanations**
> Deepthi A. and Sammulal, Jawaharlal Nehru Technological University, Hyderabad.
> Under review, *International Journal of Intelligent Engineering and Systems* (IJIES).

FediGAN-ViT trains a Vision Transformer (ViT-S/16) across three simulated clients with FedProx. Each client uses a local conditional GAN (cGAN) to top up its under-represented classes. Raw images, synthetic images and generators never leave a client; only classifier parameters are exchanged. Grad-CAM gives qualitative visual explanations. The task is four-class fundus classification (Cataract, Diabetic Retinopathy, Glaucoma, Normal) on the public Eye Diseases Classification (EDC) dataset.

**Main result** (three seeds, patient-grouped test set of 574 images): accuracy 92.45 ± 0.61%, macro-F1 92.44 ± 0.61%, macro-AUC 0.989. That is within 1.1 points of centralised training and 10.1 macro-F1 points above clients training alone. cGAN augmentation performs on par with matched random oversampling.

## Repository contents

```
FediGAN_ViT_Kaggle.ipynb          Complete pipeline: data, cGANs, federated training, evaluation, figures
src/fedigan_core.py               Read-only copy of the training engine (the notebook writes and runs this file)
analysis/revision_statistics.py   Recomputes paper Tables 13 and 14 from the saved predictions (CPU only)
results/
  tables/                         All result tables (CSV) used in the paper
  figures/                        All result figures (PNG, 300 dpi)
  runs_paper_fast/                One log (.json) and one prediction file (.npz) per experiment and seed
  manifests/                      Patient-group / split / client manifests (see "Data manifests")
  RESULTS_SUMMARY.md              Headline numbers
requirements.txt
LICENSE
```

## Data

The EDC dataset is public on Kaggle: <https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification>. It has 4,217 images in four class folders. The notebook finds it automatically under `/kaggle/input`. This repository does not redistribute the images.

## Reproducing the experiments (Kaggle)

1. Create a Kaggle notebook from `FediGAN_ViT_Kaggle.ipynb`.
2. Settings: **Accelerator GPU T4 × 2**, **Internet On**. Add the EDC dataset as input.
3. In the **Settings** cell keep `PROFILE = "paper"` and `SPEED = "fast"`. These are the settings used in the paper: ViT-S/16, up to 15 rounds, 19 experiments, seeds 42, 43 and 44.
4. Choose **Save Version → Save & Run All (Commit)**. The run takes roughly 2–4 hours. Two background workers train one experiment per GPU. The notebook then builds every table, figure and the Grad-CAM maps into `/kaggle/working/fedigan_outputs/`, and packs them into `fedigan_results_paper_fast.zip`.
5. **Resuming:** if a run stops early, add the stopped version's output as input (**Add Input → Notebook Output Files**) and commit again. Finished experiments are restored and skipped.

The other `SPEED` options (`"standard"`: ViT-B/16, 20 rounds, 32 experiments; `"full"`: 56 experiments) are included for completeness. They were **not** used in the paper.

GPU training is not bit-wise deterministic (cuDNN autotuning, mixed precision, data-loader workers). A re-run therefore reproduces the reported results up to small run-to-run differences, typically a few tenths of a percentage point. The patient groups, the splits, the client partition and the number of synthetic images are fully deterministic.

## Data manifests

The section **Data manifests** of the notebook writes the following files to `fedigan_outputs/manifests/`. Copy them to `results/manifests/`.

| File | Content |
|---|---|
| `split_manifest.csv` | One row per image: file name, class, MD5, 64-bit dHash, patient id parsed from the file name, group id, split (train / val / test / excluded duplicate), client |
| `augmentation_manifest.csv` | Real and synthetic image counts per client and class, with the synthetic-image noise seed |
| `hash_settings.json` | Duplicate-removal, perceptual-hash, grouping, split and partition settings |

The cell also checks that no patient group crosses a split or a client. It needs no GPU and no training. To produce only the manifests, run these cells in order: install, **Settings**, **Write the experiment engine**, **Load the engine into this notebook** and **Data manifests**. Skip the training cells.

## Reproducing the revision statistics (no GPU)

```bash
pip install -r requirements.txt
python analysis/revision_statistics.py results/runs_paper_fast results/tables
```

This writes `table13_per_seed_paired.csv` and `table14_per_class.csv`. They hold the per-seed paired comparisons with t-based and paired-bootstrap confidence intervals, and the per-class recall with Glaucoma/Normal confusions. They match paper Tables 13 and 14.

## Where each paper table and figure comes from

| Paper | File in `results/` |
|---|---|
| Table 2: dataset and patient-grouped split | `tables/T01_dataset.csv`, `tables/T01b_duplicates.csv` |
| Table 3: training parameters | `tables/T03_hyperparameters.csv` |
| Table 4: client data before and after augmentation | `tables/T02_partition.csv` |
| Table 5: cGAN image quality | `tables/T09_gan_quality.csv` |
| Table 6: performance across rounds | `tables/convergence_history.csv` |
| Table 7: final performance | `tables/T04_main_results.csv` |
| Table 8: class-wise performance | `tables/T05_per_class.csv` |
| Table 9: ablation | `tables/T06_ablation.csv` |
| Table 10: robustness | `tables/T08_robustness.csv` (seed-42 column: `runs_paper_fast/proposed__s42.json`) |
| Table 11: complexity and communication | `tables/T10_complexity.csv` |
| Table 12: statistical comparison | `tables/T07_significance.csv` |
| Table 13: per-seed paired comparison | `tables/table13_per_seed_paired.csv` |
| Table 14: per-class recall and Glaucoma/Normal errors | `tables/table14_per_class.csv` |
| Fig. 3: preprocessing | `figures/F03_preprocessing.png` |
| Fig. 4: client class distribution | `figures/F04_partition_and_augmentation.png` |
| Fig. 5: cGAN losses | `figures/F05_cgan_losses.png` |
| Fig. 6: real vs. synthetic (client 0) | `figures/F07_real_vs_synthetic_client0.png` (clients 1–2 also provided) |
| Fig. 7: memorisation check (client 0) | `figures/F07b_memorisation_client0.png` (clients 1–2 also provided) |
| Fig. 8: convergence | `figures/F08_convergence.png` |
| Fig. 9: confusion matrices | `figures/F09_confusion_matrix.png` |
| Fig. 10: ROC and precision–recall | `figures/F10_roc_pr.png` |
| Fig. 11: test predictions | `figures/F11_test_predictions.png` |
| Fig. 12: Grad-CAM | `figures/F12_gradcam.png` |
| Fig. 13: ablation | `figures/F13_ablation.png` |

Tables 1 and 15 summarise published work, and Figs. 1 and 2 are diagrams. None of them is produced by code.

## Run files

Each experiment and seed has two files in `results/runs_paper_fast/`:

- `<experiment>__s<seed>.json`: configuration, test metrics (overall, per class, confusion matrix), per-round training and validation history, selected round, training time and communication volume.
- `<experiment>__s<seed>.npz`: per-image test labels `y` and class probabilities `prob`. For the local-only experiment, `prob` holds one array per client model.

The experiment keys are `proposed`, `fedprox_none`, `fedprox_oversample`, `central_none`, `local_gan` (three seeds each), and `fedavg_gan`, `partial`, `loco_c0`, `fedprox_gan_resnet50` (seed 42). For `loco_c0`, `__unseen.npz` holds the predictions on the held-out client partition.

## Method settings (as in the paper)

- **Data:** de-duplication by MD5, with all copies of conflicting-label duplicates removed. Patient groups are built from `<id>_left/_right` file names plus exact dHash matches (sets of 2–5 images), using union-find before any split. The split is stratified-group 70/15/15 with seed 42.
- **Preprocessing:** crop at intensity 10, resize to 224, field-of-view mask (threshold 10, 5 × 5 erosion), illumination correction (σ = H/30), colour NLM (5, 5, 7, 21), CLAHE (clip 2.0, 8 × 8 tiles).
- **Federation:** 3 clients, group-level Dirichlet partition with α = 0.5 and each client holding at least 10% of the data. FedProx μ = 0.01, with weights from real-image counts. Up to 15 rounds of 1 local epoch each; the round with the best validation macro-AUC is kept (early-stopping patience 4). The test set is never used for selection.
- **Classifier:** ViT-S/16 (timm, ImageNet-21k pretrained), dropout 0.3, class-balanced focal loss (β = 0.9999, γ = 2) using local class counts. AdamW 1e-4 with weight decay 0.05, cosine decay, gradient clipping 1.0, FP16, batch size 32.
- **cGAN:** 64 × 64, latent 100, label embedding 50, 100 epochs, batch 32, Adam 2e-4 (β₁ = 0.5), label smoothing 0.9. Synthetic images per class are min(gap to the majority class, 100), upsampled bicubically to 224, with noise seed 10·seed + client.

## Citation

The citation will be added after publication.

## License

Code is released under the MIT License (see `LICENSE`). The EDC images remain subject to the dataset's own terms on Kaggle.

## Contact

Deepthi A., deepthi@rnd.jntuh.ac.in
