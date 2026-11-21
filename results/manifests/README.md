# Data manifests

Written by the notebook section **Data manifests** (no GPU or training needed).

| File | Content |
|---|---|
| `split_manifest.csv` | One row per image (4,217): file, class, MD5, 64-bit dHash, patient id from file name, group id, split, client |
| `augmentation_manifest.csv` | Real and synthetic image counts per client and class (paper Table 4), with the synthetic-image noise seed |
| `hash_settings.json` | Duplicate-removal, perceptual-hash, grouping, split and partition settings (paper Section 3.3) |

Summary: train 3,069 / validation 570 / test 574; 4 images removed as conflicting-label duplicates; 2,611 patient groups among the kept images. No group crosses a split or a client.
