# Lab 02 — Effect of Image Filtering on Skin-Lesion Classification

**Author:** Taha ([MuhammadTaha03](https://github.com/MuhammadTaha03))
**Course:** Lab Activity 2, built on top of Lab Activity 1 (pretrained-model comparison)

## 1. Objective

Investigate how five spatial-domain image filters affect the performance of the three best
pretrained models identified in Lab Activity 1, on the same skin-lesion classification task.

## 2. Dataset

- **Source:** `nodoubttome/skin-cancer9-classesisic` on Kaggle (ISIC-derived skin-lesion images),
  downloaded in-notebook via `kagglehub`.
- **Selected classes (4, instructor-approved subset):** `melanoma`, `nevus`,
  `pigmented benign keratosis`, `basal cell carcinoma`.
- **Split:** identical 80/20 train/val split from the provided `Train` folder (`torch.Generator`
  seeded at 42), plus the dataset's own `Test` folder as the held-out test set — the same split
  used in Lab 1, so results stay comparable across labs.
- Exact image counts per class/split are printed and plotted by the notebook (Section 1.1,
  `class_distribution.png`) once it is run — see "How to run" below.

> Note: the task sheet references the HAM10000 dataset; this notebook uses an ISIC-derived,
> instructor-approved 4-class subset instead, consistent with the "students may use ... an
> instructor-approved subset" option. If your instructor expects the class list/count to be
> reported against HAM10000 specifically, confirm this substitution with them first.

## 3. Models

Carried over from Lab Activity 1's results (ranked by test accuracy, ties broken by macro-F1):

| Rank | Model | Accuracy | Macro-F1 | AUC |
|---|---|---|---|---|
| 1 | ResNet50 | 78.12% | 76.40 | 90.98 |
| 2 | AlexNet | 75.00% | 74.15 | 90.66 |
| 3 | ResNet101 | 75.00% | 70.78 | 92.02 |

## 4. Filters applied

Average/Mean, Gaussian, Median, Sharpening, Sobel edge — plus a **No Filter** baseline. Every
filter is applied identically to train, validation and test images at the 224x224 input
resolution; split, augmentation, optimizer, schedule and evaluation are held constant across all
18 runs (3 models x 6 conditions), per the task's fair-comparison requirement.

## 5. Notebook sections

| Requirement | Notebook section |
|---|---|
| Dataset preparation | 1. Dataset preparation |
| Image filtering | 2. Image filtering |
| Model loading | 4. Model loading |
| Baseline experiment | The `"No Filter"` condition inside Section 6's experiment loop |
| Training | 5. Training |
| Evaluation | `evaluate()` / `compute_metrics()` (Section 5) + 7. Required results table |
| Visualization | 8. Visualisation (accuracy/loss curves, confusion matrices, per-class F1) |
| Comparative analysis | 9. Comparative analysis (delta vs. baseline, cross-model consistency, auto-generated findings) |
| Computational cost | 10. Computational-efficiency note |
| Lab questions (1–10) | 11. Answers to the lab questions |
| Limitations | 12. Additional analysis — limitations and honesty note |

> The task sheet asks for a standalone "Baseline experiment" section — here the baseline is the
> `"No Filter"` entry inside the single 3x6 experiment loop rather than a separate code block.
> This keeps every run on identical code paths (which is what the fair-comparison requirement
> actually needs) but doesn't literally have its own heading; mention that in your write-up if the
> instructor is checking section-by-section.

## 6. Required results table

Section 7 generates the exact `Model | Filter | Accuracy | Precision | Recall | F1-score |
Macro-F1 | AUC` table the task asks for, saved to `results/main_table.csv` and printed as
Markdown for pasting into a report.

## 7. Additional analysis coverage

| Required item | Where |
|---|---|
| Class distribution | Section 1.1, `class_distribution.png` |
| Original vs. filtered examples | Section 2.1, `filter_examples.png` |
| Confusion matrices | Section 8.2, `confusion_matrices.png` |
| Train/val accuracy curves | Section 8.1, `curves.png` |
| Train/val loss curves | Section 8.1, `curves.png` |
| Per-class precision/recall/F1 | Section 8.3, `per_class_metrics.csv`, `per_class_f1.png` |
| Macro-F1 | `results/main_table.csv` and `full_metrics.csv` |
| Balanced accuracy | `full_metrics.csv` (`*_balanced_accuracy` columns) |
| AUC | `main_table.csv` (macro, one-vs-rest); no separate ROC curve plot is drawn |

## 8. How to run

1. Open the notebook in Google Colab and set the runtime to a GPU (T4 or better).
2. Run all cells top to bottom. Expected total runtime is ~60–75 min on a T4 — the 18 runs are
   checkpointed to `results/results.json` after each one finishes, so a Colab disconnect resumes
   instead of restarting.
3. Outputs are written to `/content/results/`: `main_table.csv`, `full_metrics.csv`,
   `per_class_metrics.csv`, `delta_vs_baseline.csv`, `filter_summary.csv`, `results.json`, and the
   PNG figures listed above.
4. The final cell zips `results/` to `lab02_results.zip` and triggers a Colab download.

**Status:** this copy of the notebook has not been executed yet (no cell outputs) — the numbers in
Section 11's answers reflect the *expected* pattern of results and should be re-checked against
your own run's `main_table.csv` / `delta_vs_baseline.csv` before submitting.

## 9. Submitting to GitHub

The notebook does not push to GitHub itself. After running it in Colab:

```bash
git clone <your-repo-url>
cd <your-repo>/Lab02
cp /path/to/CV_LAB_02_Filter_Effects.ipynb .
cp /path/to/README.md .
# optionally copy results/*.csv and results/*.png too
git add .
git commit -m "Lab 02: image filtering effects on skin-lesion classification"
git push
```

## 10. Files in this folder

- `CV_LAB_02_Filter_Effects.ipynb` — the complete experimental notebook
- `README.md` — this file
