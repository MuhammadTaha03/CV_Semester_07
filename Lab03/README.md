# Lab 03 — Edge Detection Techniques and Their Impact on Classification Performance

**Author:** Taha ([MuhammadTaha03](https://github.com/MuhammadTaha03))
**Course:** Lab Activity 3, built on top of Lab 01 (pretrained models) and Lab 02 (filter effects)

## 1. Dataset

- **Source:** `nodoubttome/skin-cancer9-classesisic` on Kaggle — same dataset, classes and 80/20
  seeded split as Lab 02, so results are comparable across labs.
- **Classes:** basal cell carcinoma (BCC, 392 imgs), melanoma (MEL, 454), nevus (NV, 373),
  pigmented benign keratosis (PBK, 478).
- **Split:** 1307 train / 326 val / 64 test.

> The `Test` folder only has 64 images total (~16 per class). All accuracy numbers below carry
> real sampling noise from that — a single misclassified image moves accuracy by ~1.6 pts.

## 2. Task 1 — Comparative edge detection

`results/figures/task1_edge_comparison.png` shows Original → Sobel → Prewitt → Laplacian → LoG →
Canny for all 4 classes (exceeds the "at least three classes" requirement).

## 3. Task 2 — Effect of noise (Table 1)

| Edge Detector | Input | Noise | Preprocessing | Edge Density | Connected Components |
|---|---|---|---|---|---|
| Sobel | Original | None | None | 0.0956 | 577 |
| Sobel | Noisy | Gaussian | None | 0.8612 | 4 |
| Sobel | Noisy | Salt & Pepper | None | 0.2076 | 679 |
| Sobel | Noisy | Gaussian | Gaussian Filter | 0.7770 | 17 |
| Sobel | Noisy | Salt & Pepper | Median Filter | 0.4906 | 176 |
| Prewitt | Original | None | None | 0.1025 | 607 |
| Laplacian | Original | None | None | 0.0197 | 518 |
| LoG | Noisy | Gaussian | Gaussian Filter | 0.6894 | 28 |
| Canny | Original | None | Built-in smoothing | 0.0004 | 2 |
| Canny | Noisy | Gaussian | Gaussian Filter | 0.0000 | 0 |
| Canny | Noisy | Salt & Pepper | Median Filter | 0.0000 | 0 |

**Automated noise-sensitivity ranking** (mean edge-density shift vs. the clean baseline, for
detectors that have a matching noisy/no-preprocessing row): Sobel shifts by **0.4388** — i.e. raw
Gaussian noise pushes Sobel's edge density from ~10% of pixels to ~86%, before any filtering. Full
qualitative "Observations" (continuity/sharpness/false edges) still need your write-up — pull them
from `results/figures/task2_noise_effects.png`.

Two results worth calling out directly:
- **Gaussian noise nearly blinds Canny** at these default thresholds (density drops to 0.0000) —
  Canny's internal hysteresis rejects the noise-inflated gradients entirely rather than producing
  false edges, unlike Sobel which floods with false positives.
- **Median filtering recovers Sobel's structure better than Gaussian filtering does** for their
  respective noise types (components drop from 679→176 for salt-and-pepper+median vs.
  4→17 — Gaussian noise's component count is oddly non-monotonic here since heavy noise first
  *merges* everything into one blob before filtering separates it back out; mention this nuance
  in your report rather than reading "17 > 4" as filtering making things worse).

## 4. Task 3 — Canny parameter analysis (Table 2)

| Configuration | Low | High | Kernel | Edge Density | # Detected Edges |
|---|---|---|---|---|---|
| Canny-1 | 30 | 100 | 3x3 | 0.0015 | 74 |
| Canny-2 | 50 | 150 | 3x3 | 0.0004 | 20 |
| Canny-3 | 100 | 200 | 3x3 | 0.0003 | 15 |
| Canny-4 | 50 | 150 | 5x5 | 0.2936 | 14,732 |

Raising both thresholds (Canny-1→3) monotonically thins the edge map, as expected. Canny-4 is the
striking result: keeping the *same* thresholds as Canny-2 but widening the Sobel aperture to 5×5
jumps edge count nearly 750x (20 → 14,732), because a larger Sobel kernel produces much larger raw
gradient magnitudes, so the same absolute threshold values are far less strict — this is worth a
sentence in your report on why kernel size and thresholds can't be tuned independently.

**Selected configuration:** Canny-1 (low=30, high=100, 3×3), auto-picked as the closest edge
density to a moderate 8% target — reasonable, though it's worth a manual look at the Task 3 figure
to confirm it looks right for your images, since the heuristic doesn't know about visual quality.

## 5. Task 4/5 — Classification results (Table 3)

| Model | Accuracy Raw | Accuracy Filtered | Accuracy Edge | Precision (edge) | Recall (edge) | F1 (edge) | Train (s) | Infer (ms) |
|---|---|---|---|---|---|---|---|---|
| SVM | 0.5469 | 0.4375 | 0.2812 | 0.2864 | 0.2812 | 0.2775 | 0.50 | 0.509 |
| Random Forest | 0.5000 | 0.4844 | 0.4219 | 0.4268 | 0.4219 | 0.4104 | 9.57 | 0.345 |
| KNN | 0.5469 | 0.4375 | 0.3438 | 0.3417 | 0.3438 | 0.3388 | 0.00 | 0.112 |
| ResNet50 | 0.6562 | 0.7031 | 0.4062 | 0.3930 | 0.4062 | 0.3839 | 74.05 | 10.323 |
| AlexNet | 0.6719 | 0.6562 | 0.4375 | 0.4046 | 0.4375 | 0.3934 | 61.60 | 7.408 |

> "Filtered" used a Gaussian-blur fallback, not a verified Lab 02 best filter — `filter_summary.csv`
> wasn't reachable in this session (the notebook printed that warning). Re-run Section 6 with that
> file present if you need Set B to reflect Lab 02's actual winning filter.

**Pattern across every model:** Edge accuracy < Raw accuracy, and by a wide margin for both CNNs
(ResNet50: 65.6%→40.6%, AlexNet: 67.2%→43.8%). Filtered stays close to Raw for the CNNs (even
edging it out for ResNet50, 70.3% vs 65.6%) but is flat-to-worse for the classical classifiers.

## 6. Task 6 — Best model, confusion matrices & bar chart

Best-performing model on the edge representation: **AlexNet** (43.75% vs. next-best Random Forest
at 42.19%). `results/figures/task6_confusion_matrices.png` and `task6_bar_chart.png` show AlexNet
across Raw/Filtered/Edge. Auto-generated verdict: *edge-only input reduces accuracy vs. raw for
AlexNet (0.4375 vs 0.6719)*.

## 7. Discussion questions — answers grounded in the numbers above

1. **Noise sensitivity:** Sobel is the only detector with a directly measurable clean-vs-noisy
   comparison here (density shift 0.4388) — its single differentiation still overreacts to
   Gaussian noise; Laplacian/LoG would be expected to be worse still since they differentiate
   twice, but the notebook's table design doesn't give LoG/Laplacian a matching noisy-vs-clean
   pair to confirm that directly — note this as a limitation if your report claims it.
2. **Filtering effect:** Gaussian filtering pulled Sobel's Gaussian-noise density from 0.86 back
   to 0.78 (a small recovery); Median filtering pulled salt-and-pepper density from 0.21 to 0.49 —
   read the components column and figure together, since density alone doesn't capture broken
   vs. continuous edges.
3. **Canny thresholds:** raising both thresholds from (30,100) to (100,200) cut detected edges
   from 74 to 15; kernel size dominates over threshold choice, per the Canny-4 result above.
4. **Edge maps and classification:** reduces accuracy here, for every one of the 5 models — the
   biggest drop is ResNet50 (-25 points).
5. **Information loss:** color and texture (dermoscopic pigmentation pattern) are discarded;
   with a 3-4 class BCC/MEL/NV/PBK problem where color is a known diagnostic cue, this cost shows
   up directly in the accuracy drop above.
6. **Classical vs. deep features:** a CNN's early layers learn edge-like filters as part of a
   jointly-optimized pipeline, rather than committing to one fixed operator + one fixed threshold
   pair before training even starts.
7. **Best representation (this run):** Filtered edges out Raw for ResNet50 (70.3% vs 65.6%) but
   Raw wins for AlexNet and every classical classifier — there's no single winner across all 5
   models; Edge is never the best representation for any model in this run.

## 8. Viva questions

Answered directly in the notebook's Section 11 (conceptual, not data-dependent).

## 9. Files

- `CV_LAB_03_Edge_Detection.ipynb` — executed notebook (all 22 code cells ran without errors)
- `README.md` — this file
- `results/` — `table1_noise_effects.csv`, `table2_canny_params.csv`,
  `table3_crosslab_comparison.csv`, `results_lab3.json`, and `figures/*.png`

## 10. Submitting to GitHub

```bash
cd <your-repo>/Lab03
cp /path/to/CV_LAB_03_Edge_Detection.ipynb .
cp /path/to/README.md .
cp -r /path/to/results .
git add .
git commit -m "Lab 03: edge detection effects on skin-lesion classification"
git push origin main
```
