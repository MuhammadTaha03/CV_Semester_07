# Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

**Project Overview:** An empirical benchmark evaluating stochastic wrapper feature selection via a Genetic Algorithm (GA) to prune high-dimensional Histogram of Oriented Gradients (HOG) descriptors for chest radiograph classification on the PneumoniaMNIST dataset.

---

## 1. Executive Summary

Traditional hand-crafted visual descriptors such as the Histogram of Oriented Gradients (HOG) capture fine edge and gradient structures across localized image blocks. However, when applied to standardized medical radiography benchmarks, dense block representations generate substantial feature redundancy and noise from anatomically non-informative regions (e.g., background air margins, chest periphery).

This study implements and evaluates a **Genetic Algorithm (GA) wrapper feature selection pipeline** against an end-to-end full-descriptor baseline using an **RBF-Kernel Support Vector Machine (SVM)** classifier. 

### Key Findings
* **Dimensionality Reduction:** Pruned the feature space from **$8,100$** to **$2,312$** features (**$71.46\%$ reduction**).
* **Classification Improvement:** Boosted test set **Macro-F1 from $77.36\%$ to $80.63\%$ ($+3.27$ pp)** and overall accuracy from **$81.57\%$ to $83.81\%$ ($+2.24$ pp)**.
* **Sensitivity & Balance:** Improved Balanced Accuracy and Macro Recall by **$+3.08$ pp** ($75.51\%$ to $78.59\%$), mitigating majority-class bias.
* **Statistical Rigor:** A 2,000-iteration paired bootstrap test yielded a $95\%$ confidence interval of **$[+1.44\text{ pp}, +5.35\text{ pp}]$**, strictly excluding zero ($p < 0.05$).
* **Inference Speedup:** Reduced per-sample test-set inference latency by **$3.91\times$** ($0.86\text{ s}$ down to $0.22\text{ s}$).

---

## 2. Dataset & Problem Formulation

### 2.1 The PneumoniaMNIST Benchmark
The dataset represents standardized anterior-posterior chest X-rays categorized into binary diagnostic targets:
1. **Normal** ($y = 0$)
2. **Pneumonia** ($y = 1$)

Native images ($28 \times 28$ resolution) were upscaled via bicubic interpolation to $128 \times 128$ single-channel grayscale arrays to allow localized gradient and cell-block formation.

### 2.2 Class Imbalance and Data Partitioning
The dataset follows the canonical MedMNIST benchmark splits:

| Split | Total Samples | Normal ($y=0$) | Pneumonia ($y=1$) | Class Ratio (Normal : Pneumonia) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | $3,882$ | $388$ | $3,494$ | $\approx 1 : 9.01$ |
| **Validation** | $524$ | $135$ | $389$ | $\approx 1 : 2.88$ |
| **Test** | $624$ | $234$ | $390$ | $\approx 1 : 1.67$ |

To prevent the classifier from collapsing toward the dominant Pneumonia class, inverse class-frequency weighting (`class_weight="balanced"`) was enforced in the SVM loss formulation.

---

## 3. Feature Extraction Architecture

Histogram of Oriented Gradients (HOG) features were extracted uniformly across all splits using the following spatial parameters:

* **Image Input Resolution:** $128 \times 128$ pixels
* **Cell Dimensions:** $8 \times 8$ pixels per cell
* **Block Dimensions:** $2 \times 2$ cells per block ($16 \times 16$ pixels per block with $50\%$ overlap)
* **Orientation Bins:** $9$ unsigned gradient channels ($0^\circ$ to $180^\circ$)
* **Block Normalization:** L2-Hys (L2-norm followed by clipping at $0.2$ and re-normalizing)

### Dimension Calculation:
$$\text{Cells per axis} = \frac{128}{8} = 16$$
$$\text{Blocks per axis} = 16 - 2 + 1 = 15$$
$$\text{Total Blocks} = 15 \times 15 = 225$$
$$\text{Features per Block} = 2 \times 2 \times 9 = 36$$
$$\text{Total Feature Vector Dimension } (N) = 225 \times 36 = 8,100$$

Feature extraction required $58.50\text{ s}$ across all $5,030$ images using multi-core parallelization (`joblib`, `n_jobs=-1`).

---

## 4. Accelerated SVM Formulation

To perform hundreds of full-scale SVM trainings inside the evolutionary loop without prohibitive runtime costs, a hybrid GPU-CPU architecture was constructed:

1. **GPU Pairwise Gram Matrix Computation:**
   Pairwise Euclidean squared distance matrices are vectorized on GPU using PyTorch tensors:
   $$D^2(A, B) = \mathbf{1}_B (\mathbf{a} \odot \mathbf{a})^\top + (\mathbf{b} \odot \mathbf{b}) \mathbf{1}_A^\top - 2 A B^\top$$
   $$K(A, B) = \exp(-\gamma \cdot D^2(A, B))$$

2. **Adaptive Kernel Scaling:**
   To guarantee an unbiased comparison across changing subset cardinalities $k$, the kernel scale parameter $\gamma$ dynamically adapts to feature variance:
   $$\gamma = \frac{\gamma_{\text{mult}}}{k \cdot \text{Var}(X_{\text{sub}}) + 10^{-12}}$$
   where $\gamma_{\text{mult}} \in \{0.5, 1.0, 2.0\}$ and $C \in \{0.1, 1, 10, 100\}$.

3. **CPU Quadratic Programming:**
   The precomputed kernel matrix is passed to `sklearn.svm.SVC(kernel="precomputed")` to solve the dual optimization problem via LIBSVM.

---

## 5. Genetic Algorithm Design

The feature selection task is modeled as a binary combinatorial optimization problem.

```
+---------------------------------------------------------------------------------+
|                              Chromosome (Length: 8,100)                         |
|  [ 1 | 0 | 1 | 1 | 0 | 0 | 1 | ... | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 ]    |
|   |         |   |         |                   |         |   |                       |
|  Active    Inactive      Active             Inactive   Active                       |
+---------------------------------------------------------------------------------+
```

### 5.1 Chromosome Representation
* Length $L = 8,100$ bits.
* Binary encoding: $c_i \in \{0, 1\}$, where $c_i = 1$ denotes retention of the $i$-th HOG feature and $c_i = 0$ denotes exclusion.

### 5.2 Population Initialization
* Individuals $1 \dots P-1$ are initialized using Bernoulli distributions with densities sampled uniformly from $U(0.2, 0.8)$.
* **Seeded Baseline (Elitism Anchor):** Chromosome $0$ is explicitly set to an all-ones vector ($\mathbf{1}^{8100}$), guaranteeing that the GA population begins with access to the full baseline feature set.

### 5.3 Objective and Fitness Function
The fitness function simultaneously maximizes validation classification quality while penalizing chromosome density:

$$\text{Fitness}(\mathbf{c}) = \text{Macro-F1}_{\text{val}}(\mathbf{c}) - \lambda \cdot \left(\frac{\sum_{i=1}^{N} c_i}{N}\right)$$

* **$\text{Macro-F1}_{\text{val}}$:** Macro-averaged F1 score evaluated on the full validation split ($524$ images).
* **$\lambda = 0.05$:** Sparsity regularization coefficient.
* **$N = 8,100$:** Total feature dimension.
* **Fitness Evaluation Subsampling:** During the GA loop, models are fit on a stratified training sub-sample of $N_{\text{sub}} = 2,000$ images to enable rapid evaluation cycles.

### 5.4 Genetic Operators
* **Selection:** Tournament selection ($k_{\text{tourn}} = 3$).
* **Crossover:** Uniform crossover ($p_{\text{cx}} = 0.90$).
* **Mutation:** Independent bit-flip mutation ($p_{\text{mut}}$).
* **Elitism:** Exact preservation of the top $2$ individuals per generation.
* **Feasibility Constraint:** A hard floor of $k_{\min} = 50$ active features is enforced.

---

## 6. Experimental Configurations

### Table 1: Experimental System Configuration

| Parameter | Baseline: HOG + SVM | Proposed: GA-HOG + SVM |
| :--- | :--- | :--- |
| **Dataset** | PneumoniaMNIST | PneumoniaMNIST |
| **Input Image Dimensions** | $128 \times 128$ | $128 \times 128$ |
| **HOG Orientations** | $9$ | $9$ |
| **Pixels per Cell** | $8 \times 8$ | $8 \times 8$ |
| **Cells per Block** | $2 \times 2$ | $2 \times 2$ |
| **Block Normalization** | L2-Hys | L2-Hys |
| **Initial Feature Dimension** | $8,100$ | $8,100$ |
| **Feature Selection** | None (All features retained) | Genetic Algorithm (Binary mask) |
| **Classifier** | RBF-Kernel SVM | RBF-Kernel SVM |
| **Class Weights** | Balanced | Balanced |
| **Hyperparameter Grid** | Validation Grid ($C$, $\gamma$) | Validation Re-tuning on Selected Mask |
| **Data Split** | Fixed ($3,882$ / $524$ / $624$) | Identical Fixed Split |
| **Random Seed** | $42$ | $42$ |

---

## 7. Hyperparameter Sweep & Model Selection

Three distinct GA parameter sets were executed over $15$ generations:

### Table 2: GA Configuration Comparison (Validation Set)

| Configuration | Population | Generations | Mutation Prob. ($p_{\text{mut}}$) | Best Fitness | Sub-Sample Val F1 | Features Kept ($k$) | Reduction ($\%$) | Evaluations | GA Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GA-A** | $20$ | $15$ | $0.002$ | $0.9482$ | $96.13\%$ | $2,129$ | $73.7\%$ | $290$ | $170.7\text{ s}$ |
| **GA-B (Selected)** | $\mathbf{30}$ | $\mathbf{15}$ | $\mathbf{0.002}$ | $\mathbf{0.9497}$ | $\mathbf{96.40\%}$ | $\mathbf{2,312}$ | $\mathbf{71.5\%}$ | $\mathbf{450}$ | $\mathbf{273.3\text{ s}}$ |
| **GA-C** | $20$ | $15$ | $0.010$ | $0.9448$ | $95.86\%$ | $2,246$ | $72.3\%$ | $290$ | $179.3\text{ s}$ |

**Selection Decision:** Configuration **GA-B** achieved the highest validation fitness score ($0.9497$). Its optimal binary chromosome was exported, followed by retraining on the complete $3,882$-sample training set with validation-tuned parameters ($C = 1.0, \gamma_{\text{mult}} = 0.5$).

---

## 8. Test Set Evaluation & Benchmarking

To ensure unbiased performance reporting, the test set ($624$ unseen radiographs) was accessed only once following final pipeline selection.

### Table 3: Final Test Set Performance Comparison

| Metric | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) | Absolute Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Active Features ($k$)** | $8,100$ | $\mathbf{2,312}$ | $\mathbf{-5,788\text{ (-71.46\%)}}$ |
| **Overall Accuracy** | $81.571\%$ | $\mathbf{83.814\%}$ | $\mathbf{+2.244\text{ pp}}$ |
| **Macro Precision** | $88.255\%$ | $\mathbf{89.106\%}$ | $+0.851\text{ pp}$ |
| **Macro Recall** | $75.513\%$ | $\mathbf{78.590\%}$ | $\mathbf{+3.077\text{ pp}}$ |
| **Macro-F1 Score** | $77.364\%$ | $\mathbf{80.630\%}$ | $\mathbf{+3.266\text{ pp}}$ |
| **Weighted-F1 Score** | $79.803\%$ | $\mathbf{82.593\%}$ | $+2.790\text{ pp}$ |
| **Balanced Accuracy** | $75.513\%$ | $\mathbf{78.590\%}$ | $+3.077\text{ pp}$ |
| **Macro Specificity** | $75.513\%$ | $\mathbf{78.590\%}$ | $+3.077\text{ pp}$ |
| **Cohen's Kappa ($\kappa$)** | $0.565$ | $\mathbf{0.623}$ | $+0.059$ |
| **Matthews Corr. Coeff. (MCC)**| $0.625$ | $\mathbf{0.669}$ | $+0.044$ |

---

## 9. Statistical Significance Testing

To verify that the $+3.27\text{ pp}$ improvement in Macro-F1 was not a stochastic artifact of test set sampling, a **paired bootstrap analysis** was conducted:
* **Iterations:** $B = 2,000$ bootstrap replicates drawn with replacement from the test set.
* **Metric:** $\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{GA}} - \text{Macro-F1}_{\text{Baseline}}$
* **Mean Delta:** $+3.27\text{ pp}$
* **$95\%$ Empirical Bootstrap Confidence Interval:** $[+1.44\text{ pp}, +5.35\text{ pp}]$

$$\text{CI}_{0.95} = [1.44, 5.35] \quad (\text{Strictly excludes } 0)$$

**Conclusion:** The diagnostic enhancement is statistically significant at the $\alpha = 0.05$ significance level.

---

## 10. Computational Complexity and Overhead

### Table 4: Pipeline Execution Cost Profile

| Pipeline Stage | HOG + SVM Baseline | GA-HOG + SVM Proposed | Speedup / Overhead |
| :--- | :---: | :---: | :---: |
| **HOG Feature Extraction (All Splits)** | $58.50\text{ s}$ | $58.50\text{ s}$ | Identical ($1.0\times$) |
| **GA Evolutionary Search** | $0.00\text{ s}$ | $273.25\text{ s}$ | $+273.25\text{ s}$ (Upfront cost) |
| **Hyperparameter Grid Tuning** | $18.48\text{ s}$ | $8.00\text{ s}$ | $2.31\times$ faster |
| **Final Full-Dataset SVM Fitting** | $4.85\text{ s}$ | $1.75\text{ s}$ | $2.77\times$ faster |
| **Test Set Inference ($624$ images)** | $0.86\text{ s}$ | $\mathbf{0.22\text{ s}}$ | $\mathbf{3.91\times\text{ faster}}$ |
| **Total Pipeline Wall Time** | $\mathbf{82.69\text{ s}}$ | $\mathbf{341.72\text{ s}}$ | $4.13\times$ higher build time |
| **Retained Support Vectors** | $947$ | $\mathbf{889}$ | $58\text{ fewer support vectors}$ |

### Inference Analysis:
At test time, the RBF kernel computation scales with feature dimensionality $d$ and the number of support vectors $N_{\text{sv}}$:
$$\mathcal{O}_{\text{inference}} = \mathcal{O}(M_{\text{test}} \cdot N_{\text{sv}} \cdot d)$$

By reducing $d$ by $71.46\%$ and $N_{\text{sv}}$ by $6.12\%$, test-time prediction achieves an effective **$\approx 3.91\times$ wall-clock speedup**, making the resulting model significantly better suited for clinical edge deployments.

---

## 11. Discussion & Spatial Feature Distribution

### 11.1 Anatomical Interpretation
Spatial analysis of the retained feature mask (averaged across all $15 \times 15$ block positions) demonstrates that the Genetic Algorithm selectively preserves features localized over:
1. **The lung parenchyma and retrocardiac spaces**, where focal consolidations and infiltrates appear.
2. **Costophrenic angles and diaphragm contours**, where effusion blunting is clinically assessed.

In contrast, features corresponding to peripheral image borders, high-frequency background padding, and uniform thoracic regions were pruned, directly confirming that the algorithm removed non-diagnostic gradient variance.

### 11.2 Trade-offs and Limitations
* **Upfront Search Overhead:** While deployment inference is nearly $4\times$ faster, the search phase requires $450$ SVM evaluations ($273\text{ s}$).
* **Validation Noise Sensitivity:** With a validation split of $524$ images, aggressive mutation rates ($p_{\text{mut}} = 0.010$) exhibited signs of over-fitting validation-specific variance (yielding lower fitness in GA-C).
* **Upscaled Gradients:** The native PneumoniaMNIST resolution is $28 \times 28$. Upscaling to $128 \times 128$ introduces interpolated gradients; applying this pipeline to native high-resolution DICOM radiographs is expected to yield even clearer anatomical localization.

---

## 12. Artifact Inventory

The experimental run generated and serialized the following artifacts:

* `table1_config.csv`: Baseline vs. proposed configuration parameters.
* `table_ga_settings.csv`: Comparative population search dynamics across configurations GA-A, GA-B, and GA-C.
* `table2_main_comparison.csv`: Complete test set diagnostic benchmark metrics.
* `table3_computational_cost.csv`: Stage-by-stage wall-clock profiling and complexity data.
* `selected_feature_mask.npy`: Optimized $8,100$-bit boolean selection vector.
* `ga_history_selected.csv`: Generation-by-generation fitness, Macro-F1, and cardinality trajectories.
* `samples.png`, `ga_fitness.png`, `features_and_metrics.png`, `confusion_matrices.png`: Visualization figures.