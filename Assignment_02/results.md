# Comparative Implementation and Performance Analysis of Machine Learning Regression Algorithms

**Course:** Computer Vision / Machine Learning (Semester 7)  
**Assignment:** Assignment 02 – Multi-Model Regression Benchmark  
**Student Name:** Muhammad Taha  
**Registration ID:** FA23-BAI-055  

---

## 1. Executive Summary & Objective

This study presents a rigorous empirical and mathematical evaluation of six representative regression paradigms applied to a high-dimensional continuous valuation problem. We systematically formulate, implement, and benchmark:
1. **Ordinary Least Squares (OLS) Linear Regression** (Parametric linear baseline)
2. **Decision Tree Regressor** (Non-parametric axis-aligned recursive partitioner)
3. **Random Forest Regressor** (Ensemble bootstrap bagging of decorrelated trees)
4. **Multi-Layer Perceptron (MLP) Regressor** (Connectionist feed-forward artificial neural network)
5. **Extreme Gradient Boosting (XGBoost) Regressor** (Second-order gradient boosted decision tree ensemble)
6. **K-Nearest Neighbors (KNN) Regressor** (Instance-based non-parametric metric regressor)

The objective is to analyze convergence behavior, residual variance properties, geometric functional approximations, and computational generalization across tabular data.

---

## 2. Mathematical Formulations

### 2.1 Ordinary Least Squares (OLS) Linear Regression
OLS models the target $y \in \mathbb{R}$ as a linear combination of feature vector $\mathbf{x} \in \mathbb{R}^p$ plus additive Gaussian error $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
$$\hat{y} = \mathbf{x}^T \boldsymbol{\beta} = \beta_0 + \sum_{j=1}^p \beta_j x_j$$
The optimal parameter vector $\hat{\boldsymbol{\beta}}$ minimizes the empirical sum of squared residuals:
$$\mathcal{L}_{\text{OLS}}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$
When $\mathbf{X}^T\mathbf{X}$ is non-singular, the unique closed-form normal equation solution is:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
Under the Gauss-Markov theorem, OLS provides the Best Linear Unbiased Estimator (BLUE).

### 2.2 Decision Tree Regression
A regression tree recursively partitions the feature domain into $M$ orthogonal hyper-rectangles $R_1, R_2, \dots, R_M$. The model prediction for a query $\mathbf{x} \in R_m$ is the sample mean of target values within that region:
$$\hat{y}(\mathbf{x}) = \sum_{m=1}^M c_m \cdot \mathbb{I}(\mathbf{x} \in R_m), \quad \text{where } c_m = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$
At each internal node, split variable $j$ and split threshold $s$ are chosen by maximizing variance reduction (minimizing child MSE):
$$\min_{j, s} \left[ \sum_{i \in R_{\text{left}}(j,s)} (y_i - \bar{y}_{\text{left}})^2 + \sum_{i \in R_{\text{right}}(j,s)} (y_i - \bar{y}_{\text{right}})^2 \right]$$

### 2.3 Random Forest Regression
Random Forest mitigates the high variance of individual unpruned decision trees via Bootstrap Aggregation (Bagging) combined with random subspace feature sampling. Given $B$ bootstrapped training subsets $\mathcal{D}_1, \dots, \mathcal{D}_B$:
$$\hat{y}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
Because each individual tree $T_b$ selects among a random subset of $m \approx \sqrt{p}$ features at each split, tree correlation $\rho$ is reduced, attenuating ensemble variance from $\text{Var}(T)$ down to:
$$\text{Var}(\hat{y}_{\text{RF}}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

### 2.4 Multi-Layer Perceptron (MLP) Regressor
The MLP approximates continuous multivariate functions via affine transformations interleaved with non-linear activation functions. For an architecture with hidden representations $\mathbf{h}^{(l)}$:
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{h}^{(1)} = \sigma(\mathbf{z}^{(1)})$$
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)}, \quad \mathbf{h}^{(2)} = \sigma(\mathbf{z}^{(2)})$$
$$\hat{y} = \mathbf{w}_{\text{out}}^T \mathbf{h}^{(2)} + b_{\text{out}}$$
Here, $\sigma(z) = \max(0, z)$ denotes the Rectified Linear Unit (ReLU). Network weights $\boldsymbol{\Theta} = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}$ are optimized via backpropagation using the Adam optimizer to minimize mean squared loss with adaptive moment estimation.

### 2.5 Extreme Gradient Boosting (XGBoost)
XGBoost builds an ensemble of $K$ additive regression trees $\hat{y}_i^{(K)} = \sum_{k=1}^K f_k(\mathbf{x}_i)$ by iteratively minimizing a second-order Taylor expansion of the regularized objective:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
where first and second order gradients are:
$$g_i = \frac{\partial \ell(y_i, \hat{y}^{(t-1)}_i)}{\partial \hat{y}^{(t-1)}_i} = \hat{y}^{(t-1)}_i - y_i, \quad h_i = \frac{\partial^2 \ell(y_i, \hat{y}^{(t-1)}_i)}{\partial (\hat{y}^{(t-1)}_i)^2} = 1$$
The tree regularization term is $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$. The optimal leaf weight $w_j^*$ for leaf $j$ and the corresponding split gain are:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \quad \text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

### 2.6 K-Nearest Neighbors (KNN) Regressor
KNN predicts the target value of a query sample $\mathbf{x}^*$ by querying its $k$ closest neighbors under Euclidean metric $d(\mathbf{x}^*, \mathbf{x}_i) = \|\mathbf{x}^* - \mathbf{x}_i\|_2$. Using inverse distance weighting:
$$\hat{y}(\mathbf{x}^*) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x}^*)} w_i y_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x}^*)} w_i}, \quad \text{where } w_i = \frac{1}{d(\mathbf{x}^*, \mathbf{x}_i)}$$

---

## 3. Experimental Setup & Preprocessing

* **Dataset Size**: 1,000 synthetic property transactions generated via realistic multivariate socio-demographic and physical spatial functions.
* **Target Attribute**: Continuous market price (`Price` in thousands of dollars, $\$k$).
* **Input Features ($p = 6$)**:
  1. `Square_Feet`: Property interior living area ($850.0 - 4800.0\text{ sq ft}$)
  2. `Bedrooms`: Discrete room count ($1 - 6$)
  3. `Bathrooms`: Fractional hygiene installations ($1.0 - 5.0$)
  4. `Age_Years`: Structural age of building ($1 - 45\text{ years}$)
  5. `Distance_City_Center`: Euclidean distance to metropolitan center ($1.2 - 28.5\text{ km}$)
  6. `Location_Score`: Urban locality amenity index ($1 - 10$)
* **Partitioning**: 80% training split ($N_{\text{train}} = 800$), 20% held-out test split ($N_{\text{test}} = 200$) fixed with `random_state = 101`.
* **Standardization**: Distance-sensitive estimators (Linear Regression, MLP Regressor, and KNN) are trained on features standardized via:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  Tree-based estimators (Decision Tree, Random Forest, XGBoost) operate directly on native unscaled feature space due to scale-invariance of monotonic axis-aligned threshold splits.

---

## 4. Quantitative Results & Benchmark

The evaluated models were scored across the 200 held-out test instances on four standard metrics:
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
- **Mean Squared Error (MSE)**: $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$)**: $1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$

### Table 1: Comprehensive Performance Benchmark on Held-Out Test Set

| Rank | Algorithm | MAE ($\$k$) | MSE ($\$k^2$) | RMSE ($\$k$) | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Neural Network (MLP)** | **12.2658** | **227.1058** | **15.0700** | **0.9940** |
| **2** | **Linear Regression (OLS)** | **13.8015** | **315.7333** | **17.7689** | **0.9917** |
| **3** | **XGBoost Regressor** | **18.9880** | **579.2707** | **24.0680** | **0.9848** |
| **4** | **Random Forest Regressor** | **27.2357** | **1211.2762** | **34.8034** | **0.9682** |
| **5** | **K-Nearest Neighbors (KNN)** | **33.8464** | **1717.9021** | **41.4476** | **0.9549** |
| **6** | **Decision Tree Regressor** | **52.1141** | **4077.4698** | **63.8551** | **0.8929** |

---

## 5. Residual Error & Diagnostic Analysis

To evaluate structural model bias and homoscedasticity, we examine the residual distribution $e_i = y_i - \hat{y}_i$:

### Table 2: Residual Error Summary Statistics

| Rank | Algorithm | Mean Error / Bias ($\$k$) | Residual Std Dev ($\$k$) | Min Residual ($\$k$) | Max Residual ($\$k$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Neural Network (MLP) | $+0.4844$ | $15.0622$ | $-36.5274$ | $+48.6322$ |
| 2 | Linear Regression | $+1.5730$ | $17.6991$ | $-46.3924$ | $+59.7621$ |
| 3 | XGBoost Regressor | $+1.6966$ | $24.0082$ | $-81.2247$ | $+70.5454$ |
| 4 | Random Forest | $+1.9886$ | $34.7465$ | $-102.7346$ | $+90.8925$ |
| 5 | K-Nearest Neighbors | $+1.5895$ | $41.4171$ | $-121.4331$ | $+119.6128$ |
| 6 | Decision Tree | $+6.8900$ | $63.4823$ | $-159.3250$ | $+156.1473$ |

### Residual Properties
- **Normality & Zero Centering**: The Multi-Layer Perceptron and Linear Regression models exhibit tight Gaussian residual distributions centered near zero ($\mu_e = +0.48\$k$ and $+1.57\$k$ respectively) with the lowest standard deviations ($15.06\$k$ and $17.70\$k$). This closely matches the standard deviation of the true irreducible data generation noise ($\sigma = 16.5\$k$).
- **Variance Outliers**: Single Decision Tree models display high dispersion ($s = 63.48\$k$) and significant extreme errors ($\pm 159\$k$), underscoring the instability and boundary discretization limitations of single trees.

---

## 6. In-Depth Comparative Discussion & Key Findings

### 6.1 Supremacy of Multi-Layer Perceptron and Linear Regression
* **Continuous Function Approximation**: The physical pricing function contains strong continuous additive factors combined with a non-linear interaction between property square footage and location index:
  $$\text{Interaction} \propto \frac{\text{Square\_Feet}}{1000} \times \text{Location\_Score}$$
* **MLP Non-Linear Capacity**: The 2-layer MLP (128, 64) equipped with ReLU activations successfully recovered both the linear dominant planes and the bilinear interaction term, achieving the top benchmark position ($R^2 = 0.9940$, $\text{RMSE} = 15.0700\$k$).
* **Gauss-Markov Efficiency**: Linear Regression performed exceptionally well ($R^2 = 0.9917$, $\text{RMSE} = 17.7689\$k$). In predominantly additive physical phenomena, parametric models provide minimum-variance unbiased estimates without susceptibility to tree step-discretization errors.

### 6.2 Quantization and Discretization Artifacts in Tree-Based Models
* **Piecewise-Constant Approximation**: Decision Trees, Random Forests, and XGBoost partition the continuous 6-dimensional input space into orthogonal hyper-rectangles with flat, piecewise-constant predictions within each leaf.
* **Curvature and Slope Truncation**: Approximating smooth, continuous linear hyperplanes via discrete axis-aligned step functions requires deep trees, which introduces quantization error. Consequently, the standalone Decision Tree suffered the highest test error ($\text{RMSE} = 63.8551\$k$, $R^2 = 0.8929$).
* **Ensemble Mitigation**:
  - **Random Forest** smoothed out individual tree step discrepancies through 180-tree bagging, improving $R^2$ to $0.9682$.
  - **XGBoost** went further by iteratively fitting successive trees to the exact pseudo-residuals (negative gradient) with shrinkage ($\eta = 0.07$), achieving $R^2 = 0.9848$ and an RMSE of $24.0680\$k$.

### 6.3 Metric Degeneration in K-Nearest Neighbors
* **Curse of Dimensionality**: KNN relies on local sample density. In a 6-dimensional feature space, distance concentration diminishes relative neighborhood contrast:
  $$\lim_{p \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$
* **Spatial Sparsity**: With 800 training observations spread across 6 standardized dimensions, the nearest 5 neighbors are relatively far apart in Euclidean space, pulling predictions toward broader regional averages ($\text{RMSE} = 41.4476\$k$, $R^2 = 0.9549$).

---

## 7. Conclusions & Recommendations

1. **Model Selection for Tabular Regression**: When data generation mechanisms are governed by continuous additive physical laws with smooth interactions, neural networks with proper scaling and linear regression baselines outperform tree ensembles.
2. **Gradient Boosting vs. Bagging**: XGBoost decisively outperformed Random Forest ($\text{RMSE } 24.07\$k\text{ vs } 34.80\$k$), demonstrating that gradient-directed residual minimization converges to continuous functional curves more effectively than uniform bootstrap averaging.
3. **Execution Artifacts**: The accompanying self-contained Jupyter Notebook (`regression_comparison.ipynb`) contains the complete executable pipeline, inline parity scatter plots, residual distributions, and comparative bar charts.
