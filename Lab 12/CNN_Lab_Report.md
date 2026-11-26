# CNN Feature Visualization, Parameter Audit & Critical Analysis

| | |
|---|---|
| **Student Name** | Muhammad Taha |
| **Registration / Roll No.** | FA23-BAI-055 |


---

## 1. Objective

To observe how a CNN turns raw pixels into feature maps by applying five 3×3 kernels, tracing the effect of ReLU and pooling, manually auditing the parameter counts of two architectures, and analysing the results.

**Tools:** Python, NumPy, SciPy (`correlate2d`), Matplotlib, PyTorch (Google Colab).

---

## 2. Part I: Filter Pipeline

### 2.1 Input
A grayscale geometric image with clear edges, resized to 256 × 256 and normalised to [0, 1].

### 2.2 Kernels

| # | Filter | Kernel (3×3) |
|---|---|---|
| 1 | Horizontal Edge (Sobel-X) | `[[-1,-2,-1],[0,0,0],[1,2,1]]` |
| 2 | Vertical Edge (Sobel-Y) | `[[-1,0,1],[-2,0,2],[-1,0,1]]` |
| 3 | Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` |
| 4 | Box Blur | `ones(3,3) / 9` |
| 5 | Emboss | `[[-2,-1,0],[-1,1,1],[0,1,2]]` |

### 2.3 Pipeline stages

| Stage | Operation | Output size |
|---|---|---|
| A | Raw convolution (cross-correlation, `same` padding) | 256 × 256 |
| B | ReLU: `max(0, x)` | 256 × 256 |
| C | Max pooling, 2×2, stride 2 (on Stage B) | 128 × 128 |
| D | Average pooling, 2×2, stride 2 (on Stage B) | 128 × 128 |

### 2.4 Visualization grid (5 × 4)

![5x4 grid](cnn_grid_5x4.png)

*Insert the grid saved from the notebook (`cnn_grid_5x4.png`). Stage A uses a diverging colormap centred on 0 (blue = negative, red = positive). Stages B to D share one 0 to max range per row.*

### 2.5 Observations

- **Horizontal and Vertical rows:** Stage A shows positive (red) and negative (blue) lines on opposite sides of each edge. After ReLU, only the positive responses remain, so each edge shows up as a single thin bright line.
- **Sharpen:** Edges are boosted and flat regions stay close to the original brightness.
- **Box Blur:** The image is smoothed and all values stay positive, so ReLU changes almost nothing.
- **Emboss:** A 3D relief effect appears, with strong positive and negative responses on opposite sides of each edge.
- **Max vs average pooling:** Max pooling (C) keeps thin edges bright. Average pooling (D) makes them dimmer.

---

## 3. Part II: Architecture Comparison & Parameter Audit

**Architecture (both models):** Input 32×32×3 → Conv1 (5 filters) → ReLU → Pool → Conv2 (10 filters) → ReLU → Pool → Global Average Pooling → Dense(10).
Convolutions use `padding='same'`. Pooling, ReLU and GAP have 0 trainable parameters.

**Formula:** `Params per conv layer = (Kh × Kw × Cin + 1) × Cout`

### 3.1 Hand calculations

**Model A (3×3, Max Pooling)**

| Layer | Calculation | Parameters |
|---|---|---|
| Conv1 | (3×3×3 + 1) × 5 = 28 × 5 | 140 |
| Conv2 | (3×3×5 + 1) × 10 = 46 × 10 | 460 |
| Dense | (10 + 1) × 10 | 110 |
| **Total** | 140 + 460 + 110 | **710** |

**Model B (5×5, Average Pooling)**

| Layer | Calculation | Parameters |
|---|---|---|
| Conv1 | (5×5×3 + 1) × 5 = 76 × 5 | 380 |
| Conv2 | (5×5×5 + 1) × 10 = 126 × 10 | 1260 |
| Dense | (10 + 1) × 10 | 110 |
| **Total** | 380 + 1260 + 110 | **1750** |

### 3.2 Feature-map shapes

| Stage | Shape |
|---|---|
| Input | 32 × 32 × 3 |
| Conv1 (same padding) | 32 × 32 × 5 |
| Pool 1 | 16 × 16 × 5 |
| Conv2 (same padding) | 16 × 16 × 10 |
| Pool 2 | 8 × 8 × 10 |
| Global Average Pooling | 1 × 1 × 10 |
| Dense | 10 |

### 3.3 Audit table (manual vs code)

| Model Configuration | Conv Architecture | Kernel Size | Pooling Type | Manual Count | Code Output |
|---|---|---|---|---|---|
| Model A (Standard) | Conv1: 5, Conv2: 10 | 3 × 3 | Max Pooling (2×2, s2) | 710 | 710 |
| Model B (Large Kernel) | Conv1: 5, Conv2: 10 | 5 × 5 | Avg Pooling (2×2, s2) | 1750 | 1750 |

**Result:** Manual and code counts match for both models.

*Attach the `torchinfo` model summary screenshot here.*

---

## 4. Part III: Critical Analysis

### Q1. The ReLU "Blackout" Effect

- **Which regions go black:** After ReLU, all the negative pixels in Stage A become 0. These are 47.5% of the Horizontal Edge map and 45.7% of the Vertical Edge map. They lie on the side of each edge where the kernel's negative weights cover the brighter pixels (bright to dark going down for Sobel-X, and going left to right for Sobel-Y). Flat regions give about 0 and also look black.
- **Why negative:** The kernel is a weighted sum with positive and negative weights. If the side with negative weights is brighter, the sum is negative.

| Activation | Formula | Effect on feature maps |
|---|---|---|
| ReLU | max(0, x) | Negative edges set to 0 (pure black). |
| Leaky ReLU (α = 0.1) | x if x > 0, else 0.1x | Negative edges stay visible as dim lines at 10% strength. The gradient is never 0, so there are no dead units. |
| Sigmoid | 1 / (1 + e⁻ˣ) | Output in (0, 1). A zero input becomes 0.5, so flat regions turn mid-gray. Large values saturate, which lowers contrast, and the gradient is at most 0.25. |

### Q2. Max Pooling vs Average Pooling

| Measure (Sharpen row) | Stage C (Max) | Stage D (Avg) |
|---|---|---|
| Maximum value | 3.09 | 1.789 |

- Max pooling keeps the strongest response in each 2×2 window, so thin sharp edges stay at full brightness. Average pooling mixes the edge pixel with its neighbours, so the edges are dimmer and washed out.
- **Dark ink on white (255):** Average pooling preserves the strokes better. In a test patch with two ink pixels, max pooling gave all 1.0 (the ink was erased), while average pooling gave 0.75 where the ink was. Max pooling would work if the image was inverted first or if min pooling was used.

### Q3. Parameter Explosion & Receptive Field

- **Increase:** (1750 − 710) / 710 = 1040 / 710 = **146.48%**
- **Layers needed:** Two stacked 3×3 layers give a 5×5 receptive field (3 + 3 − 1 = 5).
- **Savings (C channels in and out):**

| Option | Parameters (C = 5) | Formula |
|---|---|---|
| One 5×5 layer | 630 | 25C² + C |
| Two 3×3 layers | 460 | 18C² + 2C |
| **Saved** | **170 (27%)** | |

Without biases, the saving is about 28%. The stack also adds an extra non-linearity.

### Q4. Operation Commutativity

**Answer:** The output is **identical**.

**Proof with the patch [[−3, −1], [−2, 4]]:**

| Order | Steps | Result |
|---|---|---|
| Conv → ReLU → MaxPool | ReLU gives [[0,0],[0,4]], then max | 4 |
| Conv → MaxPool → ReLU | max = 4, then ReLU(4) | 4 |

For an all-negative patch [[−3,−1],[−2,−5]], both orders give 0.

**General reason:** ReLU is monotonic non-decreasing, so max(ReLU(xᵢ)) = ReLU(max(xᵢ)). A check on 10,000 random 2×2 patches also gave identical results.

**Efficiency:** Pooling first means ReLU runs on 1 value instead of 4 per window, which is 75% fewer ReLU operations. This does not hold for average pooling.

### Q5. Diagnosing Filter Redundancy

| # | Root cause | Why it happens | Fix |
|---|---|---|---|
| 1 | Poor or symmetric weight initialization (zeros, identical values, or tiny weights) | Filters get identical gradients, so they stay identical | Use random He/Kaiming initialization (`nn.init.kaiming_normal_`) |
| 2 | Dead ReLUs from a high learning rate | Weights are pushed so far that units always output 0, so the gradient vanishes and the maps look alike | Lower the learning rate (or use warm-up or Adam), add BatchNorm, or use Leaky ReLU |

---

## 5. Summary of Results

| Item | Result |
|---|---|
| Kernels applied | 5 (Sobel-X, Sobel-Y, Sharpen, Box Blur, Emboss) |
| Grid | 5 × 4 (5 filters × Stages A to D) |
| Model A parameters (manual = code) | 710 |
| Model B parameters (manual = code) | 1750 |
| Parameter increase A → B | 146.48% |
| Saving with two 3×3 layers vs one 5×5 (C = 5) | 170 parameters (27%) |
| ReLU and MaxPool order | Identical output (proved) |

## 6. Conclusion

The pipeline shows that edge filters respond strongly at intensity changes, ReLU keeps only the positive responses, and max pooling keeps sharp features better than average pooling. The parameter audit shows that 5×5 kernels cost 146.48% more parameters than 3×3 kernels, while stacked 3×3 layers give the same receptive field with fewer parameters.
