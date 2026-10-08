# Pipeline Summary and Analysis Explanation

This notebook demonstrates a complete machine learning pipeline for **Industrial Defect Detection and Classification** using **Histogram of Oriented Gradients (HOG)** features and classical classifiers. Below is an explanation of what occurs in each phase of the project:

### 1. Dataset Setup & Characterization
* **Source:** The NEU Steel Surface Defect dataset (containing 1,800 images distributed across 6 defect types: *crazing*, *inclusion*, *patches*, *pitted surface*, *rolled-in scale*, and *scratches*).
* **The Challenge:** The dataset has no labeled "normal" images. To build a robust binary classifier, the pipeline automatically extracts **defective crops** centered on annotated defect bounding boxes and extracts **non-defective (normal) crops** from safe regions that do not overlap with any bounding boxes.

### 2. Feature Extraction (HOG)
* **Preprocessing:** Images are converted to grayscale and resized to a standard $96 \times 96$ pixels resolution.
* **HOG Parameters:** We extract the shape and directional structures using local gradient orientations. In the parameter sweep, we search across different cell sizes ($4 \times 4$, $8 \times 8$, $16 \times 16$ pixels) and orientations ($6, 9, 12$).
* **Hyperparameter Selection:** On the validation set, the $8 \times 8$ cell size with **12 orientations** yielded the highest $F_1$-score ($0.9236$) and Accuracy ($87.74\%$).

### 3. Binary Classification Results
After selecting the optimal HOG configuration, three classifiers are trained on the combined training and validation data, then evaluated on the untouched test set:
* **HOG + SVM (RBF):** Achieved an accuracy of **$82.63\%$** with an $F_1$-score of **$0.8927$** on the test set. It offers a solid balance with high recall ($95.56\%$) for defects.
* **HOG + Random Forest:** Achieved a competitive accuracy of **$81.79\%$** and $F_1$-score of **$0.8922$**. It got exceptional defect Recall ($99.63\%$), but suffered on Non-defective precision.
* **HOG + KNN:** Struggled compared to the others, yielding an accuracy of only **$61.90\%$**.

### 4. Robustness and Deployment
* **Robustness:** Testing under simulated environmental perturbations (such as brightness shifts, Gaussian noise, rotation, and blurring) showed that the SVM model remains highly robust. The metric shifts remain relatively low, meaning the model can withstand sensor or lighting imperfections in factory settings.
* **Tiled Inspection:** For full-size production images ($200 \times 200$), the module slides a crop window across the surface. If any region exceeds the classification probability threshold, the entire product is labeled **REJECT**; otherwise, it is **ACCEPTED**.
* **Multi-Class Extension:** Finally, the notebook expands the SVM and Random Forest models to perform multi-class defect classification (identifying the specific defect out of the 6 types). The HOG + SVM setup achieves an impressive **$87.78\%$ macro accuracy** for classification.