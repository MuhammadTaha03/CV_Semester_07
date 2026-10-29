# Lab Mid Report: HOG vs SIFT on Blood-Cell Images

**Name:** Taha  
**Registration number:** 055  
**Course:** Computer Vision Lab

---

## 1. Aim

The aim of this lab was to build two classical (non-deep-learning) image classifiers for blood cells and find out which one is more reliable when images are **rotated** or **noisy**.

- **Pipeline 1:** HOG features + LinearSVC
- **Pipeline 2:** SIFT features (fixed grid of keypoints) + LinearSVC

## 2. My personal settings

My registration number is 055, so the rules gave me these settings:

| Setting | Rule | My value |
|---|---|---|
| Random seed | Last two digits | **55** |
| HOG cell size | Last digit 5 is odd | **16 x 16** |
| SIFT grid | Last digit 5 is in the range 5 to 9 | **6 x 6 (36 keypoints)** |

The other settings were the same for everyone: BloodMNIST (size 64), four classes (eosinophil, monocyte, neutrophil, platelet), grayscale images resized to 128 x 128, HOG with 9 orientations and 2 x 2 cells per block, LinearSVC with C chosen from {0.1, 1, 10} on the validation set, a rotation of 20 degrees, and Gaussian noise with sigma = 0.05.

---

## 3. Q1: Data setup

**What was done:** I loaded the official BloodMNIST train, validation and test splits and kept only the four required classes. From each class I randomly picked 100 training, 30 validation and 30 test images, using seed 55 so the choice is repeatable. Then I converted every image to grayscale and resized it to 128 x 128.

**What happened:**

| Split | Images | Per class |
|---|---|---|
| Train | 400 | 100 |
| Validation | 120 | 30 |
| Test | 120 | 30 |

All four classes are perfectly balanced in every split. After preparation, the image arrays have shape (400, 128, 128), (120, 128, 128) and (120, 128, 128). I also plotted four example images per class to check that the labels were correct.

---

## 4. Q2: HOG + SVM

### How HOG works (in simple words)
HOG cuts the image into small squares called cells. In each cell, it counts how many edges point in each of 9 directions. Then it groups the cells into blocks and normalises each block, so the features are less affected by lighting changes.

### Feature length (hand calculation)
- 128 / 16 = 8 cells per side.
- Blocks of 2 x 2 cells moving by one cell give 8 - 2 + 1 = 7 blocks per side, so 7 x 7 = 49 blocks.
- Each block has 2 x 2 x 9 = 36 values.
- Total = 49 x 36 = **1764** features per image. The program printed the same number.

### HOG pictures
I drew the HOG picture of one image from each class. The bright lines show the strongest edge directions in each cell, and they follow the outline of the cell and its nucleus.

### Training result
I trained a LinearSVC for C = 0.1, 1 and 10. All three gave the same validation accuracy of 0.6833, so the first one, **C = 0.1**, was kept. This probably means the training data is easy to separate with so many features, so C changed little (this reason was not tested).

| Metric (validation) | Value |
|---|---|
| Accuracy | **0.6833** |
| Macro-F1 | **0.6855** |

![HOG + SVM confusion matrix](images/q2_hog_confusion_matrix.png)

**Reading the confusion matrix:**
- Platelet was almost always correct (29 of 30).
- Eosinophil was the weakest class (14 of 30 correct).
- The biggest confusion was between eosinophil and monocyte (9 + 8 = 17 mistakes).

---

## 5. Q3: SIFT + SVM

### Part 1: Normal SIFT
When SIFT finds keypoints by itself, the number is different for each image. On the training images:

| Statistic | Value |
|---|---|
| Average keypoints per image | 72.52 |
| Minimum | 12 |
| Maximum | 188 |
| Images with zero keypoints | 0 |

Each keypoint gives a 128-number descriptor, so one image could produce 12 descriptors and another 188. An SVM needs the same number of features for every image, so these varying-length features cannot be used directly.

### Part 2: The fix with a fixed grid
Instead of letting SIFT choose the keypoints, I placed a regular 6 x 6 grid of 36 keypoints (size 32) over every 128 x 128 image and computed one descriptor at each point. Joining them gives the same length for every image:

36 grid points x 128 values = **4608** features per image. The program printed the same number.

### Part 3: Training result
Again, all three values of C gave the same validation accuracy (0.7000), so **C = 0.1** was kept.

| Metric (validation) | Value |
|---|---|
| Accuracy | **0.7000** |
| Macro-F1 | **0.6947** |

![SIFT + SVM confusion matrix](images/q3_sift_confusion_matrix.png)

**Reading the confusion matrix:**
- Platelet was again the easiest (28 of 30 correct).
- Eosinophil was again the weakest (14 of 30 correct).
- The biggest confusion was between eosinophil and neutrophil (8 + 5 = 13 mistakes). The next pair, eosinophil and monocyte, had 7 + 5 = 12.

### Comparison on clean validation images
SIFT (0.7000) and HOG (0.6833) are very close. On clean images neither is clearly better.

---

## 6. Q4: Robustness

### How the test was done
I froze both models (no re-training and no new choice of C). On the 120 test images, I made three versions: the **clean** images, the images **rotated by 20 degrees**, and the images with **Gaussian noise** (sigma = 0.05 on a 0 to 1 scale). The rotation and noise were applied to the images **before** feature extraction, and then the same models predicted the classes.

### Prediction
*(To be written by me: what I believed before running the test.)*

### Results

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.7000 | 0.6167 | 0.2667 |
| SIFT + SVM | 0.6583 | 0.3167 | 0.6250 |

Accuracy drops compared with clean images:

| Model | Drop under rotation | Drop under noise |
|---|---|---|
| HOG + SVM | 0.0833 | 0.4333 |
| SIFT + SVM | 0.3417 | 0.0333 |

### What happened

- **Rotation:** HOG lost only a little (0.7000 to 0.6167), but SIFT lost a lot (0.6583 to 0.3167). Normal SIFT is rotation invariant because it measures the direction of each keypoint. In this lab, however, the keypoints were placed on a fixed grid, so no direction was estimated, and the descriptors change when the image turns.
- **Noise:** HOG collapsed (0.7000 to 0.2667), which is almost the chance level for four classes (0.25), while SIFT barely changed (0.6583 to 0.6250). HOG is built from edge gradients, and noise creates many small random edges. A likely reason SIFT stayed stable is that it smooths the image and collects information from a larger patch (size 32), so random noise is partly averaged out. This reason was not tested.
- **Main result:** neither method is best in every case. HOG was more robust to rotation, and SIFT was more robust to noise.
- **Caution:** the test set has only 120 images, so one image is 0.83%. Small differences, such as 0.7000 against 0.6583 on clean images, are weak evidence. The large drops are much more reliable.

### Was my prediction correct?
*(To be written by me after comparing my prediction with the table above.)*

---

## 7. Q5: Short answers

**(a) Why does HOG use block normalisation?**  
Block normalisation makes the features less sensitive to changes in lighting and contrast, because each block is scaled by its own gradient strength. Without it, the same edge could give very different values in a bright image and in a dark image.

**(b) Why is it wrong to choose C using the test set?**  
The test set must stay unseen to give an honest estimate on new data. If C is chosen using the test result, the model is tuned to that data and the final accuracy becomes too optimistic. C must be chosen on the validation set, and the test set is used only once, at the end.

**Most confused classes:**  
HOG confused **eosinophil and monocyte** most (17 mistakes in total), and SIFT confused **eosinophil and neutrophil** most (13 mistakes in total). Eosinophil appears in both pairs, while platelet was rarely confused. In grayscale, colour is lost, such as the reddish granules of the eosinophil, so the three white blood cells all look like a dark lobed nucleus on a pale background, and only shape and texture are left to tell them apart.

---

## 8. Conclusion

- Both pipelines worked at a similar level on clean images (about 0.68 to 0.70 validation accuracy).
- Platelet was easy to recognise, and the three white blood cell types were often mixed up.
- Under rotation, HOG was more reliable. Under noise, the fixed-grid SIFT was more reliable.
- Normal SIFT could not be used directly with an SVM because the number of keypoints changes from image to image, and the fixed grid solved this.

---

## 9. AI log

| Prompt (one line) | What I changed | How I checked the result |
|---|---|---|
| *(to be filled by me)* | | |
