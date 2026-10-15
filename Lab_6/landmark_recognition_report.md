# Landmark Recognition Using the SIFT Algorithm

**Lab Assignment Report**
**Name:** ______________________  **Date:** ______________________
**Source file:** `landmark_recognition_sift_colab.ipynb` (Python 3, OpenCV 5.0.0, Google Colab)

---

## 1. Objective

The goal of this lab is to build a computer vision application that recognizes famous landmarks from a query photograph. The system compares the query image with a database of reference images using the Scale-Invariant Feature Transform (SIFT). It must:

1. Identify the landmark even when the query image is rotated, resized, or darker.
2. Locate the landmark inside the query image.
3. Reject images of landmarks that are not in the database.

The use case is a tourism information center where visitors upload a photo and receive the name of the landmark.

---

## 2. Dataset (Task 1)

The database was created manually and stored in Google Drive (folder `Wonders of world`). Each landmark has its own sub-folder, and the folder name is used as the label.

| Landmark | Images in database | Hold-out test image |
|---|---|---|
| colosseum | 4 | 1 |
| eiffel_tower | 4 | 1 |
| great_wall | 4 | 1 |
| statue_of_liberty | 4 | 1 |
| taj_mahal | 3 | 1 |
| **Total** | **19** | **5** |

**Hold-out strategy.** The last image of every landmark is removed from the database and used only for testing. In this way the system is evaluated on photos it has never seen. Each landmark has 4 to 5 photos in total, so the hold-out photo is a different view of the same landmark.

---

## 3. Method

### 3.1 Pipeline

```
Reference images --> grayscale --> SIFT keypoints + descriptors --> database
Query image      --> grayscale --> SIFT keypoints + descriptors
                                        |
                  BFMatcher (L2) + KNN (k=2) + Lowe's ratio test
                                        |
              Best reference image per landmark --> ranking
                                        |
              Homography + RANSAC (geometric verification)
                                        |
              Decision: RECOGNIZED (with polygon) or REJECTED
```

### 3.2 Key concepts

- **SIFT.** SIFT finds keypoints in a scale-space (Difference of Gaussians), so it is robust to **scale** changes. Each keypoint gets a dominant orientation, which makes it robust to **rotation**. The descriptor has 128 values (4×4 cells × 8 gradient bins) built from normalized gradients, which makes it robust to **brightness** changes.
- **Brute-Force Matcher (`cv2.NORM_L2`).** Every query descriptor is compared with every reference descriptor using the Euclidean distance.
- **KNN matching (k=2).** For each query descriptor, the two nearest reference descriptors are returned.
- **Lowe's ratio test.** A match is kept only if `best distance < 0.75 × second-best distance`. This removes ambiguous matches.
- **Homography and RANSAC.** A homography maps the planar reference image onto the query image. RANSAC repeatedly fits a homography to 4 random matches and keeps the model with the most **inliers**, so wrong matches are ignored. The four reference corners are projected to draw a polygon around the landmark.

### 3.3 Decision rule

The assignment states that an image must not be classified only by the largest match count. The system therefore accepts a landmark only if **all three** conditions are true:

| Check | Parameter | Value |
|---|---|---|
| Enough reliable matches | `MIN_GOOD_MATCHES` | 25 |
| Enough RANSAC inliers | `MIN_INLIERS` | 12 |
| Best candidate clearly beats the runner-up | `MIN_MARGIN` | 1.3 × |

In addition, the projected polygon must be convex and have a reasonable size; otherwise a "localization unreliable" message is shown and no polygon is drawn.

Other settings: Lowe ratio = 0.75, RANSAC reprojection threshold = 5 px, images resized so the longest side is at most 800 px.

For each landmark only the **best** reference image is counted (maximum, not sum), so a landmark with more photos does not gain an unfair advantage.

---

## 4. Implementation (Tasks 2 to 5)

The notebook is organized in small functions, each with one job:

| Function | Purpose |
|---|---|
| `find_database_folders()` | Finds the landmark folders in Google Drive automatically |
| `load_image()`, `to_gray()` | Loading, resizing, grayscale conversion |
| `extract_sift()` | SIFT keypoints and descriptors (used for both reference and query images) |
| `build_database()` | Builds the database and separates the hold-out images |
| `match_with_ratio_test()` | KNN (k=2) matching and Lowe's ratio test |
| `score_landmarks()` | Counts reliable matches and ranks landmarks |
| `localize()` | Homography with RANSAC, polygon projection, sanity checks |
| `recognize()` | Full pipeline and final decision for one image |
| `show_results()` | Displays query, keypoints, matches, and localization result |
| `run_evaluation()` | Runs the Task 6 tests and saves a CSV table |

Keypoint counts per image (database construction, Task 2) ranged from **164** to **4383**. For example, `colosseum/images (2).jpg` produced 4383 keypoints, while the Eiffel Tower image `images (4).jpg` produced only 439.

Output images (keypoints, matches, localization) and the result tables are saved in `MyDrive/landmark_results/`.

---

## 5. Experimental Results (Task 6)

### 5.1 Single query example

Query: hold-out Colosseum image.

| Rank | Landmark | Reliable matches |
|---|---|---|
| 1 | colosseum | **137** |
| 2 | statue_of_liberty | 23 |
| 3 | great_wall | 19 |
| 4 | eiffel_tower | 17 |
| 5 | taj_mahal | 14 |

Query keypoints: 1456. RANSAC inliers: 94. Processing time: 1.77 s.
**Decision: RECOGNIZED, colosseum.** The best candidate has about six times more matches than the runner-up.

### 5.2 Test conditions

| Test | Predicted | Keypoints | Good matches | Inliers | Decision | Time (s) | Correct |
|---|---|---|---|---|---|---|---|
| Original (database image) | colosseum | 2556 | 2556 | 2556 | RECOGNIZED | 4.663 | Yes |
| Rotated 30° | colosseum | 2708 | 1622 | 1588 | RECOGNIZED | 3.736 | Yes |
| Resized 50% | colosseum | 807 | 603 | 598 | RECOGNIZED | 1.036 | Yes |
| Reduced brightness | colosseum | 919 | 857 | 854 | RECOGNIZED | 1.261 | Yes |
| Different viewpoint (hold-out) | colosseum | 1456 | 137 | 94 | RECOGNIZED | 1.941 | Yes |
| Unknown landmark (taj_mahal removed from database) | unknown | 1059 | 19 | 2 | REJECTED | 2.276 | Yes |

**Notes on these tests**

- The "Original" test uses an image that is itself in the database, so every keypoint matches itself (2556 of 2556). It confirms that the pipeline works but is not a real challenge.
- The rotation, resize, and brightness tests start from the same image. The system recognized all of them, with high inlier counts.
- For the unknown-landmark test, the Taj Mahal was temporarily removed from the database and its hold-out photo was used as the query. The system correctly rejected it (19 matches, only 2 inliers).

### 5.3 Test on all unseen (hold-out) photos

| True landmark | Predicted | Good matches | Inliers | Decision | Time (s) | Correct |
|---|---|---|---|---|---|---|
| colosseum | colosseum | 137 | 94 | RECOGNIZED | 2.444 | Yes |
| eiffel_tower | unknown | 40 | 38 | REJECTED | 0.789 | No |
| great_wall | unknown | 29 | 7 | REJECTED | 2.818 | No |
| statue_of_liberty | unknown | 6 | 0 | REJECTED | 0.213 | No |
| taj_mahal | unknown | 26 | 11 | REJECTED | 1.287 | No |

**Accuracy on unseen photos: 1 out of 5 (20%).**
No image was assigned to a **wrong** landmark. All four failures were rejections.

---

## 6. Discussion

### 6.1 What worked well

- **Invariance to image transformations.** Under rotation (30°), 50% downscaling, and reduced brightness, the Colosseum was recognized with hundreds of matches and an inlier ratio above 97%. This agrees with the theory: SIFT is designed to be robust to rotation, scale, and illumination.
- **Strong separation when the match is real.** For the Colosseum, the best landmark had 137 matches and the runner-up had only 23. Random matches from other landmarks stay low (14 to 23), which shows that the ratio test removes most ambiguous matches.
- **Safe failure mode.** The system never named a wrong landmark. The strict decision rule trades recall for precision, which is acceptable for an information system that should not give misleading answers.
- **Correct rejection of an unknown landmark.** Only 2 inliers survived RANSAC, so the geometric check clearly separated the unknown image.

### 6.2 What did not work well

Four of five hold-out photos were rejected. The reasons below are taken from the numbers in the table. The notebook did not print the rejection reason in the hold-out table, so the Eiffel Tower case is an interpretation.

| Landmark | Observation | Likely cause |
|---|---|---|
| statue_of_liberty | Only 6 matches, 0 inliers. The hold-out image has only **164 keypoints**. | The test photo is small or low in detail, so SIFT finds too few features. |
| great_wall | 29 matches but only 7 inliers (below 12). | Repeating wall and brick texture creates matches that are not geometrically consistent. |
| taj_mahal | 26 matches and 11 inliers, only **one inlier below** the threshold of 12. | A borderline case. The threshold is strict for this viewpoint. |
| eiffel_tower | 40 matches and 38 inliers (both above the thresholds) but still rejected. | Probably the margin check (runner-up too close) or the polygon sanity check. This should be confirmed by printing `res["reason"]`. |

### 6.3 Why the viewpoint test is hard

A homography is exact only for a **flat** object. Landmarks are three-dimensional, and a new viewpoint changes which surfaces are visible. Even for the successful Colosseum case, the ratio of matches to query keypoints dropped from about 93% (brightness test) to about 9% (viewpoint test: 137 of 1456).

### 6.4 Effect of the rotation test

The 30° rotation used the original image size, so the corners of the rotated image were cropped. This explains why the matches fell from 2556 to 1622, even though SIFT is rotation-invariant: part of the image content was lost.

### 6.5 Processing time

Processing time was between 0.2 s and 4.7 s per image on a Colab CPU. Time grows with the number of query keypoints, because brute-force matching compares every query descriptor with every database descriptor (19 images). The slowest run (4.66 s) was the image with the most keypoints (2556 in the original and 2708 in the rotated test).

---

## 7. Limitations

- **Small dataset.** The database has 19 images and the test set has only 5 hold-out photos. The 20% accuracy has a very large uncertainty, and the thresholds should not be considered tuned or reliable.
- **Planar assumption.** Homography works poorly for 3D objects under large viewpoint changes.
- **Repeating textures** (Great Wall, Colosseum arches) cause ambiguous matches.
- **Low-resolution or low-detail images** produce too few keypoints (Statue of Liberty hold-out: 164).
- **Scalability.** Brute-force matching becomes slow for large databases.
- **Hand-set thresholds** (25 matches, 12 inliers, margin 1.3) may not generalize to other photo conditions such as night, fog, crowds, or occlusion.
- **Evaluation design.** The "Original" test uses a database image, and the rotation, resize, and brightness tests are all based on one landmark (Colosseum). Other landmarks were tested only with one hold-out photo each.

---

## 8. Possible Improvements

1. Add more reference images per landmark (more viewpoints, distances, and lighting) and use higher-resolution photos.
2. Print the rejection reason for every test, then adjust thresholds based on a larger validation set, not on the five test images.
3. Verify the **top 3** candidates with RANSAC instead of only the top 1.
4. Use RootSIFT, or a FLANN matcher / vocabulary tree for faster search.
5. Use learned features (for example SuperPoint with LightGlue) or a deep landmark-retrieval model for large viewpoint changes.
6. Rotate images with an enlarged canvas in the rotation test so that no content is cropped.

---

## 9. Conclusion

The notebook implements the full SIFT-based landmark recognition pipeline: database creation, feature extraction, KNN matching with Lowe's ratio test, ranking, RANSAC localization, and a threshold-based decision. The system is very robust to rotation, scale, and brightness changes (Colosseum: 1622, 603, and 857 reliable matches) and correctly rejects an unknown landmark. On unseen photos taken from different viewpoints, however, it recognized only 1 of 5 landmarks, and rejected the other four without ever giving a wrong answer. The main causes are the small and uneven image database, the planar-object assumption of the homography, textured repeating patterns, and strict thresholds. With more reference images and a larger validation set, the performance should improve.

---

## 10. Files

| File | Description |
|---|---|
| `landmark_recognition_sift_colab.ipynb` | Complete source code |
| `Wonders of world/` (Google Drive) | Landmark image database (5 folders) |
| `landmark_results/evaluation.csv` | Task 6 result table |
| `landmark_results/heldout_accuracy.csv` | Hold-out test results |
| `landmark_results/*.jpg` | Screenshots: keypoints, matches, localization |
