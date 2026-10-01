# Lab Assignment: Skin Lesion Boundary Detection Using Canny Edge Detection

**Dataset:** HAM10000 (5 images selected automatically)
**Tools:** Python, OpenCV, NumPy, Pandas, Matplotlib

---

## 1. Method Summary

| Step | What was done |
|------|---------------|
| Task 1 | Loaded 5 skin lesion images from HAM10000 and displayed the originals. |
| Task 2 | Converted each image to grayscale and applied a Gaussian filter (5x5 kernel, sigma = 1.4). |
| Task 3 | Applied Canny edge detection with three threshold settings and displayed the three edge maps. |
| Task 4 | Compared the edge maps and selected the threshold that gave the clearest boundary (scored by contour size, compactness, and edge clutter). |
| Task 5 | Closed gaps in the edges with morphological closing, found the external contours, kept the largest contour as the lesion, and drew it on the original image. |
| Task 6 | Calculated the area (pixels inside the filled contour) and the perimeter (contour length). |

Final visualization for each image: **Original -> Grayscale -> Gaussian Filter -> Canny -> Lesion Boundary**.

---

## 2. Results Table

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|-------|-------------|-------------|---------------|--------------------|
| Image 1 | Gaussian 5x5, sigma=1.4 | Canny (50-100) | 460 | 251.7 |
| Image 2 | Gaussian 5x5, sigma=1.4 | Canny (50-100) | 340 | 257.4 |
| Image 3 | Gaussian 5x5, sigma=1.4 | Canny (50-100) | 55 | 85.2 |
| Image 4 | Gaussian 5x5, sigma=1.4 | Canny (50-100) | 0 | 0.0 |
| Image 5 | Gaussian 5x5, sigma=1.4 | Canny (50-100) | 103 | 50.9 |

**Note:** The areas are small (0-460 pixels) compared with the image size (about 196,000 pixels). This means the detected boundaries are only partial and do not cover the whole lesion. No boundary was found for Image 4.

---

## 3. Final Comparison

### Numeric metrics (average of 5 images)

| Method | Noise (fragments, lower = better) | Edge Quality (continuity) | Boundary (Dice) | Overall Score |
|--------|-----------------------------------|---------------------------|-----------------|---------------|
| Original + Sobel | 2729.8 | 0.377 | 0.424 | 0.283 |
| Original + Canny | 2.2 | 0.517 | 0.014 | 0.460 |
| Average + Sobel | 601.4 | 0.639 | 0.694 | 0.703 |
| Average + Canny | 0.0 | 0.000 | 0.000 | 0.300 |
| Gaussian + Sobel | 736.6 | 0.630 | 0.682 | 0.681 |
| Gaussian + Canny | 0.0 | 0.000 | 0.000 | 0.300 |
| Median + Sobel | 841.4 | 0.540 | 0.658 | 0.633 |
| Median + Canny | 0.2 | 0.200 | 0.000 | 0.360 |

**How the metrics were calculated:**
- **Noise:** number of small, isolated edge fragments (fewer is better).
- **Edge quality:** edge continuity (the largest connected edge divided by all edge pixels).
- **Boundary detection:** Dice score between the detected lesion mask and a reference mask made with Otsu thresholding. The reference mask is an approximation, not true ground truth.
- **Overall score:** weighted mix of the three metrics.

### Final comparison table

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|--------|----------------|--------------|--------------------|---------------------|
| Original + Sobel | Poor | Fair | Fair | Poor |
| Original + Canny | Fair | Fair | Fair | Fair |
| Average + Sobel | Fair | Good | Good | Good |
| Average + Canny | Good | Poor | Poor | Poor |
| Gaussian + Sobel | Poor | Good | Good | Good |
| Gaussian + Canny | Good | Poor | Poor | Poor |
| Median + Sobel | Poor | Good | Good | Good |
| Median + Canny | Good | Poor | Poor | Fair |

**Note:** The labels (Poor, Fair, Good) are relative. They rank the 8 methods against each other. "Good" noise handling for the Canny methods is misleading, because Canny found almost no edges at all.

### Discussion

Sobel with smoothing gave the best results in this experiment. Average + Sobel had the highest overall score (0.703), followed by Gaussian + Sobel (0.681) and Median + Sobel (0.633). These methods reached a Dice score of about 0.66-0.69. Smoothing reduced noise strongly: Original + Sobel had about 2,730 noisy fragments, while the smoothed versions had about 600-840.

Canny with the selected thresholds failed on the smoothed images (Dice 0.00). The lesion borders in HAM10000 are soft, so the gradient is weak, and the thresholds were too high.

---

## 4. Answers to the Questions

### Question 1: Why is Gaussian filtering applied before Canny detection?

Skin images contain noise, hair, and fine skin texture. Edge detectors react strongly to sudden changes in intensity, so this noise creates many false edges. Gaussian filtering smooths the image and removes the noise, but it keeps the main lesion border. In our results, smoothing reduced the number of noisy edge fragments from about 2,730 (original image) to about 737 (Gaussian filter) with Sobel. However, smoothing also lowers the edge strength, so the Canny thresholds must be chosen carefully.

### Question 2: How did the three Canny threshold settings affect the result?

Higher thresholds keep fewer edges.
- **50-100:** kept the most edges. Some were real lesion edges, but many were small fragments.
- **100-200:** removed most edges. Only very strong edges remained, and almost nothing was left on the smoothed images.
- **150-250:** removed almost all edges, so the edge maps were nearly empty.

The lesion borders are soft, so their gradient is below the high thresholds.

### Question 3: Which threshold produced the best lesion boundary?

The threshold **50-100** was selected for all five images. It is the best of the three tested settings because it keeps the most lesion edges. However, the detected areas were small (0-460 pixels), so it did not give a complete lesion boundary. Lower thresholds (for example 10-30 or 20-60) should be tested.

### Question 4: Why are edges useful for detecting skin lesions?

A lesion usually has a different colour and brightness from the normal skin around it. This creates a strong change in intensity at the lesion border. Edge detection finds this border and gives the shape, size, area, perimeter, and border irregularity of the lesion. These features are important for diagnosis, for example in the ABCD rule (Asymmetry, Border, Colour, Diameter).

### Question 5: What problems did you observe in detecting the lesion boundary?

- Canny with the 100-200 setting found almost no edges on the smoothed images.
- The detected boundaries were small fragments, not the full lesion, and Image 4 had no boundary (area = 0).
- Original + Canny had a Dice score of only 0.014, which means almost no overlap with the reference lesion.
- Without smoothing, Sobel produced a lot of noise (about 2,730 fragments per image).
- Soft, low-contrast borders made the edges weak and broken.
- Hair and skin texture in HAM10000 images can create false edges.

### Question 6: How could your method be improved?

- Use lower Canny thresholds (for example 10-30, 20-60) or automatic thresholds based on the image median.
- Use stronger morphological closing, then fill the holes inside the contour.
- Remove hair before filtering (black-hat filter and inpainting).
- Combine edge detection with Otsu thresholding or use other colour spaces such as LAB.
- Use advanced methods such as active contours, GrabCut, or deep learning (for example U-Net).

---

## 5. Conclusion

Smoothing the image before edge detection is important because it reduces noise. In this experiment, Sobel with Average or Gaussian smoothing gave the best lesion boundary (Dice about 0.68-0.69). Canny with the tested thresholds did not work well, because the lesion borders in HAM10000 are soft and the thresholds were too high. Using lower thresholds, hair removal, and better post-processing should improve the results.
