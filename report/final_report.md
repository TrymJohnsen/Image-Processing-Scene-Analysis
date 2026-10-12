# Image Processing and Feature Analysis

## 1. Introduction

This report investigates Fourier processing, principal component analysis (PCA), Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), blob detection and contour detection for IT3212 Assignment 2 [1]. We examine how these methods represent image information, how parameters change their results, and where their outputs support or limit interpretation.

## 2. Datasets and Experimental Setup

We use the Intel Image Classification dataset supplied through Blackboard. Its six outdoor scene categories include buildings, forests, glaciers, mountains, sea and streets. Different image types are selected to expose differences in frequency content, texture and shape.

Fourier experiments use `buildings/1001.jpg` at 150 × 150 pixels. PCA uses 200 randomly selected building images, with seed 42. HOG uses `sea/10399.jpg`, `buildings/1001.jpg`, `street/10019.jpg` and `forest/10037.jpg`, resized to 128 × 128. Blob and contour detection share four original 150 × 150 images: `sea/10399.jpg` (boat), `buildings/10310.jpg` (windows), `street/10015.jpg` (lamps) and `sea/1021.jpg` (rocks).

LBP combines the forest image with scikit-image's `brick` texture, originally from CC0Textures under CC0, and a face crop from `astronaut`, NASA's photograph of Eileen Collins [4]. The crop uses rows 10:210 and columns 130:330. Processing uses grayscale; detection overlays retain colour. PCA and HOG intensities are scaled to [0, 1]; Fourier errors use [0, 255].

## 3. Fourier Transform

### 3.1 Frequency spectrum

The spatial image stores intensities at pixel locations. Its two-dimensional discrete Fourier transform (DFT), computed with NumPy's `fft2`, represents spatial frequency components. `fftshift` moves zero frequency to the centre; plotting `log(1 + |F|)` makes weaker magnitudes visible [2].

Figure 1 concentrates energy near the centre, corresponding to gradual intensity changes and large structures. Its horizontal and vertical frequency patterns are consistent with repeated floors, windows and columns. The magnitude spectrum describes frequency strength and direction, but does not directly locate individual objects.

![Building and Fourier magnitude spectrum](../results/fourier/fourier_spectrum.png)

*Figure 1. Original building image and centred logarithmic DFT magnitude.*

### 3.2 Low-pass filtering

We add Gaussian noise with mean 0, standard deviation 25 and seed 42, clipping intensities to [0, 255]. Circular masks retain frequencies within radii 10, 20, 35, 50 and 70. Removing higher frequencies reduces noise, but also removes genuine edges. Figure 2 and Table 1 compare inverse-transform reconstructions with the clean original. MSE averages squared pixel differences after clipping filtered images.

![Low-pass comparisons](../results/fourier/low_pass_comparison.png)

*Figure 2. Clean image, fixed noisy image and five low-pass reconstructions.*

*Table 1. Low-pass MSE against the clean original.*

| Image/filter | MSE |
|---|---:|
| Noisy, unfiltered | 596.65 |
| Radius 10 | 1412.22 |
| Radius 20 | 954.71 |
| Radius 35 | 724.83 |
| Radius 50 | 563.39 |
| Radius 70 | 492.88 |

Radius 10 heavily blurs windows and railings. Radii 20 and 35 restore progressively more structure. All three look smoother than the noisy image while increasing MSE: the lost building detail outweighs the noise reduction. Radii 50 and 70 preserve finer edges and improve MSE, although noise remains visible. Radius 70 is best among the tested values. The useful cutoff depends on the image, noise level and desired balance between smoothing and detail preservation.

### 3.3 High-pass filtering

Complementary circular masks with radii 10, 20 and 35 suppress low frequencies in the clean image. Figure 3 displays absolute inverse-transform responses on a shared scale. Smooth regions and overall brightness disappear because they depend mainly on low frequencies.

![High-pass comparisons](../results/fourier/high_pass_comparison.png)

*Figure 3. Absolute high-pass responses for three cutoff radii.*

Radius 10 retains broad outlines and texture. At 20, windows, columns and overhead wires stand out; at 35, fine transitions dominate and wider structures weaken. These are edge-like responses, rather than normally sharpened photographs. High-pass filtering would also retain high-frequency noise. The abrupt cutoff can cause ringing near strong edges.

### 3.4 Fourier compression

We rank conjugate groups by coefficient magnitude and retain the strongest groups. Both members of each conjugate pair are kept, preserving the symmetry required for a real reconstruction [2]. Self-conjugate coefficients count once. Completing a pair can exceed the requested count by one, so Table 2 reports actual percentages. Symmetry and imaginary residuals are checked before using the real part; the largest residual is 7.21 × 10⁻¹⁴ intensity units.

![Fourier compression comparisons](../results/fourier/compression_comparison.png)

*Figure 4. Reconstructions using increasing proportions of Fourier coefficients.*

*Table 2. Fourier reconstruction error and total-to-retained coefficient ratio.*

| Requested | Actual retained | MSE | Coefficient ratio |
|---:|---:|---:|---:|
| 1% | 1.0000% | 1118.34 | 100.00:1 |
| 5% | 5.0044% | 613.75 | 19.98:1 |
| 10% | 10.0000% | 395.36 | 10.00:1 |
| 20% | 20.0044% | 206.67 | 5.00:1 |
| 50% | 50.0044% | 46.48 | 2.00:1 |
| 100% | 100.0000% | 0.00 | 1.00:1 |

Figure 4 remains recognisable at 1%, but strong blur and line patterns replace fine details. At 5–10%, floors and windows become clearer. Approximately 20% preserves major structure with fivefold coefficient reduction; 50% recovers much more edge detail. MSE decreases throughout, reaching numerical round-off at full retention. Compression MSE uses unclipped reconstructions; only the displayed images are clipped. These ratios count coefficients, not compressed file bytes: values, positions, precision and encoding overhead also require storage.

## 4. Principal Component Analysis

Each building image is converted to grayscale, resized to 64 × 64 and divided by 255. Flattening produces a 200 × 4096 matrix, with images as rows and pixels as columns. Our NumPy implementation subtracts the pixel-wise mean to obtain centred data, Xc.

Pixel covariance is C = XcᵀXc / (n − 1). Instead, we diagonalize G = XcXcᵀ / (n − 1), a 200 × 200 matrix rather than 4096 × 4096. This reduces computation when images are far fewer than pixels. Its eigenvectors are mapped into pixel space and normalized, yielding the principal directions of C. Eigenvalues and corresponding directions are sorted in descending order. Projection onto the first k directions gives k coefficients per image; reconstruction multiplies by those directions and restores the mean. The functions `fit_pca`, `project` and `reconstruct` implement this pipeline.

Figure 5 compares reconstructions; Figure 6 plots cumulative explained variance. Table 3 reports MSE over all 200 images using unclipped reconstructions; display values are clipped to [0, 1].

![PCA reconstruction comparison](../results/pca/reconstruction_comparison.png)

*Figure 5. Three training images reconstructed with increasing numbers of components.*

![PCA cumulative explained variance](../results/pca/cumulative_explained_variance.png)

*Figure 6. Cumulative explained variance, with the selected 90% criterion marked.*

*Table 3. PCA variance, reconstruction error and per-image dimensionality reduction.*

| k | Explained variance (%) | MSE | 4096/k |
|---:|---:|---:|---:|
| 1 | 23.0208 | 0.04540999 | 4096.00:1 |
| 5 | 50.4459 | 0.02923193 | 819.20:1 |
| 10 | 59.2781 | 0.02402183 | 409.60:1 |
| 25 | 72.1147 | 0.01644952 | 163.84:1 |
| 50 | 82.3081 | 0.01043643 | 81.92:1 |
| 85 | 90.0621 | 0.00586236 | 48.19:1 |
| 100 | 92.3104 | 0.00453607 | 40.96:1 |
| 150 | 97.3965 | 0.00153580 | 27.31:1 |
| 199 | 100.0000 | 0.00000000 | 20.58:1 |

At k = 1, only broad intensity structure survives. At 10, coarse building layouts return; at 50, buildings are recognisable but blurred. Increasing to 85 and 100 restores more structure while still losing fine texture. Gains diminish as later components capture less variance.

The computed minimum reaching 90% variance is k = 85, a selected compromise using a 90% explained-variance criterion. It retains 90.0621%, with MSE 0.00586236 and approximately 48.19:1 dimensionality reduction per image. This criterion makes the choice explicit without claiming universal optimality.

Fitting and evaluation use the same 200 images. Near-zero MSE with all 199 meaningful components therefore shows training-set reconstruction, not generalization. The 4096/k ratio excludes the shared PCA basis and mean image, which must also be stored. It is not complete storage compression.

## 5. HOG Features

Our script uses scikit-image HOG [3]. Gradient magnitudes weight local orientation histograms within cells. Overlapping blocks normalize neighbouring cells using L2-Hys, reducing sensitivity to local contrast. Concatenating normalized histograms produces the descriptor. The displayed Sobel magnitude is a separate visualization; HOG computes its own gradients internally.

Figure 7 compares four scenes using nine orientation bins, 8 × 8-pixel cells and 2 × 2-cell blocks. The sea has relatively sparse strong structure around the horizon and boat, although weak water texture still contributes. The building has repeated horizontal and vertical patterns. The street combines vertical façades with perspective diagonals. The forest has dense, irregular responses from branches, trunks and ground texture.

![HOG sea scene](../results/hog/simple_sea_comparison.png)

![HOG building scene](../results/hog/building_comparison.png)

![HOG street scene](../results/hog/street_comparison.png)

![HOG forest scene](../results/hog/forest_comparison.png)

*Figure 7. Sea, building, street and forest, from top to bottom. Each row shows grayscale input, Sobel magnitude and HOG visualization.*

Summing normalized descriptor values gives relative HOG strength, not percentages of physical edges. The sea and building place 29.9% and 23.5% respectively in the 90° gradient bin. The street places 36.4% in the bins nearest 0° and 180°; forest shares range from 10.1% to 13.2%. Gradient directions are perpendicular to image edges.

Figure 8 and Table 4 vary one baseline parameter at a time on the building.

![HOG parameter comparison](../results/hog/parameter_comparison.png)

*Figure 8. Cell size, block size and orientation-bin experiments on the building.*

*Table 4. HOG descriptor lengths for 128 × 128 inputs.*

| Configuration | Bins | Cell pixels | Block cells | Values |
|---|---:|---|---|---:|
| Baseline | 9 | 8 × 8 | 2 × 2 | 8100 |
| Smaller cells | 9 | 4 × 4 | 2 × 2 | 34596 |
| Larger cells | 9 | 16 × 16 | 2 × 2 | 1764 |
| Smaller blocks | 9 | 8 × 8 | 1 × 1 | 2304 |
| Larger blocks | 9 | 8 × 8 | 3 × 3 | 15876 |
| Fewer bins | 6 | 8 × 8 | 2 × 2 | 5400 |
| More bins | 12 | 8 × 8 | 2 × 2 | 10800 |

Smaller cells retain finer spatial detail; larger cells merge local patterns. More bins refine directional resolution. Larger blocks change normalization context and descriptor length, but not the displayed cell-histogram pattern, which precedes block normalization. Overlapping blocks repeat cell information, so descriptor growth does not imply proportionally more information. More spatial or directional detail generally costs a larger descriptor and more computation; descriptor length alone does not measure usefulness.

## 6. Local Binary Patterns

Our NumPy function `compute_lbp` compares each centre pixel with its eight direct neighbours. A neighbour at least as bright sets a 1; otherwise it sets a 0. Bits run clockwise from the top-left neighbour, assigned the lowest bit, producing codes 0–255. The separate function `lbp_histogram` counts these codes. Input images are 128 × 128; skipping borders produces 126 × 126 LBP images and 15,876 codes each.

Figure 9 displays the natural scene, regular texture and face, with their LBP outputs and histograms. The histograms describe the frequency of local texture patterns, but discard their positions. Uniform codes have at most two circular bit transitions. Table 5 includes their share and histogram entropy, which measures how broadly codes are distributed.

![Forest LBP and histogram](../results/lbp/forest_lbp.png)

![Brick LBP and histogram](../results/lbp/brick_lbp.png)

![Face LBP and histogram](../results/lbp/face_lbp.png)

*Figure 9. Forest, brick and face, from top to bottom: input, LBP image and normalized histogram.*

*Table 5. LBP texture summaries.*

| Image | Uniform patterns (%) | Entropy (bits) | Three most frequent codes |
|---|---:|---:|---|
| Forest | 70.49 | 6.9247 | 255 (7.74%), 0 (6.68%), 60 (1.72%) |
| Brick | 86.00 | 5.7860 | 255 (16.57%), 30 (4.32%), 2 (3.71%) |
| Face | 84.71 | 6.3520 | 255 (9.54%), 135 (3.92%), 240 (3.45%) |

Brick has the most concentrated histogram, consistent with repeated straight joints and smooth surfaces after resizing. Forest uses all 256 codes and has the highest entropy, reflecting varied neighbourhoods. The face lies between them: smooth shading mixes with curved facial boundaries. Code 255 dominates all three and means no neighbour is darker than the centre, occurring in flat regions and local minima.

LBP ignores contrast magnitude, so weak noise can change codes as readily as strong texture. It is invariant to monotonic intensity transformations that preserve neighbour comparisons, but clipping and quantization can change patterns. Resizing changes the texture scale, and basic LBP codes also depend on rotation.

## 7. Blob Detection

Our detector constructs a scale-normalized Laplacian of Gaussian response, σ²∇²(Gσ * I), for 11 scales from σ = 2 to 12. OpenCV supplies Gaussian blur, the Laplacian and `cv2.dilate` for spatial maxima [5]. We compare adjacent scales and retain responses equal to the neighbourhood maximum and strictly above threshold 0.2. Negating the response detects bright blobs; the original response detects dark blobs. Radius is estimated as √2σ.

Within each polarity, detections are ranked by response. We suppress weaker circles when overlap exceeds half the smaller circle's area. Figure 10 shows baseline detections on the original images; Table 6 reports sizes and example positions. Full positions, scales and responses are provided in [blobs.csv](../results/blob/blobs.csv). Coordinates start at the top-left, with x increasing rightwards and y downwards.

![Blob and contour comparison](../results/contour/blob_comparison.png)

*Figure 10. Boat, windows, lamps and rocks. Top: bright blobs in yellow and dark blobs in pink. Bottom: contours with a unique colour per contour.*

*Table 6. Baseline blob statistics; example centres identify the strongest response in each image.*

| Image | Count (bright/dark) | Radius min/median/max (px) | Example centre (x, y) |
|---|---|---|---|
| Boat | 1 (0/1) | 4.24 / 4.24 / 4.24 | (79, 81) |
| Windows | 14 (3/11) | 2.83 / 4.95 / 14.14 | (137, 44) |
| Lamps | 22 (21/1) | 2.83 / 2.83 / 2.83 | (80, 15) |
| Rocks | 23 (8/15) | 2.83 / 2.83 / 11.31 | (35, 133) |

The boat gives one dark response on its hull. Separate windows are detected, but the long window band produces multiple or oversized responses. Lamps all receive the smallest available radius, so their true sizes are not resolved. Rocks produce responses on shadows, gaps and bright surfaces rather than one detection per stone. Blobs are contrast structures, not semantic objects. Cropped structures and filter border handling can both influence border responses.

Figure 11 shows that lowering the rocks threshold to 0.15 increases detections from 23 to 71; raising it to 0.25 leaves three. Low thresholds accept weak texture, while high thresholds discard moderate-contrast structures. Restricting σ to 2–4 gives 22 detections; σ = 6–12 gives three broader responses. Disabling overlap pruning gives 24. Threshold and scale matter more than overlap for this example.

![Blob parameter comparison](../results/blob/parameter_comparison.png)

*Figure 11. Effects of threshold, scale range and overlap suppression on rocks.*

## 8. Contour Detection

We process the same four images with 5 × 5 Gaussian smoothing and Otsu thresholding. OpenCV's `findContours` uses `RETR_LIST` and `CHAIN_APPROX_NONE`; contours smaller than 15 pixels² are discarded. `RETR_LIST` returns outer boundaries and holes without hierarchical relationships [6]. Otsu chooses a global threshold minimizing weighted within-class intensity variance [7]; pixels above it become white and others black. Area and perimeter are measured with `contourArea` and `arcLength`. Figure 10 uses a unique visible colour for every contour; Table 7 summarizes measurements. Individual measurements are available in [contours.csv](../results/contour/contours.csv).

*Table 7. Baseline contour statistics. Areas are in pixels² and perimeters in pixels.*

| Image | Otsu threshold | Count | Area median/max | Perimeter median/max |
|---|---:|---:|---|---|
| Boat | 129 | 2 | 6724.5 / 13343.5 | 260.4 / 468.3 |
| Windows | 101 | 6 | 124.0 / 14060.0 | 43.7 / 946.5 |
| Lamps | 52 | 18 | 29.8 / 7539.0 | 22.7 / 1293.3 |
| Rocks | 117 | 17 | 40.0 / 10698.0 | 40.1 / 608.6 |

Every image includes a large background-region contour. Contour count is therefore not object count, and holes can add boundaries for one object. For the five separate windows, contours preserve rectangular outlines while blobs describe centres and approximate scales. The window band merges with a larger boundary. Contours follow the boat hull better than a circle, but also outline a broad region caused by uneven brightness.

For lamps, blobs separate many bright spots while thresholded lamps, shop fronts and illuminated ground merge. Neither method isolates individual stones: blobs mark contrast peaks and contours follow bright patches. Blobs are preferable for locating compact light sources; contours are preferable for measuring separated windows' shapes and areas.

Figure 12 shows rocks contours changing from 17 at baseline to 13 with threshold 80 and two with 160. Threshold changes can split or merge regions, so counts need not decrease monotonically. Removing blur gives 33 contours; 11 × 11 blur gives ten. Removing the area filter gives 63. Smoothing suppresses small irregularities but can merge structures, while area filtering removes small boundaries.

![Contour parameter comparison](../results/contour/parameter_comparison.png)

*Figure 12. Effects of threshold, smoothing and minimum area on rocks contours.*

## 9. Overall Discussion and Limitations

Parameter choices determine which information survives: frequency cutoffs select detail, PCA selects variance, and feature settings select spatial or contrast scales. More detections or descriptor values do not automatically mean better results. Our examples are small and deliberately selected, detection accuracy lacks annotated ground truth, and PCA evaluation uses its fitting images. Grayscale conversion and resizing discard information. MSE measures pixel error, not perceptual similarity or semantic accuracy. Fourier and PCA MSE values use different intensity scales and cannot be compared directly.

## 10. Conclusion

The experiments show why visual inspection and numerical evaluation should be used together. Useful representations preserve information relevant to the task, while their limitations must remain explicit. Low-level frequency, texture and shape measurements do not by themselves identify objects.

## 11. References

[1] IT3212. [Assignment 2](../IT3212%20-%20Assignment%202.pdf). Course assignment supplied with the project.

[2] NumPy developers. [Discrete Fourier Transform documentation](https://numpy.org/doc/stable/reference/routines.fft.html).

[3] scikit-image developers. [Histogram of Oriented Gradients API documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog).

[4] scikit-image developers. [Example image documentation](https://scikit-image.org/docs/stable/api/skimage.data.html), entries for [`brick`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.brick) and [`astronaut`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut).

[5] OpenCV developers. [Image Filtering documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html), including Gaussian blur, Laplacian, Sobel and dilation.

[6] OpenCV developers. [Structural Analysis and Shape Descriptors documentation](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html), including contour retrieval, area and perimeter.

[7] OpenCV developers. [Image Thresholding tutorial](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html), including Otsu's method.

## 12. AI Declaration

Generative AI tools were used for code review, debugging, report drafting, language refinement and discussion of possible improvements. Suggestions and report claims were checked against the code, experimental results and assignment requirements. The authors are responsible for reviewing the final submission and take responsibility for the submitted work.
