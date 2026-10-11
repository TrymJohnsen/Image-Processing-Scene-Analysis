## Fourier Transform

### Frequency spectrum

The grayscale image is represented in the spatial domain as a grid of pixel intensities. The 2D Fourier transform represents the same image using spatial frequencies. Low frequencies describe gradual intensity changes and large-scale structure, while high frequencies describe fast changes such as edges and fine details. `fftshift` moves the low-frequency components to the center of the spectrum, which makes the result easier to read. A logarithmic magnitude is used because the Fourier coefficient magnitudes cover a large range and weaker components would otherwise be difficult to see.

Most of the energy is concentrated near the center of the spectrum. This shows that gradual intensity changes and large structures make up much of the building image. The spectrum also has a clear horizontal and vertical pattern. This is consistent with the repeated floors, railings, window rows and supporting columns, which create strong changes and repetition along the image axes. The spectrum describes the strength and direction of frequency content, but it does not show where individual objects are located.

### Low-pass filtering

Gaussian noise with mean 0 and standard deviation 25 was added using a fixed random seed. Cutoff radii of 10, 20 and 35 pixels were selected as increasingly permissive filters, making the balance between noise removal and retained detail easy to observe.

Radius 10 removes much of the noise, but it also removes most fine structure. The building remains recognizable, while windows, railings and sharp edges are heavily blurred. Radius 20 gives an intermediate result, with more building structure visible and much of the noise still reduced. Radius 35 preserves substantially more detail, but also keeps more noise. Noise and real image edges can both occupy high-frequency regions, so the filter cannot separate them perfectly.

### High-pass filtering

The same cutoff radii of 10, 20 and 35 pixels were used so that the low-pass and high-pass results can be compared directly. At radius 10, broad outlines and much of the building texture remain visible. At radius 20, smooth areas become darker and the edges of windows, railings, columns and overhead wires become clearer. At radius 35, fine details and sharp transitions dominate, while wider structures are less visible.

Edges are emphasized because quick changes between light and dark pixels contain high spatial frequencies. Removing low frequencies suppresses smooth regions and overall brightness, so the displayed absolute responses look like edge maps rather than naturally sharpened images. High-pass filtering can also make noise more visible. The hard circular cutoff may produce faint ringing near strong edges.

| Method | Cutoff radius | Main observed effect |
|---|---:|---|
| Low-pass | 10 | Strong smoothing; most noise and fine detail removed |
| Low-pass | 20 | More structure visible while much of the noise is reduced |
| Low-pass | 35 | Most structure retained, but more noise remains |
| High-pass | 10 | Broad outlines and texture remain visible |
| High-pass | 20 | Windows, railings, columns and wires stand out clearly |
| High-pass | 35 | Fine edges dominate and wider structures fade |

### Fourier compression

The compression experiment keeps the Fourier coefficients with the largest magnitudes and sets the rest to zero. The image is then reconstructed with the inverse Fourier transform.

| Retained coefficients | MSE | Theoretical compression ratio |
|---:|---:|---:|
| 1% | 1118.34 | 100:1 |
| 5% | 613.89 | 20:1 |
| 10% | 395.36 | 10:1 |
| 20% | 206.69 | 5:1 |
| 50% | 46.48 | 2:1 |
| 100% | 0.00 | 1:1 |

The reconstruction error decreases as more coefficients are retained. At 1%, the building is still recognizable, but the image is heavily blurred and has visible line patterns. At 5% and 10%, the floors, roof and window layout become clearer, although fine details are still missing. Keeping 20% preserves most major structure while reducing the coefficient count by a factor of five, making it a useful compromise in this experiment. At 50%, the reconstruction is visually close to the original and many fine edges have returned. Keeping 100% reproduces the original image with a reported MSE of 0.00.

The theoretical compression ratio only compares the total number of Fourier coefficients with the number retained. It is not the size of a compressed file. A real format would also need to store coefficient positions and values, as well as file information and encoding overhead.

## Principal Component Analysis

The PCA experiment used 200 images from the buildings category of the Intel training set, and PCA was fitted using all 200 images. This category was selected because no facial-emotion dataset was available locally. Using one category also gives a coherent set with recurring structures such as façades, roofs, windows and sky, while retaining variation in viewpoint, shape and lighting that PCA can model.

### PCA representation

Each image was converted to grayscale, resized to 64 × 64 pixels and normalized to the range [0, 1]. The images were then flattened so that each could be represented by one row of the data matrix. Before dimensionality reduction, each image therefore had 4096 pixel values, and the resulting matrix had 200 rows for images and 4096 columns for pixels. Normalization gives every pixel a consistent scale. The mean image is the pixel-wise average of all 200 images. It appears blurred because the buildings, roofs, sky regions and other structures are not spatially aligned across the dataset. The mean was subtracted before PCA so that the analysis describes variation around the average building image instead of overall brightness levels.

### Principal components

Covariance describes how intensity patterns vary together across this set of building images. Since there are far fewer images than pixels, covariance was computed in the 200 × 200 sample space instead of directly forming a 4096 × 4096 pixel covariance matrix. The resulting eigenvectors were mapped back into pixel space and normalized.

The image-space eigenvectors are directions of variation, while their eigenvalues measure how much variation each direction captures. Sorting components by decreasing eigenvalue places the most informative directions first. An image can then be projected onto the first *k* components to obtain *k* PCA coefficients.

### Reconstruction experiment

The reconstruction figure shows three representative examples from the 200 images used to fit PCA. Only three are shown to keep the comparison readable. One component retains only broad intensity structure, so the individual buildings are not identifiable. At *k* = 10, the coarse building layout returns, but edges remain heavily blurred. At *k* = 50, the buildings are recognizable but still visibly blurred. The *k* = 100 reconstructions are visually close to the originals, although some fine texture and edge sharpness are still reduced. With all 199 meaningful components, the training images are reconstructed with effectively zero MSE.

### Explained variance and reconstruction error

| k | Cumulative explained variance | MSE |
|---:|---:|---:|
| 1 | 0.2302 | 0.045410 |
| 5 | 0.5045 | 0.029232 |
| 10 | 0.5928 | 0.024022 |
| 25 | 0.7211 | 0.016450 |
| 50 | 0.8231 | 0.010436 |
| 100 | 0.9231 | 0.004536 |
| 150 | 0.9740 | 0.001536 |
| 199 | 1.0000 | 0.000000 |

Cumulative explained variance increases with *k* because each added component accounts for another direction of variation. MSE decreases because reconstruction takes place in a larger subspace and therefore discards less information. The largest early gains occur before *k* = 50, after which each additional component produces a smaller improvement.

For this experiment, *k* = 50 to 100 is a useful range rather than a universally optimal choice. At *k* = 50, the representation retains 82.31% of the variance and the buildings are recognizable, but the reconstructions remain visibly soft. At *k* = 100, 92.31% of the variance is retained, MSE falls to 0.004536, and the reconstruction is visually close to the original. The appropriate point within this range depends on whether stronger dimensionality reduction or finer image detail matters more.

### Dimensionality reduction

The reported pixel reduction values compare the original 4096 pixel values per image with *k* PCA coefficients. For example, *k* = 100 reduces the per-image representation by a factor of about 40.96. These are not actual file compression ratios because reconstruction also requires the shared PCA basis and mean image. Their storage cost and other file overhead would need to be included in a real compression calculation.

## HOG Features

The experiment used `buildings/1001.jpg`, `street/10019.jpg` and `forest/10037.jpg`. Each image was converted to grayscale and resized to 128 × 128 pixels. The Sobel gradient magnitude shows where intensity changes are strongest. In the building and street images, bright gradient responses follow roof lines, windows, doors and façade boundaries. The forest response is denser because trunks, branches, leaves and the ground create edges across most of the image.

HOG summarizes local edge directions by collecting gradient orientations within small cells. With the baseline settings of 9 orientations, 8 × 8 pixels per cell and 2 × 2 cells per block, the building produces repeated horizontal and vertical patterns. The street also has strong vertical edges, together with diagonals that follow the street perspective. The forest pattern is less regular, although the tree trunks still produce clear vertical responses.

The aggregated baseline HOG values support these visual differences. The building has its strongest response in the 90° gradient bin, which contains 23.5% of its total orientation strength. The street places 36.4% in the two bins nearest 0° and 180°, with additional responses in diagonal bins. The forest distribution is flatter: each bin contains between 10.1% and 13.2%. This shows that its local edge directions are spread more evenly than those of the structured scenes. Gradient directions are perpendicular to the corresponding image edges.

### Parameter experiment

Smaller 4 × 4 cells preserve finer local changes, but increase the descriptor from 8100 to 34596 values. Larger 16 × 16 cells give a much coarser pattern and reduce it to 1764 values. Changing the block from 1 × 1 to 3 × 3 does not visibly change the line pattern in the scikit-image visualization because that image displays the cell histograms before block normalization. It does change how neighboring cells are grouped and normalized, and changes the descriptor size from 2304 to 15876 values.

Using 6 orientation bins gives a coarser set of directions and 5400 descriptor values. With 12 bins, the direction sampling is finer and the descriptor grows to 10800 values, although the visual difference from 9 bins is modest in this image. Smaller cells, larger blocks and more orientation bins can describe more detail, but they also increase descriptor size and computation. The baseline keeps the main building structure clear without producing the largest descriptor.

## Local Binary Patterns

The experiment used three grayscale images with different kinds of texture. The natural scene is `forest/10037.jpg` from the Intel training set. The other two are sample images bundled with scikit-image: `brick`, a regular man-made texture, and the face region cropped from `astronaut`. Each image was converted to grayscale and resized to 128 × 128 pixels, so that all three histograms are built from the same number of pixels.

### LBP computation

The LBP function was written with NumPy instead of a library routine. Each pixel is compared with its eight direct neighbours. A neighbour that is at least as bright as the centre gives a 1, and a darker neighbour gives a 0. The eight bits are read clockwise from the top-left neighbour, which is the lowest bit, and together they form a code between 0 and 255. Border pixels do not have eight neighbours, so only interior pixels receive a code and the LBP image is 126 × 126.

The code records only whether each neighbour is brighter or darker, not by how much. An overall change in brightness therefore leaves the codes unchanged. The same property means that weak noise in a smooth area produces codes just as readily as a strong edge does.

### LBP images

In the forest image the LBP response is dense across the whole frame. The trunks appear as vertical streaks in the upper half, and the leaf-covered ground gives diagonal streaks. The brick image shows dark lines that follow the mortar joints, while the brick faces are mostly bright with scattered speckles. In the face image the outlines of the hair, eyes, nose, mouth and collar are clearly visible. The smooth background and skin become speckled, because small intensity differences there are treated the same way as large ones.

### Histograms

The histogram counts how often each of the 256 codes occurs. It describes which local patterns a texture is built from, but not where they are located. The bars are coloured by pattern type. A uniform pattern has at most two changes between 0 and 1 around the circle of neighbours, which corresponds to flat areas, spots, line ends, edges and corners. Only 58 of the 256 codes are uniform. The remaining codes describe irregular neighbourhoods.

| Image | Uniform patterns | Entropy (bits) | Most common codes |
|---|---:|---:|---|
| Forest | 70.5% | 6.92 | 255 (7.7%), 0 (6.7%), 60 (1.7%) |
| Brick | 86.0% | 5.79 | 255 (16.6%), 30 (4.3%), 2 (3.7%) |
| Face | 84.7% | 6.35 | 255 (9.5%), 135 (3.9%), 240 (3.5%) |

Code 255 is the largest bin in all three images. It means that no neighbour is darker than the centre, which happens in flat areas and at local dark spots. It is most frequent in the brick image, where the brick faces are nearly constant after resizing.

The brick histogram is the most concentrated. It has the lowest entropy and the highest share of uniform patterns, and its energy sits in a few groups of codes. Codes such as 30 and 60 have four adjacent neighbours set, which is the pattern of a straight edge. This matches a texture made of flat faces separated by straight joints with a similar direction.

The forest histogram is the flattest. All 256 codes occur, the entropy is the highest of the three, and almost 30% of the pixels have a non-uniform pattern. Code 0, where the centre is brighter than every neighbour, is the second most common code. This reflects the many small bright specks from leaves and light between the branches. The texture is irregular and has no dominant edge direction at this scale.

The face lies between the two. Its share of uniform patterns is close to that of the brick, but the peaks are spread over more edge codes, such as 135 and 240, which have four adjacent neighbours set on different sides. The face contains smooth shading and curved contours in many directions instead of one repeated direction.

### Limitations

The basic method only looks one pixel away from the centre, so it describes the finest scale in the image. Resizing changes what that scale is. The brick image was reduced from 512 × 512, which smooths the surface of each brick. The codes also depend on direction. Rotating the brick image would move its peaks to other codes, even though the texture is the same.

## Blob Detection

Blob detection was applied to four images from the Intel training set, chosen to cover different kinds of blobs: `sea/10399.jpg` (a single boat on calm water), `buildings/10310.jpg` (dark windows on a plain wall), `street/10015.jpg` (street lamps at night) and `sea/1021.jpg` (rocks in shallow water). The images were used at their original 150 × 150 size. Detection was done on the grayscale image scaled to [0, 1], and the results are drawn on the colour original.

### Method

A blob is a region that is brighter or darker than its surroundings. The detector uses the Laplacian of Gaussian (LoG). The image is first smoothed with a Gaussian of width σ, and the Laplacian then measures how much each pixel differs from its neighbourhood. The response is strongest at the centre of a blob whose size matches σ. A bright blob gives a negative response and a dark blob a positive one, so the sign separates the two types.

The algorithm was implemented with the following steps, using OpenCV only for the Gaussian blur and the Laplacian filter:

1. Compute the LoG response for 11 values of σ from 2 to 12. Each response is multiplied by σ², which makes responses from different scales comparable.
2. Find points that are larger than all 26 neighbours in position and scale, and above a threshold. Each such point is a blob with centre (x, y) and radius √2 · σ.
3. Remove overlapping detections. Blobs are sorted by response strength, and a blob is dropped if more than half of the smaller circle is covered by a stronger blob.

Bright and dark blobs are found separately by running steps 2 and 3 on the negated and the original response. The implementation was checked on synthetic discs with known position and radius, where it returned the correct centres and the nearest available radius. For an ideal disc the response is about 0.74 times the contrast between the disc and its background. The baseline threshold of 0.2 therefore corresponds to a contrast of roughly 0.27 on the [0, 1] intensity scale.

### Results

| Image | Blobs | Bright | Dark | Radius min / median / max (pixels) |
|---|---:|---:|---:|---|
| Boat | 1 | 0 | 1 | 4.24 / 4.24 / 4.24 |
| Windows | 14 | 3 | 11 | 2.83 / 4.95 / 14.14 |
| Lamps | 22 | 21 | 1 | 2.83 / 2.83 / 2.83 |
| Rocks | 23 | 8 | 15 | 2.83 / 2.83 / 11.31 |

The position, radius and response of every blob are stored in `results/blob/blobs.csv`.

The boat image gives a single dark blob on the hull. The boat is long and narrow, so the circle covers only part of it. The smooth sea and sky produce no detections.

In the windows image, each of the five separate windows is found as one dark blob with a radius of 4 to 6 pixels, which matches their size. The long band of windows is not resolved into single panes. The panes are separated by thin frames, and after smoothing the band acts as one elongated dark region. It gives one large blob where the band is tallest and a few small ones. Several blobs lie on the image border, where the edge of the picture cuts through the sky and a brighter strip of wall.

The lamps image gives 21 bright blobs on the street lamps and lit windows. All have the smallest available radius, 2.83 pixels. The lamps are smaller than the lowest scale in the search, so their true size is not measured. The dark road and trees give no detections.

The rocks image is the most complex. Most of the 15 dark blobs are shadows and gaps between rocks, and the bright blobs are light-coloured stones. The rocks themselves are not outlined one by one, and the count is far below the real number of stones. A rock is not a region of uniform brightness. It has a lit side and a shadowed side, which the detector sees as separate bright and dark blobs.

### Effect of parameters

The threshold has the largest effect. The table shows the number of blobs for three thresholds, with all other parameters at the baseline.

| Threshold | Boat | Windows | Lamps | Rocks |
|---:|---:|---:|---:|---:|
| 0.10 | 4 | 87 | 106 | 136 |
| 0.20 | 1 | 14 | 22 | 23 |
| 0.30 | 0 | 6 | 8 | 0 |

A low threshold accepts weak responses from wall texture, ripples and foliage, and the count rises sharply. A high threshold keeps only strong contrasts. The lamps and windows fall slowly at high thresholds because they contain a few very strong structures. The rocks fall from 23 to 0 between 0.20 and 0.30, which shows that the image contains many blobs of similar, moderate contrast. No single threshold suits all four images. At 0.20 the boat is barely detected, while the windows image still contains border detections.

The scale range controls which sizes can be found. On the rocks image, searching only σ = 2 to 4 gives 22 blobs, almost the same as the baseline of 23, because most blobs are small. Searching only σ = 6 to 12 gives 3 blobs, and the two dark rocks on the left are then merged into one large blob. The smallest σ also sets a lower limit on the reported radius, as seen in the lamps image.

The overlap limit had little effect here. Without pruning, the rocks image gives 24 blobs instead of 23. The extra detection is a large circle around two smaller blobs that describe the same pair of rocks. Pruning matters more at low thresholds, where many weak responses occur at neighbouring scales.

### Limitations

The detector describes every blob as a circle, so elongated or rectangular objects such as the boat and the window band are represented poorly. It reacts to contrast, not to objects, and can therefore split one object into several blobs or respond to shadows. Detections on the image border are caused by the image edge and not by real structures.

## Contour Detection

Contour detection was applied to the same four Intel images as blob detection: the boat, the windows, the lamps and the rocks. This makes the two methods directly comparable.

### Method

A contour is the closed outline of a connected region. The detector first turns the grayscale image into a binary image and then traces the border of every region in it:

1. Smooth the image with a 5 × 5 Gaussian filter to remove small details and noise.
2. Threshold the image. Pixels brighter than the threshold become white and the rest black. The baseline uses Otsu's method, which chooses the threshold that best separates the image histogram into a dark and a bright group.
3. Trace the borders with OpenCV's `findContours`, which follows the edge of each white region and of each black hole inside a white region.
4. Discard contours with an area below 15 pixels².

For each contour the area, the perimeter and the centre of mass are calculated. The measurements were checked on synthetic shapes. A rectangle was measured exactly. For a disc of radius 20 the area was 1200 pixels² instead of 1257 and the perimeter 131.9 instead of 125.7, because the outline follows the pixel grid in small steps.

### Results

| Image | Otsu threshold | Contours | Area median / max (pixels²) | Perimeter median / max (pixels) |
|---|---:|---:|---|---|
| Boat | 129 | 2 | 6724.5 / 13343.5 | 260.4 / 468.3 |
| Windows | 101 | 6 | 124.0 / 14060.0 | 43.7 / 946.5 |
| Lamps | 52 | 18 | 29.8 / 7539.0 | 22.7 / 1293.3 |
| Rocks | 117 | 17 | 40.0 / 10698.0 | 40.1 / 608.6 |

The position, area and perimeter of every contour are stored in `results/contour/contours.csv`.

Every image has one very large contour, which is why the maximum area is far above the median. It is the outline of the bright side of the binary image, not of an object. In the boat image this region covers 59% of the picture. Its lower border is an arc across the water, caused by the gradual darkening towards the bottom and the corners. The second contour follows the hull of the boat closely, with an area of 105.5 pixels².

In the windows image the five separate windows are each outlined with their rectangular shape, with areas from 69.5 to 182 pixels². The long band of windows does not get its own contour. In the binary image it is connected to the dark sky at its left end, so it becomes part of the outline of the wall.

The lamps image has a low Otsu threshold of 52 because most of the picture is dark. Two lamps at the top are outlined separately. The other lamps, the lit shop fronts and the lit ground merge into one region with a perimeter of 1293 pixels.

In the rocks image the sky and the sea form the large region, and its ragged lower border runs where the rocks begin. The other contours are bright patches on the rocks. Most of them cover part of a rock or several rocks together, and only a few stones are outlined as single objects.

The plot of area against perimeter shows how compact the regions are. A circle has the shortest perimeter possible for its area. The windows lie close to that line. The rock contours lie well above it, which reflects their irregular outlines, and the large background regions lie furthest above.

### Effect of parameters

The threshold decides which pixels belong together, so it changes the regions themselves and not only how many are kept. On the rocks image, a threshold of 80 gives 13 contours. More of the picture becomes white, bright patches merge, and the two dark rocks on the left appear as holes. A threshold of 160 leaves only 2 contours, around the brightest sky and one light stone. The number of contours does not fall steadily as the threshold rises, as it did for blob detection. It goes up and down, because regions split and merge. The windows image shows this clearly. It has 6 contours for thresholds from 110 to 130 and 14 at 150, where the wall itself starts to break up.

The blur size controls how much detail survives. Without blurring, the rocks image gives 33 contours with ragged outlines. The baseline 5 × 5 filter gives 17, and an 11 × 11 filter gives 10 with smoother outlines. The windows image is stable at 6 contours for every blur size from 5 upwards, and the boat image gives 2 contours for all sizes. Blurring therefore matters most for textured scenes.

The minimum area removes small specks. Without it, the rocks image gives 63 contours instead of 17.

### Comparison with blob detection

| Image | Blobs | Contours |
|---|---:|---:|
| Boat | 1 | 2 |
| Windows | 14 | 6 |
| Lamps | 22 | 18 |
| Rocks | 23 | 17 |

The counts are of the same order, but the two methods mark different things.

For the five separate windows the methods agree closely. The blob centres and the contour centres differ by less than one pixel. The difference is in the description. A blob is a circle with one of 11 possible radii, while a contour follows the rectangular window and gives its area and perimeter.

The boat shows the same pattern. The blob is a circle on part of the hull, and the contour follows the whole hull. Blob detection gives no other detections in this image, while contour detection also returns the large region created by the brightness gradient.

The lamps are handled better by blob detection. It marks 21 lamps and lit windows as separate bright blobs. Contour detection merges neighbouring lamps and lit surfaces into one region, because they are connected in the binary image.

Neither method isolates the individual rocks. Blob detection marks shadows and gaps, and contour detection outlines bright patches.

### Advantages and limitations

Blob detection compares each point with its local surroundings. It is therefore not disturbed by gradual brightness changes across the image, and it can separate bright spots that lie close together. It also searches several sizes at once. Its limitations are that every blob is a circle, that the size is limited to the scales in the search, and that it gives no information about shape. It also produces detections at the image border.

Contour detection gives the actual outline of a region, of any shape, together with its area and perimeter. Its main limitation is the single global threshold. Regions that touch in the binary image merge, uneven lighting creates contours that are not objects, and the background itself is returned as a contour. The result can change abruptly with the threshold.

### When to use which

Blob detection is more suitable when the task is to find or count small, compact structures, especially when the lighting is uneven or the structures lie close together. The street lamps are an example, and stars or cells under a microscope would be similar cases.

Contour detection is more suitable when well-separated objects stand out clearly from an even background and their shape or size matters. The windows on the wall and the boat on the water are examples. Measuring the area of a window is possible from its contour, but not from a blob.
