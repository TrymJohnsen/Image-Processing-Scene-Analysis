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
