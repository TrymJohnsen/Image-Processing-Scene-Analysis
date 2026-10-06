**Fourier Transform**

The image was converted to grayscale before applying a 2D Fourier transform. The resulting spectrum was shifted so that the lowest spatial frequencies appear at the center. A logarithmic magnitude was used for visualization since the Fourier coefficients vary considerably in magnitude.

Most of the energy is concentrated around the center of the spectrum, indicating that low spatial frequencies make up a large part of the image. This is expected since the image contains larger regions with gradual changes in intensity. A clear cross-shaped structure can also be observed in the spectrum. The building contains many repeated horizontal and vertical features, including floors, railings, windows and supporting columns, which produce strong directional frequency components. Weaker components extend further from the center and represent finer details and sharper intensity changes in the image.

The result shows that the Fourier transform does not describe where individual objects are located. Instead, it reveals how strongly different spatial frequencies and directions are represented in the image.

**Low-pass filtering**

Gaussian noise was added to the grayscale image before filtering so that the effect of the low-pass filter could be evaluated on known high-frequency noise. Three cutoff radii were tested: 10, 20 and 35 pixels.

A cutoff radius of 10 removed a large amount of the noise, but also removed most fine image structure. The building remained recognizable, but windows, railings and sharp edges were heavily blurred. Increasing the radius to 20 restored more structural detail while still reducing much of the noise. At radius 35, the original structure was preserved more clearly, but more high-frequency noise was also retained.

The experiment shows the trade-off of frequency-domain smoothing. A small cutoff gives stronger noise suppression but also removes useful detail, while a larger cutoff preserves more image information at the cost of weaker noise removal. The filter cannot distinguish between unwanted high-frequency noise and genuine high-frequency structures such as edges, since both occupy similar parts of the frequency spectrum.
