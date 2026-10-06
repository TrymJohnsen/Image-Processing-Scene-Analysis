from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
image_path = (
    project_root
    / "intel-image-classification"
    / "seg_train"
    / "seg_train"
    / "buildings"
    / "1001.jpg"
)
output_path = project_root / "results" / "fourier" / "fourier_spectrum.png"

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

height, width = image.shape
print(f"Image dimensions: {height} x {width} pixels")

# Transform the image to the frequency domain and center low frequencies.
fourier_transform = np.fft.fft2(image)
shifted_transform = np.fft.fftshift(fourier_transform)

# A logarithmic scale makes weaker frequency components easier to see.
magnitude_spectrum = np.log1p(np.abs(shifted_transform))

figure, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original grayscale image")
axes[0].axis("off")

axes[1].imshow(magnitude_spectrum, cmap="gray")
axes[1].set_title("Magnitude spectrum")
axes[1].axis("off")

plt.tight_layout()
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to: {output_path}")
plt.show()
