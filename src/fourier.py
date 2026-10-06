from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def circular_low_pass_mask(shape, cutoff_radius):
    rows, columns = shape
    center_row, center_column = rows // 2, columns // 2
    y, x = np.ogrid[:rows, :columns]
    distance_squared = (y - center_row) ** 2 + (x - center_column) ** 2
    return distance_squared <= cutoff_radius**2


project_root = Path(__file__).resolve().parents[1]
image_path = (
    project_root
    / "data"
    / "intel-image-classification"
    / "seg_train"
    / "seg_train"
    / "buildings"
    / "1001.jpg"
)
output_path = project_root / "results" / "fourier" / "fourier_spectrum.png"
low_pass_output_path = (
    project_root / "results" / "fourier" / "low_pass_comparison.png"
)

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

random_seed = 42
noise_mean = 0
noise_standard_deviation = 25
cutoff_radii = [10, 20, 35]

random_generator = np.random.default_rng(random_seed)
noise = random_generator.normal(
    noise_mean, noise_standard_deviation, image.shape
)
noisy_image = np.clip(image.astype(float) + noise, 0, 255)

noisy_transform = np.fft.fft2(noisy_image)
shifted_noisy_transform = np.fft.fftshift(noisy_transform)

filtered_images = []
for radius in cutoff_radii:
    mask = circular_low_pass_mask(image.shape, radius)
    filtered_transform = shifted_noisy_transform * mask

    # Move zero frequency back before reconstructing the spatial image.
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_transform))
    filtered_image = np.clip(np.real(filtered_image), 0, 255)
    filtered_images.append(filtered_image)

comparison_figure, comparison_axes = plt.subplots(1, 5, figsize=(16, 4))

comparison_axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
comparison_axes[0].set_title("Original")

comparison_axes[1].imshow(noisy_image, cmap="gray", vmin=0, vmax=255)
comparison_axes[1].set_title("Gaussian noise")

for axis, radius, filtered_image in zip(
    comparison_axes[2:], cutoff_radii, filtered_images
):
    axis.imshow(filtered_image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"Low-pass, radius {radius}")

for axis in comparison_axes:
    axis.axis("off")

comparison_figure.tight_layout()
comparison_figure.savefig(low_pass_output_path, dpi=150, bbox_inches="tight")
print(
    f"Gaussian noise: mean={noise_mean}, "
    f"standard deviation={noise_standard_deviation}, seed={random_seed}"
)
print(f"Cutoff radii: {cutoff_radii}")
print(f"Low-pass comparison saved to: {low_pass_output_path}")
plt.show()
