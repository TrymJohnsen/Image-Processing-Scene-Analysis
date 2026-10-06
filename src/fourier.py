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
high_pass_output_path = (
    project_root / "results" / "fourier" / "high_pass_comparison.png"
)
compression_output_path = (
    project_root / "results" / "fourier" / "compression_comparison.png"
)
compression_metrics_output_path = (
    project_root / "results" / "fourier" / "compression_metrics.png"
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

high_pass_images = []
for radius in cutoff_radii:
    mask = ~circular_low_pass_mask(image.shape, radius)
    filtered_transform = shifted_transform * mask

    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_transform))
    high_pass_images.append(np.abs(filtered_image))

high_pass_figure, high_pass_axes = plt.subplots(1, 4, figsize=(13, 4))

high_pass_axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
high_pass_axes[0].set_title("Original")

high_pass_display_limit = max(
    np.percentile(filtered_image, 99) for filtered_image in high_pass_images
)

for axis, radius, filtered_image in zip(
    high_pass_axes[1:], cutoff_radii, high_pass_images
):
    axis.imshow(
        filtered_image, cmap="gray", vmin=0, vmax=high_pass_display_limit
    )
    axis.set_title(f"High-pass, radius {radius}")

for axis in high_pass_axes:
    axis.axis("off")

high_pass_figure.tight_layout()
high_pass_figure.savefig(high_pass_output_path, dpi=150, bbox_inches="tight")
print(f"High-pass comparison saved to: {high_pass_output_path}")

retained_percentages = [1, 5, 10, 20, 50, 100]
total_coefficients = fourier_transform.size
flat_transform = fourier_transform.ravel()
coefficient_order = np.argsort(np.abs(flat_transform))[::-1]

reconstructed_images = []
mse_values = []
compression_ratios = []

for percentage in retained_percentages:
    retained_count = max(
        1, round(total_coefficients * percentage / 100)
    )
    retained_indices = coefficient_order[:retained_count]

    compressed_coefficients = np.zeros_like(flat_transform)
    compressed_coefficients[retained_indices] = flat_transform[retained_indices]
    compressed_transform = compressed_coefficients.reshape(image.shape)

    reconstructed_image = np.real(np.fft.ifft2(compressed_transform))
    mse = np.mean((image.astype(float) - reconstructed_image) ** 2)
    compression_ratio = total_coefficients / retained_count

    reconstructed_images.append(reconstructed_image)
    mse_values.append(mse)
    compression_ratios.append(compression_ratio)

compression_figure, compression_axes = plt.subplots(2, 4, figsize=(12, 6))
compression_axes = compression_axes.ravel()

compression_axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
compression_axes[0].set_title("Original")

for axis, percentage, reconstructed_image in zip(
    compression_axes[1:], retained_percentages, reconstructed_images
):
    axis.imshow(
        np.clip(reconstructed_image, 0, 255),
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axis.set_title(f"{percentage}% retained")

for axis in compression_axes:
    axis.axis("off")

compression_figure.tight_layout()
compression_figure.savefig(
    compression_output_path, dpi=150, bbox_inches="tight"
)

metrics_figure, metrics_axis = plt.subplots(figsize=(6, 3))
metrics_axis.axis("off")
table_rows = [
    [f"{percentage}%", f"{mse:.2f}", f"{ratio:.1f}:1"]
    for percentage, mse, ratio in zip(
        retained_percentages, mse_values, compression_ratios
    )
]
metrics_table = metrics_axis.table(
    cellText=table_rows,
    colLabels=["Coefficients retained", "MSE", "Theoretical ratio"],
    cellLoc="center",
    loc="center",
)
metrics_table.auto_set_font_size(False)
metrics_table.set_fontsize(10)
metrics_table.scale(1, 1.4)
metrics_axis.set_title("Fourier compression metrics")
metrics_figure.tight_layout()
metrics_figure.savefig(
    compression_metrics_output_path, dpi=150, bbox_inches="tight"
)

print(f"Compression comparison saved to: {compression_output_path}")
print(f"Compression metrics saved to: {compression_metrics_output_path}")
print("Theoretical ratios count coefficients, not file size on disk.")
for percentage, mse, ratio in zip(
    retained_percentages, mse_values, compression_ratios
):
    print(f"{percentage:>3}% retained: MSE={mse:.2f}, ratio={ratio:.1f}:1")

plt.show()
