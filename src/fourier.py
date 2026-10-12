import csv
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


def retain_symmetric_coefficients(transform, percentage):
    """Keep the strongest conjugate groups, reaching the requested count."""
    rows, columns = transform.shape
    row_indices, column_indices = np.indices(transform.shape)
    indices = np.arange(transform.size)
    partners = (
        (-row_indices % rows) * columns + (-column_indices % columns)
    ).ravel()
    # Each pair is visited once; DC and other self-conjugate bins count once.
    representatives = indices[indices <= partners]
    magnitudes = np.abs(transform.ravel())
    strengths = (
        magnitudes[representatives] + magnitudes[partners[representatives]]
    ) / 2
    order = representatives[np.argsort(-strengths, kind="stable")]
    requested_count = max(1, round(transform.size * percentage / 100))
    selected = np.zeros(transform.size, dtype=bool)
    retained_count = 0
    for index in order:
        partner = partners[index]
        selected[index] = selected[partner] = True
        retained_count += 1 if index == partner else 2
        if retained_count >= requested_count:
            break
    compressed = np.where(selected, transform.ravel(), 0).reshape(transform.shape)
    return compressed, retained_count


def checked_real_reconstruction(transform):
    """Check conjugate symmetry and imaginary residual before using real values."""
    rows, columns = transform.shape
    conjugate_partners = np.conj(transform[np.ix_(
        -np.arange(rows) % rows, -np.arange(columns) % columns
    )])
    symmetry_error = float(np.max(np.abs(transform - conjugate_partners)))
    if not np.allclose(transform, conjugate_partners, rtol=1e-12, atol=1e-8):
        raise ValueError(f"Fourier conjugate symmetry is broken: {symmetry_error}")
    inverse = np.fft.ifft2(transform)
    imaginary_residual = float(np.max(np.abs(inverse.imag)))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(inverse.real))))
    if imaginary_residual > tolerance:
        raise ValueError(f"Non-negligible imaginary residual: {imaginary_residual}")
    return inverse.real, imaginary_residual, symmetry_error


# Shared setup
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

# Task 1: Fourier magnitude spectrum
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

# Task 2: Low-pass filtering
random_seed = 42
noise_mean = 0
noise_standard_deviation = 25
cutoff_radii = [10, 20, 35]
low_pass_cutoff_radii = cutoff_radii + [50, 70]

random_generator = np.random.default_rng(random_seed)
noise = random_generator.normal(
    noise_mean, noise_standard_deviation, image.shape
)
noisy_image = np.clip(image.astype(float) + noise, 0, 255)

noisy_transform = np.fft.fft2(noisy_image)
shifted_noisy_transform = np.fft.fftshift(noisy_transform)

filtered_images = []
low_pass_metrics = [
    ("Noisy image", "", np.mean((image.astype(float) - noisy_image) ** 2))
]
for radius in low_pass_cutoff_radii:
    mask = circular_low_pass_mask(image.shape, radius)
    filtered_transform = shifted_noisy_transform * mask

    # Move zero frequency back before reconstructing the spatial image.
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_transform))
    filtered_image = np.clip(np.real(filtered_image), 0, 255)
    filtered_images.append(filtered_image)
    low_pass_metrics.append((
        "Low-pass", radius, np.mean((image.astype(float) - filtered_image) ** 2)
    ))

comparison_figure, comparison_axes = plt.subplots(2, 4, figsize=(12, 6))
comparison_axes = comparison_axes.ravel()

comparison_axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
comparison_axes[0].set_title("Original")

comparison_axes[1].imshow(noisy_image, cmap="gray", vmin=0, vmax=255)
comparison_axes[1].set_title("Gaussian noise")

for axis, radius, filtered_image in zip(
    comparison_axes[2:], low_pass_cutoff_radii, filtered_images
):
    axis.imshow(filtered_image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"Low-pass, radius {radius}")

for axis in comparison_axes:
    axis.axis("off")
comparison_axes[-1].set_visible(False)

comparison_figure.tight_layout()
comparison_figure.savefig(low_pass_output_path, dpi=150, bbox_inches="tight")
print(
    f"Gaussian noise: mean={noise_mean}, "
    f"standard deviation={noise_standard_deviation}, seed={random_seed}"
)
print(f"Low-pass cutoff radii: {low_pass_cutoff_radii}")
print(f"Low-pass comparison saved to: {low_pass_output_path}")
low_pass_metrics_path = output_path.parent / "low_pass_metrics.csv"
with low_pass_metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(["method", "cutoff_radius", "mse_against_clean"])
    for method, radius, mse in low_pass_metrics:
        writer.writerow([method, radius, f"{mse:.8f}"])
        print(f"{method}, radius={radius}: MSE against clean={mse:.2f}")
print(f"Low-pass metrics saved to: {low_pass_metrics_path}")

# Task 3: High-pass filtering
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

# Task 4: Fourier compression
retained_percentages = [1, 5, 10, 20, 50, 100]
total_coefficients = fourier_transform.size

reconstructed_images = []
actual_percentages = []
compression_metrics = []

for percentage in retained_percentages:
    compressed_transform, retained_count = retain_symmetric_coefficients(
        fourier_transform, percentage
    )
    reconstructed_image, imaginary_residual, symmetry_error = (
        checked_real_reconstruction(compressed_transform)
    )
    mse = np.mean((image.astype(float) - reconstructed_image) ** 2)
    compression_ratio = total_coefficients / retained_count
    actual_percentage = retained_count / total_coefficients * 100

    reconstructed_images.append(reconstructed_image)
    actual_percentages.append(actual_percentage)
    compression_metrics.append((
        percentage, actual_percentage, retained_count, mse, compression_ratio,
        imaginary_residual, symmetry_error,
    ))

compression_csv_path = output_path.parent / "compression_metrics.csv"
with compression_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow([
        "requested_percentage", "actual_retained_percentage", "retained_count",
        "mse", "coefficient_ratio", "max_imaginary_residual", "max_symmetry_error",
    ])
    for requested, actual, count, mse, ratio, residual, symmetry_error in compression_metrics:
        writer.writerow([
            requested, f"{actual:.8f}", count, f"{mse:.8f}", f"{ratio:.8f}",
            f"{residual:.8e}", f"{symmetry_error:.8e}",
        ])

compression_figure, compression_axes = plt.subplots(2, 4, figsize=(12, 6))
compression_axes = compression_axes.ravel()

compression_axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
compression_axes[0].set_title("Original")

for axis, percentage, actual_percentage, reconstructed_image in zip(
    compression_axes[1:], retained_percentages, actual_percentages, reconstructed_images
):
    axis.imshow(
        np.clip(reconstructed_image, 0, 255),
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axis.set_title(f"Target {percentage}%\nActual {actual_percentage:.4f}% retained")

for axis in compression_axes:
    axis.axis("off")

compression_figure.tight_layout()
compression_figure.savefig(
    compression_output_path, dpi=150, bbox_inches="tight"
)

metrics_figure, metrics_axis = plt.subplots(figsize=(9, 3))
metrics_axis.axis("off")
table_rows = [
    [f"{requested}%", f"{actual:.4f}%", f"{mse:.2f}", f"{ratio:.2f}:1", f"{residual:.1e}"]
    for requested, actual, count, mse, ratio, residual, symmetry_error in compression_metrics
]
metrics_table = metrics_axis.table(
    cellText=table_rows,
    colLabels=["Target", "Actual retained", "MSE", "Coefficient ratio", "Max |imaginary|"],
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
print(f"Compression CSV saved to: {compression_csv_path}")
print("Theoretical ratios count coefficients, not file size on disk.")
for requested, actual, count, mse, ratio, residual, symmetry_error in compression_metrics:
    print(
        f"Target {requested:>3}%, actual {actual:.6f}% ({count} coefficients): "
        f"MSE={mse:.2f}, ratio={ratio:.2f}:1, max imaginary={residual:.3e}"
    )

plt.close("all")
