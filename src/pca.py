import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = (64, 64)
NUMBER_OF_IMAGES = 200
RANDOM_SEED = 42
TESTED_COMPONENT_COUNTS = [1, 5, 10, 25, 50, 100, 150]


def load_images(image_directory, number_of_images, image_size):
    image_paths = sorted(image_directory.glob("*.jpg"))
    if len(image_paths) < number_of_images:
        raise ValueError(
            f"Requested {number_of_images} images, but found {len(image_paths)}"
        )

    random_generator = np.random.default_rng(RANDOM_SEED)
    selected_indices = random_generator.choice(
        len(image_paths), size=number_of_images, replace=False
    )
    selected_paths = [image_paths[index] for index in sorted(selected_indices)]

    images = []
    original_shapes = set()
    for image_path in selected_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_shapes.add(image.shape)
        resized_image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        images.append(resized_image.astype(np.float64) / 255.0)

    return np.array(images), original_shapes


def fit_pca(data_matrix):
    mean_image = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean_image

    # The sample-space covariance is 200 x 200 instead of 4096 x 4096.
    covariance_matrix = centered_data @ centered_data.T / (
        centered_data.shape[0] - 1
    )
    eigenvalues, sample_eigenvectors = np.linalg.eigh(covariance_matrix)

    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_order]
    sample_eigenvectors = sample_eigenvectors[:, descending_order]

    # Numerical round-off can produce eigenvalues just below zero.
    tolerance = np.finfo(eigenvalues.dtype).eps * max(
        covariance_matrix.shape
    ) * eigenvalues[0]
    positive = eigenvalues > tolerance
    eigenvalues = eigenvalues[positive]
    sample_eigenvectors = sample_eigenvectors[:, positive]

    # Map sample-space eigenvectors back to directions in pixel space.
    components = centered_data.T @ sample_eigenvectors
    components /= np.sqrt(
        (centered_data.shape[0] - 1) * eigenvalues
    )
    components = components.T

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    return mean_image, centered_data, eigenvalues, components, explained_variance_ratio


def project(centered_data, components, number_of_components):
    return centered_data @ components[:number_of_components].T


def reconstruct(projected_data, components, mean_image):
    return projected_data @ components[: projected_data.shape[1]] + mean_image


def save_mean_image(mean_image, image_shape, output_directory):
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(mean_image.reshape(image_shape), cmap="gray", vmin=0, vmax=1)
    axis.set_title("Mean building image")
    axis.axis("off")
    figure.tight_layout()
    output_path = output_directory / "mean_image.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Mean image saved to: {output_path}")


def save_explained_variance(explained_variance_ratio, output_directory):
    cumulative_variance = np.cumsum(explained_variance_ratio)
    component_counts = np.arange(1, len(cumulative_variance) + 1)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(component_counts, cumulative_variance, color="tab:blue")
    for level in (0.8, 0.9, 0.95):
        axis.axhline(level, color="gray", linestyle="--", linewidth=0.8)
    axis.set(
        xlabel="Number of principal components (k)",
        ylabel="Cumulative explained variance",
        title="PCA cumulative explained variance",
        xlim=(1, len(cumulative_variance)),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = output_directory / "cumulative_explained_variance.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Explained variance plot saved to: {output_path}")


def evaluate_reconstructions(
    data_matrix,
    centered_data,
    mean_image,
    components,
    explained_variance_ratio,
    component_counts,
):
    cumulative_variance = np.cumsum(explained_variance_ratio)
    reconstructions = {}
    metrics = []

    for component_count in component_counts:
        projected_data = project(centered_data, components, component_count)
        reconstructed_data = reconstruct(
            projected_data, components, mean_image
        )
        mse = np.mean((data_matrix - reconstructed_data) ** 2)
        reconstructions[component_count] = reconstructed_data
        metrics.append(
            (
                component_count,
                cumulative_variance[component_count - 1],
                mse,
            )
        )

    return reconstructions, metrics


def save_reconstruction_comparison(
    data_matrix,
    reconstructions,
    image_shape,
    component_counts,
    output_directory,
):
    displayed_counts = [1, 10, 50, 100, component_counts[-1]]
    image_indices = [0, len(data_matrix) // 2, len(data_matrix) - 1]
    figure, axes = plt.subplots(
        len(image_indices),
        len(displayed_counts) + 1,
        figsize=(12, 6),
    )

    for row, image_index in enumerate(image_indices):
        axes[row, 0].imshow(
            data_matrix[image_index].reshape(image_shape),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 0].set_title("Original" if row == 0 else "")

        for column, component_count in enumerate(displayed_counts, start=1):
            reconstructed_image = reconstructions[component_count][image_index]
            axes[row, column].imshow(
                np.clip(reconstructed_image.reshape(image_shape), 0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, column].set_title(
                f"k = {component_count}" if row == 0 else ""
            )

    for axis in axes.ravel():
        axis.axis("off")

    figure.suptitle("Original and PCA-reconstructed building images")
    figure.tight_layout()
    output_path = output_directory / "reconstruction_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Reconstruction comparison saved to: {output_path}")


def save_reconstruction_error(metrics, output_directory):
    component_counts = [row[0] for row in metrics]
    mse_values = [row[2] for row in metrics]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(component_counts, mse_values, marker="o", color="tab:red")
    axis.set(
        xlabel="Number of principal components (k)",
        ylabel="Mean squared error",
        title="PCA reconstruction error",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = output_directory / "reconstruction_error.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Reconstruction error plot saved to: {output_path}")


def save_metrics(metrics, pixel_count, output_directory):
    output_path = output_directory / "metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["k", "cumulative_explained_variance", "mse", "pixel_reduction"]
        )
        for component_count, cumulative_variance, mse in metrics:
            writer.writerow(
                [
                    component_count,
                    f"{cumulative_variance:.6f}",
                    f"{mse:.8f}",
                    f"{pixel_count / component_count:.2f}:1",
                ]
            )

    print(f"Metrics saved to: {output_path}")
    print("\n  k | cumulative variance |        MSE | pixel reduction")
    print("----|---------------------|------------|----------------")
    for component_count, cumulative_variance, mse in metrics:
        print(
            f"{component_count:>3} | {cumulative_variance:>19.4f} | "
            f"{mse:>10.6f} | {pixel_count / component_count:>10.2f}:1"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]
    image_directory = (
        project_root
        / "data"
        / "intel-image-classification"
        / "seg_train"
        / "seg_train"
        / "buildings"
    )
    output_directory = project_root / "results" / "pca"
    output_directory.mkdir(parents=True, exist_ok=True)

    images, original_shapes = load_images(
        image_directory, NUMBER_OF_IMAGES, IMAGE_SIZE
    )
    image_shape = images.shape[1:]
    data_matrix = images.reshape(images.shape[0], -1)

    print(f"Number of images: {images.shape[0]}")
    print(f"Original dimensions found: {sorted(original_shapes)}")
    print(f"Common image dimensions: {image_shape}")
    print(f"X shape: {data_matrix.shape}")

    (
        mean_image,
        centered_data,
        eigenvalues,
        components,
        explained_variance_ratio,
    ) = fit_pca(data_matrix)

    projected_data = project(centered_data, components, len(components))
    assert components.shape == (len(eigenvalues), data_matrix.shape[1])
    assert projected_data.shape == (data_matrix.shape[0], len(components))
    assert np.all(np.diff(eigenvalues) <= 0)
    assert np.allclose(components @ components.T, np.eye(len(components)))

    print(f"Sample covariance shape: ({images.shape[0]}, {images.shape[0]})")
    print(f"Principal components shape: {components.shape}")
    print(f"Projected data shape: {projected_data.shape}")
    print(f"Total explained variance: {np.sum(explained_variance_ratio):.6f}")

    save_mean_image(mean_image, image_shape, output_directory)
    save_explained_variance(explained_variance_ratio, output_directory)

    component_counts = TESTED_COMPONENT_COUNTS + [len(components)]
    reconstructions, metrics = evaluate_reconstructions(
        data_matrix,
        centered_data,
        mean_image,
        components,
        explained_variance_ratio,
        component_counts,
    )
    mse_values = np.array([row[2] for row in metrics])
    cumulative_variances = np.array([row[1] for row in metrics])
    assert np.all(np.diff(mse_values) <= 1e-12)
    assert np.all(np.diff(cumulative_variances) >= 0)

    save_reconstruction_comparison(
        data_matrix,
        reconstructions,
        image_shape,
        component_counts,
        output_directory,
    )
    save_reconstruction_error(metrics, output_directory)
    save_metrics(metrics, data_matrix.shape[1], output_directory)


if __name__ == "__main__":
    main()
