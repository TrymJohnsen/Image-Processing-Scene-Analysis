from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SIZE = (64, 64)
NUMBER_OF_IMAGES = 200
RANDOM_SEED = 42


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


if __name__ == "__main__":
    main()
