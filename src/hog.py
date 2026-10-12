from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog


IMAGE_SIZE = (128, 128)
BASELINE = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
}
PARAMETER_CONFIGURATIONS = [
    ("Baseline", 9, (8, 8), (2, 2)),
    ("Smaller cells", 9, (4, 4), (2, 2)),
    ("Larger cells", 9, (16, 16), (2, 2)),
    ("1 x 1 block", 9, (8, 8), (1, 1)),
    ("3 x 3 block", 9, (8, 8), (3, 3)),
    ("6 orientations", 6, (8, 8), (2, 2)),
    ("12 orientations", 12, (8, 8), (2, 2)),
]


def load_grayscale_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    resized_image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return resized_image.astype(np.float64) / 255.0


def gradient_magnitude(image):
    # Separate illustration: scikit-image HOG computes its own gradients.
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    if magnitude.max() > 0:
        magnitude /= magnitude.max()
    return magnitude


def compute_hog(image, orientations, pixels_per_cell, cells_per_block):
    features, visualization = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    upper_limit = np.percentile(visualization, 99)
    visualization = exposure.rescale_intensity(
        visualization, in_range=(0, upper_limit), out_range=(0, 1)
    )
    return features, visualization


def save_image_comparison(image, image_name, output_directory):
    magnitude = gradient_magnitude(image)
    features, hog_visualization = compute_hog(image, **BASELINE)

    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    panels = [
        (image, "Original grayscale"),
        (magnitude, "Sobel gradient magnitude"),
        (hog_visualization, "HOG visualization"),
    ]
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"HOG features: {image_name}")
    figure.tight_layout()
    output_path = output_directory / f"{image_name}_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"{image_name}: {len(features)} baseline HOG features")
    print(f"Comparison saved to: {output_path}")


def save_parameter_comparison(image, output_directory):
    figure, axes = plt.subplots(2, 4, figsize=(14, 9), layout="constrained")
    axes = axes.ravel()

    for axis, configuration in zip(axes, PARAMETER_CONFIGURATIONS):
        label, orientations, pixels_per_cell, cells_per_block = configuration
        features, visualization = compute_hog(
            image, orientations, pixels_per_cell, cells_per_block
        )
        axis.imshow(visualization, cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"{label}\n"
            f"ori={orientations}, cell={pixels_per_cell[0]}, "
            f"block={cells_per_block[0]}\n"
            f"features={len(features)}",
            fontsize=9,
        )
        axis.axis("off")
        print(
            f"{label}: orientations={orientations}, "
            f"pixels_per_cell={pixels_per_cell}, "
            f"cells_per_block={cells_per_block}, "
            f"features={len(features)}"
        )

    axes[-1].axis("off")
    figure.suptitle("Effect of HOG parameters on the building image")
    output_path = output_directory / "parameter_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Parameter comparison saved to: {output_path}")


def baseline_orientation_distribution(image):
    # Relative descriptor strength after block normalization, not edge counts.
    feature_blocks = hog(
        image,
        orientations=BASELINE["orientations"],
        pixels_per_cell=BASELINE["pixels_per_cell"],
        cells_per_block=BASELINE["cells_per_block"],
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=False,
    )
    distribution = feature_blocks.sum(axis=(0, 1, 2, 3))
    return distribution / distribution.sum()


def save_orientation_comparison(images, output_directory):
    bin_width = 180 / BASELINE["orientations"]
    bin_centers = np.arange(BASELINE["orientations"]) * bin_width + bin_width / 2

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for image_name, image in images.items():
        distribution = baseline_orientation_distribution(image)
        axis.plot(
            bin_centers,
            distribution * 100,
            marker="o",
            linewidth=2,
            label=image_name.capitalize(),
        )
        values = ", ".join(f"{value:.1%}" for value in distribution)
        print(f"{image_name} orientation distribution: {values}")

    axis.set(
        title="Baseline HOG orientation distributions",
        xlabel="Gradient orientation bin center (degrees)",
        ylabel="Relative HOG strength (%)",
        xticks=bin_centers,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = output_directory / "orientation_distribution.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Orientation comparison saved to: {output_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = (
        project_root
        / "data"
        / "intel-image-classification"
        / "seg_train"
        / "seg_train"
    )
    image_paths = {
        "simple_sea": dataset_directory / "sea" / "10399.jpg",
        "building": dataset_directory / "buildings" / "1001.jpg",
        "street": dataset_directory / "street" / "10019.jpg",
        "forest": dataset_directory / "forest" / "10037.jpg",
    }
    output_directory = project_root / "results" / "hog"
    output_directory.mkdir(parents=True, exist_ok=True)

    images = {}
    print("Selected images:")
    for image_name, image_path in image_paths.items():
        print(f"  {image_name}: {image_path.relative_to(project_root)}")
        images[image_name] = load_grayscale_image(image_path)
        save_image_comparison(images[image_name], image_name, output_directory)

    save_parameter_comparison(images["building"], output_directory)
    save_orientation_comparison(images, output_directory)


if __name__ == "__main__":
    main()
