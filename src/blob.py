import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


IMAGE_FILES = {
    "boat": ("sea", "10399.jpg"),
    "windows": ("buildings", "10310.jpg"),
    "lamps": ("street", "10015.jpg"),
    "rocks": ("sea", "1021.jpg"),
}
BASELINE = {
    "min_sigma": 2,
    "max_sigma": 12,
    "num_sigma": 11,
    "threshold": 0.2,
    "overlap": 0.5,
}
PARAMETER_CONFIGURATIONS = [
    ("Baseline", {}),
    ("Lower threshold", {"threshold": 0.15}),
    ("Higher threshold", {"threshold": 0.25}),
    ("Small scales only", {"max_sigma": 4, "num_sigma": 3}),
    ("Large scales only", {"min_sigma": 6, "num_sigma": 7}),
    ("No overlap pruning", {"overlap": 1.0}),
]
PARAMETER_IMAGE = "rocks"
SWEEP_THRESHOLDS = np.round(np.arange(0.04, 0.301, 0.02), 2)
POLARITY_COLORS = {"bright": "#ffd60a", "dark": "#ff2d95"}
IMAGE_STYLES = {
    "boat": ("#2a78d6", "o"),
    "windows": ("#eb6834", "s"),
    "lamps": ("#1baf7a", "^"),
    "rocks": ("#eda100", "D"),
}


def load_image(image_path):
    color_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image, grayscale_image.astype(np.float64) / 255.0


def log_scale_space(image, sigmas):
    responses = []
    for sigma in sigmas:
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=1)
        # Multiplying by sigma squared makes responses comparable across scales.
        responses.append(sigma**2 * laplacian)
    return np.array(responses)


def local_maxima(scale_space, threshold):
    kernel = np.ones((3, 3), np.uint8)
    neighborhood_maximum = np.array(
        [cv2.dilate(layer, kernel) for layer in scale_space]
    )
    # Extend the 3 x 3 spatial comparison to the scales directly above and below.
    spatial_maximum = neighborhood_maximum.copy()
    neighborhood_maximum[1:] = np.maximum(
        neighborhood_maximum[1:], spatial_maximum[:-1]
    )
    neighborhood_maximum[:-1] = np.maximum(
        neighborhood_maximum[:-1], spatial_maximum[1:]
    )
    is_peak = (scale_space == neighborhood_maximum) & (scale_space > threshold)
    return np.argwhere(is_peak)


def overlap_fraction(first_blob, second_blob):
    first_radius = first_blob["radius"]
    second_radius = second_blob["radius"]
    distance = np.hypot(
        first_blob["x"] - second_blob["x"], first_blob["y"] - second_blob["y"]
    )
    if distance >= first_radius + second_radius:
        return 0.0
    if distance <= abs(first_radius - second_radius):
        return 1.0

    first_angle = np.arccos(
        (distance**2 + first_radius**2 - second_radius**2)
        / (2 * distance * first_radius)
    )
    second_angle = np.arccos(
        (distance**2 + second_radius**2 - first_radius**2)
        / (2 * distance * second_radius)
    )
    triangle_area = 0.5 * np.sqrt(
        (-distance + first_radius + second_radius)
        * (distance + first_radius - second_radius)
        * (distance - first_radius + second_radius)
        * (distance + first_radius + second_radius)
    )
    intersection_area = (
        first_radius**2 * first_angle
        + second_radius**2 * second_angle
        - triangle_area
    )
    return intersection_area / (np.pi * min(first_radius, second_radius) ** 2)


def prune_overlapping(blobs, overlap):
    # Strongest blobs are kept first; weaker blobs that overlap them are dropped.
    kept_blobs = []
    for blob in sorted(blobs, key=lambda blob: blob["response"], reverse=True):
        if all(
            overlap_fraction(blob, kept_blob) <= overlap
            for kept_blob in kept_blobs
        ):
            kept_blobs.append(blob)
    return kept_blobs


def detect_blobs(image, min_sigma, max_sigma, num_sigma, threshold, overlap):
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    scale_space = log_scale_space(image, sigmas)

    blobs = []
    # A bright blob gives a negative Laplacian at its centre, a dark blob a positive one.
    for polarity, responses in (("bright", -scale_space), ("dark", scale_space)):
        polarity_blobs = [
            {
                "polarity": polarity,
                "x": int(column),
                "y": int(row),
                "sigma": float(sigmas[scale_index]),
                "radius": float(np.sqrt(2) * sigmas[scale_index]),
                "response": float(responses[scale_index, row, column]),
            }
            for scale_index, row, column in local_maxima(responses, threshold)
        ]
        blobs.extend(prune_overlapping(polarity_blobs, overlap))

    return blobs


def summarize_blobs(blobs):
    radii = np.array([blob["radius"] for blob in blobs])
    bright_count = sum(blob["polarity"] == "bright" for blob in blobs)
    return {
        "blobs": len(blobs),
        "bright": bright_count,
        "dark": len(blobs) - bright_count,
        "min_radius": radii.min() if len(blobs) else np.nan,
        "median_radius": np.median(radii) if len(blobs) else np.nan,
        "max_radius": radii.max() if len(blobs) else np.nan,
    }


def draw_blobs(axis, color_image, blobs):
    axis.imshow(color_image)
    outline = [patheffects.withStroke(linewidth=2.6, foreground="black")]
    for blob in blobs:
        axis.add_patch(
            Circle(
                (blob["x"], blob["y"]),
                blob["radius"],
                fill=False,
                linewidth=1.3,
                edgecolor=POLARITY_COLORS[blob["polarity"]],
                path_effects=outline,
            )
        )
    axis.set_xlim(-0.5, color_image.shape[1] - 0.5)
    axis.set_ylim(color_image.shape[0] - 0.5, -0.5)
    axis.axis("off")


def polarity_legend_handles():
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=9,
            markerfacecolor=color,
            markeredgecolor="black",
            label=f"{polarity.capitalize()} blob",
        )
        for polarity, color in POLARITY_COLORS.items()
    ]


def save_detections(color_images, detections, output_directory):
    figure, axes = plt.subplots(
        2, len(color_images), figsize=(14, 7.6), layout="constrained"
    )

    for column, (image_name, color_image) in enumerate(color_images.items()):
        blobs = detections[image_name]
        summary = summarize_blobs(blobs)

        axes[0, column].imshow(color_image)
        axes[0, column].set_title(f"Original: {image_name}")
        axes[0, column].axis("off")

        draw_blobs(axes[1, column], color_image, blobs)
        axes[1, column].set_title(
            f"Blobs: {summary['blobs']} "
            f"({summary['bright']} bright, {summary['dark']} dark)"
        )

    figure.legend(
        handles=polarity_legend_handles(),
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    figure.suptitle("Blob detection with the Laplacian of Gaussian")
    output_path = output_directory / "detections.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Detections saved to: {output_path}")


def save_parameter_comparison(color_image, image, output_directory):
    figure, axes = plt.subplots(2, 3, figsize=(11, 8), layout="constrained")

    for axis, (label, changes) in zip(axes.ravel(), PARAMETER_CONFIGURATIONS):
        parameters = {**BASELINE, **changes}
        blobs = detect_blobs(image, **parameters)
        draw_blobs(axis, color_image, blobs)
        axis.set_title(
            f"{label}\n"
            f"sigma={parameters['min_sigma']}-{parameters['max_sigma']}, "
            f"threshold={parameters['threshold']}, "
            f"overlap={parameters['overlap']}\n"
            f"blobs={len(blobs)}",
            fontsize=9,
        )
        print(f"{label}: {parameters}, blobs={len(blobs)}")

    figure.legend(
        handles=polarity_legend_handles(),
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    figure.suptitle(
        f"Effect of blob detection parameters on the {PARAMETER_IMAGE} image"
    )
    output_path = output_directory / "parameter_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Parameter comparison saved to: {output_path}")


def save_threshold_sensitivity(images, output_directory):
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")

    for image_name, image in images.items():
        counts = [
            len(detect_blobs(image, **{**BASELINE, "threshold": threshold}))
            for threshold in SWEEP_THRESHOLDS
        ]
        color, marker = IMAGE_STYLES[image_name]
        axis.plot(
            SWEEP_THRESHOLDS,
            counts,
            marker=marker,
            markersize=5,
            linewidth=2,
            color=color,
            label=image_name.capitalize(),
        )
        values = ", ".join(str(count) for count in counts)
        print(f"{image_name} blob counts over thresholds: {values}")

    axis.axvline(
        BASELINE["threshold"], color="gray", linestyle="--", linewidth=0.8
    )
    axis.set(
        title="Number of detected blobs for different thresholds",
        xlabel="Threshold on the scale-normalised LoG response",
        ylabel="Number of blobs",
        yscale="symlog",
    )
    axis.set_yticks([0, 1, 2, 5, 10, 20, 50, 100, 200])
    axis.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100", "200"])
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    output_path = output_directory / "threshold_sensitivity.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Threshold sensitivity saved to: {output_path}")


def save_size_distribution(detections, output_directory):
    sigmas = np.linspace(
        BASELINE["min_sigma"], BASELINE["max_sigma"], BASELINE["num_sigma"]
    )
    radii = np.sqrt(2) * sigmas
    figure, axes = plt.subplots(
        1,
        len(detections),
        figsize=(14, 3.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for axis, (image_name, blobs) in zip(axes, detections.items()):
        counts = [
            sum(np.isclose(blob["radius"], radius) for blob in blobs)
            for radius in radii
        ]
        axis.bar(radii, counts, width=1.0, color="#2a78d6")
        axis.set_title(image_name.capitalize())
        axis.set_xlabel("Blob radius (pixels)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Number of blobs")
    figure.suptitle("Blob sizes with the baseline parameters")
    output_path = output_directory / "size_distribution.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Size distribution saved to: {output_path}")


def save_blob_table(detections, output_directory):
    output_path = output_directory / "blobs.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["image", "polarity", "x", "y", "sigma", "radius", "response"]
        )
        for image_name, blobs in detections.items():
            for blob in blobs:
                writer.writerow(
                    [
                        image_name,
                        blob["polarity"],
                        blob["x"],
                        blob["y"],
                        f"{blob['sigma']:.2f}",
                        f"{blob['radius']:.2f}",
                        f"{blob['response']:.4f}",
                    ]
                )
    print(f"Blob positions and sizes saved to: {output_path}")


def save_metrics(detections, output_directory):
    output_path = output_directory / "metrics.csv"
    summaries = {
        image_name: summarize_blobs(blobs)
        for image_name, blobs in detections.items()
    }
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image",
                "blobs",
                "bright",
                "dark",
                "min_radius",
                "median_radius",
                "max_radius",
            ]
        )
        for image_name, summary in summaries.items():
            writer.writerow(
                [
                    image_name,
                    summary["blobs"],
                    summary["bright"],
                    summary["dark"],
                    f"{summary['min_radius']:.2f}",
                    f"{summary['median_radius']:.2f}",
                    f"{summary['max_radius']:.2f}",
                ]
            )

    print(f"Metrics saved to: {output_path}")
    print("\nimage   | blobs | bright | dark | radius min / median / max")
    print("--------|-------|--------|------|--------------------------")
    for image_name, summary in summaries.items():
        print(
            f"{image_name:<7} | {summary['blobs']:>5} | {summary['bright']:>6} | "
            f"{summary['dark']:>4} | {summary['min_radius']:>6.2f} / "
            f"{summary['median_radius']:>6.2f} / {summary['max_radius']:>6.2f}"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = (
        project_root
        / "data"
        / "intel-image-classification"
        / "seg_train"
        / "seg_train"
    )
    output_directory = project_root / "results" / "blob"
    output_directory.mkdir(parents=True, exist_ok=True)

    color_images = {}
    images = {}
    detections = {}
    print("Selected images:")
    for image_name, (category, file_name) in IMAGE_FILES.items():
        image_path = dataset_directory / category / file_name
        print(f"  {image_name}: {image_path.relative_to(project_root)}")
        color_images[image_name], images[image_name] = load_image(image_path)
        detections[image_name] = detect_blobs(images[image_name], **BASELINE)

    print(f"Baseline parameters: {BASELINE}")
    save_detections(color_images, detections, output_directory)
    save_parameter_comparison(
        color_images[PARAMETER_IMAGE], images[PARAMETER_IMAGE], output_directory
    )
    save_threshold_sensitivity(images, output_directory)
    save_size_distribution(detections, output_directory)
    save_blob_table(detections, output_directory)
    save_metrics(detections, output_directory)


if __name__ == "__main__":
    main()
