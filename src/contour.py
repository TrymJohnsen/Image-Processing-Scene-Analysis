import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D

from blob import BASELINE as BLOB_BASELINE
from blob import IMAGE_FILES, IMAGE_STYLES, detect_blobs, draw_blobs, load_image


# A threshold of None lets Otsu's method choose the value from the image histogram.
BASELINE = {
    "blur_size": 5,
    "threshold": None,
    "min_area": 15,
}
PARAMETER_CONFIGURATIONS = [
    ("Baseline", {}),
    ("Lower threshold", {"threshold": 80}),
    ("Higher threshold", {"threshold": 160}),
    ("No blur", {"blur_size": 1}),
    ("Stronger blur", {"blur_size": 11}),
    ("No minimum area", {"min_area": 0}),
]
PARAMETER_IMAGE = "rocks"
SWEEP_THRESHOLDS = np.arange(20, 231, 10)
SWEEP_BLUR_SIZES = [1, 3, 5, 7, 9, 11, 13, 15]


def contour_colors(number_of_contours):
    # Evenly spaced hues are unique; golden-angle ordering separates neighbours.
    hue_order = np.argsort((np.arange(number_of_contours) * 0.61803398875) % 1)
    hues = hue_order / max(number_of_contours, 1)
    return hsv_to_rgb(np.column_stack([
        hues, np.full(number_of_contours, 0.75), np.ones(number_of_contours)
    ]))


def to_uint8(image):
    return np.round(image * 255).astype(np.uint8)


def find_contours(image, blur_size, threshold, min_area):
    if blur_size > 1:
        image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    if threshold is None:
        threshold, binary_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary_image = cv2.threshold(
            image, threshold, 255, cv2.THRESH_BINARY
        )

    # RETR_LIST returns the outline of every bright region and of every dark hole.
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_area
    ]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours, binary_image, int(threshold)


def describe_contour(contour):
    area = cv2.contourArea(contour)
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]
    else:
        x, y = contour[:, 0, :].mean(axis=0)
    return {
        "x": float(x),
        "y": float(y),
        "area": float(area),
        "perimeter": float(cv2.arcLength(contour, True)),
    }


def summarize_contours(contours):
    descriptions = [describe_contour(contour) for contour in contours]
    areas = np.array([description["area"] for description in descriptions])
    perimeters = np.array(
        [description["perimeter"] for description in descriptions]
    )
    has_contours = len(contours) > 0
    return {
        "contours": len(contours),
        "median_area": np.median(areas) if has_contours else np.nan,
        "max_area": areas.max() if has_contours else np.nan,
        "median_perimeter": np.median(perimeters) if has_contours else np.nan,
        "max_perimeter": perimeters.max() if has_contours else np.nan,
    }


def draw_contours(axis, color_image, contours):
    axis.imshow(color_image)
    outline = [patheffects.withStroke(linewidth=2.6, foreground="black")]
    colors = contour_colors(len(contours))
    for contour, color in zip(contours, colors):
        points = contour[:, 0, :]
        closed_points = np.vstack([points, points[:1]])
        axis.plot(
            closed_points[:, 0],
            closed_points[:, 1],
            color=color,
            linewidth=1.3,
            path_effects=outline,
        )
    axis.set_xlim(-0.5, color_image.shape[1] - 0.5)
    axis.set_ylim(color_image.shape[0] - 0.5, -0.5)
    axis.axis("off")


def save_detections(color_images, results, output_directory):
    figure, axes = plt.subplots(
        3, len(color_images), figsize=(14, 11), layout="constrained"
    )

    for column, (image_name, color_image) in enumerate(color_images.items()):
        contours, binary_image, threshold = results[image_name]

        axes[0, column].imshow(color_image)
        axes[0, column].set_title(f"Original: {image_name}")
        axes[0, column].axis("off")

        axes[1, column].imshow(binary_image, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title(f"Binary image, Otsu threshold {threshold}")
        axes[1, column].axis("off")

        draw_contours(axes[2, column], color_image, contours)
        axes[2, column].set_title(f"Contours: {len(contours)}")

    figure.suptitle("Contour detection with Otsu thresholding")
    output_path = output_directory / "detections.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Detections saved to: {output_path}")


def save_parameter_comparison(color_image, image, output_directory):
    figure, axes = plt.subplots(2, 3, figsize=(11, 8), layout="constrained")

    for axis, (label, changes) in zip(axes.ravel(), PARAMETER_CONFIGURATIONS):
        parameters = {**BASELINE, **changes}
        contours, _, threshold = find_contours(image, **parameters)
        draw_contours(axis, color_image, contours)
        threshold_label = (
            f"Otsu ({threshold})"
            if parameters["threshold"] is None
            else str(threshold)
        )
        axis.set_title(
            f"{label}\n"
            f"blur={parameters['blur_size']}, threshold={threshold_label}, "
            f"min area={parameters['min_area']}\n"
            f"contours={len(contours)}",
            fontsize=9,
        )
        print(
            f"{label}: blur={parameters['blur_size']}, "
            f"threshold={threshold_label}, min_area={parameters['min_area']}, "
            f"contours={len(contours)}"
        )

    figure.suptitle(
        f"Effect of contour detection parameters on the {PARAMETER_IMAGE} image"
    )
    output_path = output_directory / "parameter_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Parameter comparison saved to: {output_path}")


def save_parameter_sensitivity(images, output_directory):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    for image_name, image in images.items():
        color, marker = IMAGE_STYLES[image_name]
        threshold_counts = [
            len(find_contours(image, **{**BASELINE, "threshold": int(value)})[0])
            for value in SWEEP_THRESHOLDS
        ]
        axes[0].plot(
            SWEEP_THRESHOLDS,
            threshold_counts,
            marker=marker,
            markersize=4,
            linewidth=2,
            color=color,
            label=image_name.capitalize(),
        )
        otsu_contours, _, otsu_threshold = find_contours(image, **BASELINE)
        axes[0].plot(
            otsu_threshold,
            len(otsu_contours),
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            linestyle="none",
        )

        blur_counts = [
            len(find_contours(image, **{**BASELINE, "blur_size": size})[0])
            for size in SWEEP_BLUR_SIZES
        ]
        axes[1].plot(
            SWEEP_BLUR_SIZES,
            blur_counts,
            marker=marker,
            markersize=5,
            linewidth=2,
            color=color,
            label=image_name.capitalize(),
        )
        print(
            f"{image_name}: Otsu threshold {otsu_threshold}, "
            f"counts over thresholds "
            f"{', '.join(str(count) for count in threshold_counts)}"
        )
        print(
            f"{image_name}: counts over blur sizes "
            f"{', '.join(str(count) for count in blur_counts)}"
        )

    axes[0].set(
        title="Contours for different threshold values",
        xlabel="Threshold (gray level, 0-255)",
        ylabel="Number of contours",
    )
    axes[1].set(
        title="Contours for different blur sizes",
        xlabel="Gaussian blur kernel size (pixels)",
        ylabel="Number of contours",
        xticks=SWEEP_BLUR_SIZES,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
    )
    labels.append("Otsu threshold")
    axes[0].legend(handles, labels, frameon=False)
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    output_path = output_directory / "parameter_sensitivity.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Parameter sensitivity saved to: {output_path}")


def save_area_perimeter(results, output_directory):
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

    for image_name, (contours, _, _) in results.items():
        descriptions = [describe_contour(contour) for contour in contours]
        color, marker = IMAGE_STYLES[image_name]
        axis.scatter(
            [description["area"] for description in descriptions],
            [description["perimeter"] for description in descriptions],
            color=color,
            marker=marker,
            s=42,
            edgecolors="white",
            linewidths=0.8,
            label=image_name.capitalize(),
            zorder=3,
        )

    # A circle has the shortest possible perimeter for a given area.
    areas = np.geomspace(10, 22500, 100)
    axis.plot(
        areas,
        2 * np.sqrt(np.pi * areas),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Perfect circle",
    )
    axis.set(
        title="Area and perimeter of the detected contours",
        xlabel="Contour area (pixels²)",
        ylabel="Contour perimeter (pixels)",
        xscale="log",
        yscale="log",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    output_path = output_directory / "area_perimeter.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Area and perimeter plot saved to: {output_path}")


def save_blob_comparison(color_images, images, results, output_directory):
    figure, axes = plt.subplots(
        2, len(color_images), figsize=(14, 7.4), layout="constrained"
    )

    for column, (image_name, color_image) in enumerate(color_images.items()):
        blobs = detect_blobs(images[image_name], **BLOB_BASELINE)
        draw_blobs(axes[0, column], color_image, blobs)
        axes[0, column].set_title(f"{image_name.capitalize()}, blobs: {len(blobs)}")

        contours = results[image_name][0]
        draw_contours(axes[1, column], color_image, contours)
        axes[1, column].set_title(
            f"{image_name.capitalize()}, contours: {len(contours)}"
        )
        print(f"{image_name}: {len(blobs)} blobs, {len(contours)} contours")

    figure.suptitle("Blob detection (top) and contour detection (bottom)")
    output_path = output_directory / "blob_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Blob comparison saved to: {output_path}")


def save_contour_table(results, output_directory):
    output_path = output_directory / "contours.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "contour", "x", "y", "area", "perimeter"])
        for image_name, (contours, _, _) in results.items():
            for index, contour in enumerate(contours, start=1):
                description = describe_contour(contour)
                writer.writerow(
                    [
                        image_name,
                        index,
                        f"{description['x']:.1f}",
                        f"{description['y']:.1f}",
                        f"{description['area']:.1f}",
                        f"{description['perimeter']:.1f}",
                    ]
                )
    print(f"Contour positions and sizes saved to: {output_path}")


def save_metrics(results, output_directory):
    output_path = output_directory / "metrics.csv"
    summaries = {
        image_name: summarize_contours(contours)
        for image_name, (contours, _, _) in results.items()
    }
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image",
                "otsu_threshold",
                "contours",
                "median_area",
                "max_area",
                "median_perimeter",
                "max_perimeter",
            ]
        )
        for image_name, summary in summaries.items():
            writer.writerow(
                [
                    image_name,
                    results[image_name][2],
                    summary["contours"],
                    f"{summary['median_area']:.1f}",
                    f"{summary['max_area']:.1f}",
                    f"{summary['median_perimeter']:.1f}",
                    f"{summary['max_perimeter']:.1f}",
                ]
            )

    print(f"Metrics saved to: {output_path}")
    print("\nimage   | Otsu | contours | area median / max | perimeter median / max")
    print("--------|------|----------|-------------------|-----------------------")
    for image_name, summary in summaries.items():
        print(
            f"{image_name:<7} | {results[image_name][2]:>4} | "
            f"{summary['contours']:>8} | "
            f"{summary['median_area']:>7.1f} / {summary['max_area']:>7.1f} | "
            f"{summary['median_perimeter']:>7.1f} / {summary['max_perimeter']:>7.1f}"
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
    output_directory = project_root / "results" / "contour"
    output_directory.mkdir(parents=True, exist_ok=True)

    color_images = {}
    blob_images = {}
    images = {}
    results = {}
    print("Selected images:")
    for image_name, (category, file_name) in IMAGE_FILES.items():
        image_path = dataset_directory / category / file_name
        print(f"  {image_name}: {image_path.relative_to(project_root)}")
        color_images[image_name], blob_images[image_name] = load_image(image_path)
        images[image_name] = to_uint8(blob_images[image_name])
        results[image_name] = find_contours(images[image_name], **BASELINE)

    print(f"Baseline parameters: {BASELINE}")
    save_detections(color_images, results, output_directory)
    save_parameter_comparison(
        color_images[PARAMETER_IMAGE], images[PARAMETER_IMAGE], output_directory
    )
    save_parameter_sensitivity(images, output_directory)
    save_area_perimeter(results, output_directory)
    save_blob_comparison(color_images, blob_images, results, output_directory)
    save_contour_table(results, output_directory)
    save_metrics(results, output_directory)


if __name__ == "__main__":
    main()
