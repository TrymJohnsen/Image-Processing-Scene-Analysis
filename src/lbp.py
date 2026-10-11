import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from skimage import data


IMAGE_SIZE = (128, 128)
FACE_CROP = (slice(10, 210), slice(130, 330))
# Clockwise from the top-left neighbour. The first offset is the lowest bit.
NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]
NUMBER_OF_CODES = 256
UNIFORM_COLOR = "#2a78d6"
NON_UNIFORM_COLOR = "#eb6834"


def prepare_image(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def load_forest_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return prepare_image(image)


def compute_lbp(image):
    if image.ndim != 2:
        raise ValueError("LBP expects a grayscale image")

    rows, columns = image.shape
    center = image[1:-1, 1:-1]
    lbp_image = np.zeros(center.shape, dtype=np.uint8)

    # Border pixels lack a full neighbourhood, so only interior pixels get a code.
    for bit, (row_offset, column_offset) in enumerate(NEIGHBOR_OFFSETS):
        neighbor = image[
            1 + row_offset : rows - 1 + row_offset,
            1 + column_offset : columns - 1 + column_offset,
        ]
        lbp_image |= (neighbor >= center).astype(np.uint8) << bit

    return lbp_image


def lbp_histogram(lbp_image):
    histogram = np.zeros(NUMBER_OF_CODES, dtype=np.int64)
    for code in lbp_image.ravel():
        histogram[code] += 1
    return histogram


def uniform_code_mask():
    # A uniform pattern has at most two 0/1 transitions around the circle.
    mask = np.zeros(NUMBER_OF_CODES, dtype=bool)
    for code in range(NUMBER_OF_CODES):
        bits = [(code >> bit) & 1 for bit in range(8)]
        transitions = sum(bits[bit] != bits[(bit + 1) % 8] for bit in range(8))
        mask[code] = transitions <= 2
    return mask


def summarize_histogram(histogram, uniform_mask):
    frequencies = histogram / histogram.sum()
    occupied = frequencies[frequencies > 0]
    entropy = -np.sum(occupied * np.log2(occupied))
    most_common = np.argsort(histogram)[::-1][:3]
    return {
        "uniform_share": frequencies[uniform_mask].sum(),
        "entropy": entropy,
        "most_common": [(int(code), frequencies[code]) for code in most_common],
    }


def plot_histogram(axis, histogram, uniform_mask):
    codes = np.arange(NUMBER_OF_CODES)
    percentages = histogram / histogram.sum() * 100
    colors = np.where(uniform_mask, UNIFORM_COLOR, NON_UNIFORM_COLOR)
    axis.bar(codes, percentages, width=1.0, color=colors)
    axis.set_xlim(-1, NUMBER_OF_CODES)
    axis.set_xticks([0, 32, 64, 96, 128, 160, 192, 224, 255])
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def histogram_legend_handles():
    return [
        Patch(color=UNIFORM_COLOR, label="Uniform pattern"),
        Patch(color=NON_UNIFORM_COLOR, label="Non-uniform pattern"),
    ]


def save_image_comparison(
    image, lbp_image, histogram, uniform_mask, image_name, output_directory
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        width_ratios=[1, 1, 1.6],
        layout="constrained",
    )

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original grayscale")
    axes[0].axis("off")

    axes[1].imshow(lbp_image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("LBP image")
    axes[1].axis("off")

    plot_histogram(axes[2], histogram, uniform_mask)
    axes[2].set(
        title="LBP histogram",
        xlabel="LBP code",
        ylabel="Share of pixels (%)",
    )
    axes[2].legend(handles=histogram_legend_handles(), frameon=False)

    figure.suptitle(f"Local Binary Patterns: {image_name}")
    output_path = output_directory / f"{image_name}_lbp.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Comparison saved to: {output_path}")


def save_histogram_comparison(histograms, uniform_mask, output_directory):
    figure, axes = plt.subplots(
        len(histograms),
        1,
        figsize=(10, 7.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for axis, (image_name, histogram) in zip(axes, histograms.items()):
        plot_histogram(axis, histogram, uniform_mask)
        axis.set_title(image_name.capitalize(), loc="left")
        axis.set_ylabel("Share of pixels (%)")

    axes[0].legend(handles=histogram_legend_handles(), frameon=False, ncols=2)
    axes[-1].set_xlabel("LBP code")
    figure.suptitle("LBP histograms of the three images")
    output_path = output_directory / "histogram_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Histogram comparison saved to: {output_path}")


def save_metrics(summaries, output_directory):
    output_path = output_directory / "metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["image", "uniform_share", "entropy_bits", "most_common_codes"]
        )
        for image_name, summary in summaries.items():
            most_common = "; ".join(
                f"{code} ({code:08b}) {share:.4f}"
                for code, share in summary["most_common"]
            )
            writer.writerow(
                [
                    image_name,
                    f"{summary['uniform_share']:.4f}",
                    f"{summary['entropy']:.4f}",
                    most_common,
                ]
            )

    print(f"Metrics saved to: {output_path}")
    print("\nimage  | uniform share | entropy (bits) | most common codes")
    print("-------|---------------|----------------|------------------")
    for image_name, summary in summaries.items():
        most_common = ", ".join(
            f"{code} ({share:.1%})" for code, share in summary["most_common"]
        )
        print(
            f"{image_name:<6} | {summary['uniform_share']:>13.1%} | "
            f"{summary['entropy']:>14.2f} | {most_common}"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]
    forest_path = (
        project_root
        / "data"
        / "intel-image-classification"
        / "seg_train"
        / "seg_train"
        / "forest"
        / "10037.jpg"
    )
    output_directory = project_root / "results" / "lbp"
    output_directory.mkdir(parents=True, exist_ok=True)

    # The texture and face images are sample images bundled with scikit-image.
    images = {
        "forest": load_forest_image(forest_path),
        "brick": prepare_image(data.brick()),
        "face": prepare_image(data.astronaut()[FACE_CROP]),
    }

    uniform_mask = uniform_code_mask()
    print(
        f"Uniform codes: {uniform_mask.sum()} of {NUMBER_OF_CODES} "
        f"({uniform_mask.mean():.1%})"
    )

    histograms = {}
    summaries = {}
    for image_name, image in images.items():
        lbp_image = compute_lbp(image)
        histogram = lbp_histogram(lbp_image)
        assert histogram.sum() == lbp_image.size

        histograms[image_name] = histogram
        summaries[image_name] = summarize_histogram(histogram, uniform_mask)
        print(
            f"{image_name}: image {image.shape}, LBP image {lbp_image.shape}"
        )
        save_image_comparison(
            image, lbp_image, histogram, uniform_mask, image_name, output_directory
        )

    save_histogram_comparison(histograms, uniform_mask, output_directory)
    save_metrics(summaries, output_directory)


if __name__ == "__main__":
    main()
