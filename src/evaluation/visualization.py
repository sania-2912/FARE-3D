
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def load_volume(path):

    image = sitk.ReadImage(str(path))

    return sitk.GetArrayFromImage(
        image
    ).astype(np.float32)


def get_slices(volume, indices=None):
    """
    Extract axial, coronal and sagittal slices.

    SimpleITK arrays use [D, H, W] = [Z, Y, X].
    """

    depth, height, width = volume.shape

    if indices is None:
        z = depth // 2
        y = height // 2
        x = width // 2
    else:
        z, y, x = indices

    if not (
        0 <= z < depth
        and 0 <= y < height
        and 0 <= x < width
    ):
        raise ValueError("Slice indices are outside volume dimensions.")

    return {
        "Axial": volume[z, :, :],
        "Coronal": volume[:, y, :],
        "Sagittal": volume[:, :, x]
    }


def visualize_comparison(
    cbct,
    ground_truth,
    anchor,
    synthetic_ct,
    output_path,
    slice_indices=None,
    intensity_window=(-1000, 2000)
):
    """
    Generate a five-row, three-column comparison figure.

    Rows:
        CBCT
        Ground-truth CT
        Stage A anchor
        Final Stage B sCT
        Absolute error

    Columns:
        Axial
        Coronal
        Sagittal
    """

    volumes = {
        "CBCT": np.asarray(cbct),
        "Ground Truth CT": np.asarray(ground_truth),
        "Stage A Anchor": np.asarray(anchor),
        "Final sCT": np.asarray(synthetic_ct)
    }

    reference_shape = volumes["Ground Truth CT"].shape

    for name, volume in volumes.items():

        if volume.shape != reference_shape:
            raise ValueError(
                f"{name} shape {volume.shape} does not match "
                f"reference shape {reference_shape}."
            )

        if not np.isfinite(volume).all():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

    lower, upper = intensity_window

    if lower >= upper:
        raise ValueError("Invalid intensity window.")

    errors = np.abs(
        volumes["Final sCT"] - volumes["Ground Truth CT"]
    )

    slices = {
        name: get_slices(volume, slice_indices)
        for name, volume in volumes.items()
    }

    error_slices = get_slices(errors, slice_indices)

    row_names = [
        "CBCT",
        "Ground Truth CT",
        "Stage A Anchor",
        "Final sCT",
        "Absolute Error"
    ]

    column_names = [
        "Axial",
        "Coronal",
        "Sagittal"
    ]

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(15, 20)
    )

    for column, plane in enumerate(column_names):

        for row, name in enumerate(row_names):

            ax = axes[row, column]

            if name == "Absolute Error":

                image = error_slices[plane]

                im = ax.imshow(
                    image,
                    cmap="magma",
                    origin="lower"
                )

            else:

                image = slices[name][plane]

                im = ax.imshow(
                    image,
                    cmap="gray",
                    vmin=lower,
                    vmax=upper,
                    origin="lower"
                )

            ax.set_title(
                f"{name} - {plane}",
                fontsize=10
            )

            ax.axis("off")

            fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

    fig.suptitle(
        "FARE-3D: CBCT to Synthetic CT Comparison",
        fontsize=16
    )

    plt.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Visualization saved to: {output_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Visualize FARE-3D reconstruction results."
    )

    parser.add_argument("--cbct", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--prediction", required=True)

    parser.add_argument(
        "--output",
        default="experiments/full_model/comparison.png"
    )

    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=[-1000, 2000]
    )

    parser.add_argument(
        "--slices",
        nargs=3,
        type=int,
        default=None,
        metavar=("Z", "Y", "X")
    )

    args = parser.parse_args()

    visualize_comparison(
        cbct=load_volume(args.cbct),
        ground_truth=load_volume(args.target),
        anchor=load_volume(args.anchor),
        synthetic_ct=load_volume(args.prediction),
        output_path=args.output,
        slice_indices=args.slices,
        intensity_window=tuple(args.window)
    )


if __name__ == "__main__":
    main()
