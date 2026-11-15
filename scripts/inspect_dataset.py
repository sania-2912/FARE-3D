
import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


SUPPORTED_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd")


def is_medical_image(path):
    return str(path).lower().endswith(SUPPORTED_EXTENSIONS)


def discover_files(directory):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and is_medical_image(path)
    )


def calculate_statistics(array):
    array = array.astype(np.float32)

    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p0_5": float(np.percentile(array, 0.5)),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_5": float(np.percentile(array, 99.5)),
    }


def inspect_volume(path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    statistics = calculate_statistics(array)

    return {
        "path": str(path),
        "shape": tuple(array.shape),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "statistics": statistics,
    }


def get_case_id(path, modality_dir):
    """
    Obtain the relative path from the modality directory.
    This helps match cases even when files are nested.
    """
    return path.relative_to(modality_dir)


def match_pairs(cbct_dir, ct_dir):
    cbct_files = discover_files(cbct_dir)
    ct_files = discover_files(ct_dir)

    cbct_map = {
        get_case_id(path, Path(cbct_dir)): path
        for path in cbct_files
    }

    ct_map = {
        get_case_id(path, Path(ct_dir)): path
        for path in ct_files
    }

    common_ids = sorted(set(cbct_map) & set(ct_map))

    missing_ct = sorted(set(cbct_map) - set(ct_map))
    missing_cbct = sorted(set(ct_map) - set(cbct_map))

    return (
        [(case_id, cbct_map[case_id], ct_map[case_id])
         for case_id in common_ids],
        missing_ct,
        missing_cbct,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SynthRAD CBCT and CT volumes"
    )

    parser.add_argument(
        "--cbct",
        required=True,
        help="Path to CBCT directory"
    )

    parser.add_argument(
        "--ct",
        required=True,
        help="Path to CT directory"
    )

    parser.add_argument(
        "--output",
        default="data/dataset_inspection.csv",
        help="Output CSV report"
    )

    args = parser.parse_args()

    cbct_dir = Path(args.cbct)
    ct_dir = Path(args.ct)

    pairs, missing_ct, missing_cbct = match_pairs(
        cbct_dir, ct_dir
    )

    print("\n========== DATASET SUMMARY ==========")
    print(f"CBCT-CT matched pairs: {len(pairs)}")
    print(f"CBCT without CT: {len(missing_ct)}")
    print(f"CT without CBCT: {len(missing_cbct)}")

    if not pairs:
        print("\nNo matching pairs found.")
        print("Check the directory structure and filenames.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_rows = []

    for case_id, cbct_path, ct_path in tqdm(
        pairs, desc="Inspecting paired volumes"
    ):
        try:
            cbct = inspect_volume(cbct_path)
            ct = inspect_volume(ct_path)

            same_shape = cbct["shape"] == ct["shape"]
            same_spacing = np.allclose(
                cbct["spacing"], ct["spacing"], atol=1e-4
            )
            same_origin = np.allclose(
                cbct["origin"], ct["origin"], atol=1e-4
            )
            same_direction = np.allclose(
                cbct["direction"], ct["direction"], atol=1e-5
            )

            row = {
                "case_id": str(case_id),
                "cbct_path": cbct["path"],
                "ct_path": ct["path"],
                "cbct_shape": str(cbct["shape"]),
                "ct_shape": str(ct["shape"]),
                "cbct_spacing": str(cbct["spacing"]),
                "ct_spacing": str(ct["spacing"]),
                "same_shape": same_shape,
                "same_spacing": same_spacing,
                "same_origin": same_origin,
                "same_direction": same_direction,
            }

            for modality, result in (
                ("cbct", cbct),
                ("ct", ct),
            ):
                for key, value in result["statistics"].items():
                    row[f"{modality}_{key}"] = value

            report_rows.append(row)

        except Exception as error:
            print(f"\nError processing {case_id}: {error}")

    if report_rows:
        fieldnames = list(report_rows[0].keys())

        with open(
            output_path, "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n========== GEOMETRY SUMMARY ==========")

    for key in (
        "same_shape",
        "same_spacing",
        "same_origin",
        "same_direction",
    ):
        count = sum(row[key] for row in report_rows)
        print(f"{key}: {count}/{len(report_rows)}")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
