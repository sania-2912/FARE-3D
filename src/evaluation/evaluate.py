import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.evaluation.metrics import calculate_all_metrics


def load_volume(path):
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    return image, volume


def check_geometry(prediction_image, target_image, case_id):

    if prediction_image.GetSize() != target_image.GetSize():
        raise ValueError(
            f"{case_id}: Prediction and target sizes differ."
        )

    if not np.allclose(
        prediction_image.GetSpacing(),
        target_image.GetSpacing(),
        atol=1e-4
    ):
        raise ValueError(
            f"{case_id}: Physical spacing differs."
        )

    if not np.allclose(
        prediction_image.GetOrigin(),
        target_image.GetOrigin(),
        atol=1e-4
    ):
        raise ValueError(
            f"{case_id}: Physical origins differ."
        )

    if not np.allclose(
        prediction_image.GetDirection(),
        target_image.GetDirection(),
        atol=1e-4
    ):
        raise ValueError(
            f"{case_id}: Physical directions differ."
        )


def evaluate_case(case, data_range):

    case_id = case["case_id"]

    prediction_image, prediction = load_volume(
        case["prediction"]
    )

    target_image, target = load_volume(
        case["target"]
    )

    check_geometry(
        prediction_image,
        target_image,
        case_id
    )

    metrics = calculate_all_metrics(
        prediction=prediction,
        target=target,
        data_range=data_range
    )

    return {
        "case_id": case_id,
        **metrics
    }


def evaluate_dataset(pairs_path, output_path, data_range):

    with open(pairs_path, "r") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Evaluation JSON must contain a non-empty list of cases."
        )

    results = []

    for index, case in enumerate(cases):

        print(
            f"Evaluating case {index + 1}/{len(cases)}: "
            f"{case['case_id']}"
        )

        result = evaluate_case(case, data_range)
        results.append(result)

        print(
            f"MAE: {result['MAE']:.3f} | "
            f"RMSE: {result['RMSE']:.3f} | "
            f"PSNR: {result['PSNR']:.3f} | "
            f"SSIM: {result['SSIM']:.4f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metric_names = ["MAE", "RMSE", "PSNR", "SSIM"]

    with open(output_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["case_id"] + metric_names
        )

        writer.writeheader()
        writer.writerows(results)

        mean_metrics = {
            metric: float(
                np.mean([result[metric] for result in results])
            )
            for metric in metric_names
        }

        writer.writerow({
            "case_id": "MEAN",
            **mean_metrics
        })

    print("\nDataset-level mean metrics:")

    for name, value in mean_metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate FARE-3D synthetic CT predictions."
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="JSON file containing prediction and target paths."
    )

    parser.add_argument(
        "--output",
        default="experiments/full_model/metrics.csv"
    )

    parser.add_argument(
        "--data-range",
        type=float,
        required=True,
        help="Intensity range in the volume units being evaluated."
    )

    args = parser.parse_args()

    evaluate_dataset(
        pairs_path=args.pairs,
        output_path=args.output,
        data_range=args.data_range
    )
