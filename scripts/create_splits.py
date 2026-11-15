
import argparse
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


def discover_cases(root_dir, cbct_subdir="CBCT", ct_subdir="CT"):

    root = Path(root_dir)

    cbct_root = root / cbct_subdir
    ct_root = root / ct_subdir

    if not cbct_root.exists():
        raise FileNotFoundError(
            f"CBCT directory not found: {cbct_root}"
        )

    if not ct_root.exists():
        raise FileNotFoundError(
            f"CT directory not found: {ct_root}"
        )

    supported_extensions = (".nii.gz", ".nii", ".mha", ".mhd")

    cbct_files = sorted(
        path for path in cbct_root.rglob("*")
        if path.is_file()
        and str(path).lower().endswith(supported_extensions)
    )

    cases = []

    for cbct_path in cbct_files:

        relative_path = cbct_path.relative_to(cbct_root)
        ct_path = ct_root / relative_path

        if not ct_path.exists():
            print(f"Missing CT pair: {relative_path}")
            continue

        cases.append({
            "case_id": relative_path.name,
            "cbct": str(cbct_path.resolve()),
            "ct": str(ct_path.resolve())
        })

    return cases


def create_splits(
    cases,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):

    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")

    if len(cases) < 3:
        raise ValueError(
            "At least three paired cases are required."
        )

    train_cases, remaining = train_test_split(
        cases,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True
    )

    relative_test_ratio = test_ratio / (
        val_ratio + test_ratio
    )

    val_cases, test_cases = train_test_split(
        remaining,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True
    )

    return {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases
    }


def main():

    parser = argparse.ArgumentParser(
        description="Create paired SynthRAD dataset splits."
    )

    parser.add_argument(
        "--root",
        required=True
    )

    parser.add_argument(
        "--output",
        default="data/splits/split.json"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    cases = discover_cases(args.root)

    if not cases:
        raise RuntimeError(
            "No paired CBCT-CT cases were discovered."
        )

    splits = create_splits(
        cases,
        seed=args.seed
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:
        json.dump(splits, file, indent=4)

    print("Dataset split summary")
    print("---------------------")
    print(f"Total cases: {len(cases)}")
    print(f"Training:    {len(splits['train'])}")
    print(f"Validation:  {len(splits['val'])}")
    print(f"Testing:     {len(splits['test'])}")
    print(f"Saved to:    {output_path}")


if __name__ == "__main__":
    main()
