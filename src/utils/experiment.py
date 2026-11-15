
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def create_experiment_directory(
    experiment_name,
    base_dir="experiments"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = (
        Path(base_dir)
        / experiment_name
        / timestamp
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (experiment_dir / "checkpoints").mkdir()
    (experiment_dir / "predictions").mkdir()
    (experiment_dir / "visualizations").mkdir()
    (experiment_dir / "metrics").mkdir()

    return experiment_dir


def save_config(config, experiment_dir):

    output_path = Path(experiment_dir) / "config.yaml"

    with open(output_path, "w") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False
        )

    return output_path


def save_metadata(metadata, experiment_dir):

    output_path = Path(experiment_dir) / "metadata.json"

    with open(output_path, "w") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            default=str
        )

    return output_path


def snapshot_config(config_path, experiment_dir):

    source = Path(config_path)
    destination = Path(experiment_dir) / source.name

    shutil.copy2(
        source,
        destination
    )

    return destination
