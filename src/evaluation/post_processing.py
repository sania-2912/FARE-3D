from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import save_volume


def denormalize_prediction(
    prediction,
    normalization_parameters
):
    """
    Convert a normalized prediction from [-1, 1]
    back to its original intensity range.
    """

    lower = normalization_parameters["lower_bound"]
    upper = normalization_parameters["upper_bound"]

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().float().cpu().numpy()

    prediction = np.asarray(prediction)

    prediction = (
        (prediction + 1.0) / 2.0
    ) * (upper - lower) + lower

    return prediction


def postprocess_prediction(
    prediction,
    normalization_parameters,
    clip_to_window=True
):
    """
    Restore the prediction to its original intensity scale.
    """

    prediction = denormalize_prediction(
        prediction,
        normalization_parameters
    )

    if clip_to_window:

        lower = normalization_parameters["lower_bound"]
        upper = normalization_parameters["upper_bound"]

        prediction = np.clip(
            prediction,
            lower,
            upper
        )

    return prediction.astype(np.float32)


def save_sct_prediction(
    prediction,
    reference_image,
    output_path,
    normalization_parameters,
    clip_to_window=True
):
    """
    Save a reconstructed sCT volume using the reference
    SimpleITK image geometry.
    """

    prediction = postprocess_prediction(
        prediction,
        normalization_parameters,
        clip_to_window=clip_to_window
    )

    # Remove batch and channel dimensions.
    if prediction.ndim == 5:
        prediction = prediction[0, 0]

    elif prediction.ndim == 4:
        prediction = prediction[0]

    if prediction.ndim != 3:
        raise ValueError(
            "Expected a 3D prediction after removing batch/channel dimensions."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_volume(
        prediction,
        reference_image,
        output_path
    )

    return output_path
