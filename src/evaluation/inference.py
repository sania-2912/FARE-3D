
import itertools
import torch
import torch.nn.functional as F


def get_patch_starts(length, patch_size, overlap):

    if length <= patch_size:
        return [0]

    stride = max(
        1,
        int(patch_size * (1.0 - overlap))
    )

    starts = list(
        range(0, length - patch_size + 1, stride)
    )

    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)

    return starts


def create_gaussian_weight(patch_size, device):

    axes = []

    for size in patch_size:

        coordinates = torch.linspace(
            -1.0,
            1.0,
            size,
            device=device
        )

        sigma = 0.5

        weights = torch.exp(
            -(coordinates ** 2) / (2 * sigma ** 2)
        )

        axes.append(weights)

    weight = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )

    weight = weight / weight.max()

    return weight.clamp(min=1e-4)[None, None]


@torch.no_grad()
def generate_sct_full_volume(
    model,
    cbct,
    patch_size=(64, 128, 128),
    overlap=0.5,
    inference_steps=8,
    device=None
):
    """
    Generate sCT from a complete 3D CBCT volume
    using overlapping sliding-window inference.

    Input:
        cbct: [1, 1, D, H, W]

    Output:
        sCT: [1, 1, D, H, W]
    """

    if cbct.ndim != 5:
        raise ValueError(
            "Expected CBCT shape [B, C, D, H, W]."
        )

    if cbct.shape[0] != 1:
        raise ValueError(
            "Full-volume inference currently supports batch size 1."
        )

    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    cbct = cbct.to(device)

    original_shape = cbct.shape[2:]

    patch_size = tuple(patch_size)

    padding = []

    for length, target in reversed(
        list(zip(original_shape, patch_size))
    ):
        total = max(0, target - length)

        left = total // 2
        right = total - left

        padding.extend([left, right])

    if any(padding):
        cbct = F.pad(
            cbct,
            padding,
            mode="replicate"
        )

    padded_shape = cbct.shape[2:]

    starts_d = get_patch_starts(
        padded_shape[0],
        patch_size[0],
        overlap
    )

    starts_h = get_patch_starts(
        padded_shape[1],
        patch_size[1],
        overlap
    )

    starts_w = get_patch_starts(
        padded_shape[2],
        patch_size[2],
        overlap
    )

    output_sum = torch.zeros_like(cbct)

    weight_sum = torch.zeros_like(cbct)

    weight = create_gaussian_weight(
        patch_size,
        device
    )

    for d, h, w in itertools.product(
        starts_d,
        starts_h,
        starts_w
    ):

        patch = cbct[
            :,
            :,
            d:d + patch_size[0],
            h:h + patch_size[1],
            w:w + patch_size[2]
        ]

        prediction = generate_sct(
            model,
            patch,
            inference_steps=inference_steps,
            device=device
        )

        output_sum[
            :,
            :,
            d:d + patch_size[0],
            h:h + patch_size[1],
            w:w + patch_size[2]
        ] += prediction * weight

        weight_sum[
            :,
            :,
            d:d + patch_size[0],
            h:h + patch_size[1],
            w:w + patch_size[2]
        ] += weight

    output = output_sum / weight_sum.clamp(min=1e-8)

    # Remove padding and restore original dimensions.
    d_pad = padding[4]
    h_pad = padding[2]
    w_pad = padding[0]

    output = output[
        :,
        :,
        d_pad:d_pad + original_shape[0],
        h_pad:h_pad + original_shape[1],
        w_pad:w_pad + original_shape[2]
    ]

    return output
