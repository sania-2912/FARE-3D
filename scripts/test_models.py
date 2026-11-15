
import torch

from src.models.fare3d import FARE3D

# assert outputs["anatomy_tokens"].shape == (1, 64, 256)
def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def check_shape(name, tensor, expected_shape):
    actual_shape = tuple(tensor.shape)

    assert actual_shape == expected_shape, (
        f"{name}: expected {expected_shape}, "
        f"received {actual_shape}"
    )

    print(f"[PASS] {name}: {actual_shape}")


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # ----------------------------------
    # Initialize model
    # ----------------------------------

    model = FARE3D(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        smooth_basis_order=(4, 4, 4),
        anatomy_embed_dim=256,
        anatomy_depth=6,
        anatomy_heads=8,
        severity_embed_dim=128,
        refiner_hidden_dim=256,
        refiner_depth=6,
        refiner_heads=8,
        use_refiner=True,
    ).to(device)

    model.eval()

    total, trainable = count_parameters(model)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # ----------------------------------
    # Synthetic input volume
    # ----------------------------------

    # Small volume for initial testing.
    # This is NOT our final training patch size.

    batch_size = 1

    spatial_shape = (16, 32, 32)

    cbct = torch.randn(
        batch_size,
        1,
        *spatial_shape,
        device=device,
    )

    noisy_residual = torch.randn_like(cbct)

    time = torch.tensor(
        [0.5],
        device=device,
    )

    print("\nInput shape:", tuple(cbct.shape))

    # ----------------------------------
    # Complete forward pass
    # ----------------------------------

    print("\nRunning FARE-3D forward pass...")

    with torch.inference_mode():

        outputs = model(
            cbct=cbct,
            noisy_residual=noisy_residual,
            time=time,
        )

    print("\nChecking Stage A outputs...")

    expected_image_shape = (
        batch_size,
        1,
        *spatial_shape,
    )

    check_shape(
        "Base prediction",
        outputs["base_prediction"],
        expected_image_shape,
    )

    check_shape(
        "Smooth field",
        outputs["smooth_field"],
        expected_image_shape,
    )

    check_shape(
        "Detail correction",
        outputs["detail"],
        expected_image_shape,
    )

    check_shape(
        "Final anchor",
        outputs["anchor"],
        expected_image_shape,
    )

    # ----------------------------------
    # Conditioning outputs
    # ----------------------------------

    print("\nChecking conditioning outputs...")

    check_shape(
        "Anatomy tokens",
        outputs["anatomy_tokens"],
        (1, 512, 256),
    )

    check_shape(
        "Severity embedding",
        outputs["severity_embedding"],
        (1, 128),
    )

    # ----------------------------------
    # Stage B outputs
    # ----------------------------------

    print("\nChecking Stage B outputs...")

    check_shape(
        "Predicted velocity",
        outputs["velocity"],
        expected_image_shape,
    )

    check_shape(
        "Raw velocity",
        outputs["raw_velocity"],
        expected_image_shape,
    )

    # ----------------------------------
    # Numerical stability
    # ----------------------------------

    print("\nChecking numerical stability...")

    for name, tensor in outputs.items():

        assert torch.isfinite(tensor).all(), (
            f"Non-finite values found in {name}"
        )

        print(f"[PASS] {name} contains finite values")

    # ----------------------------------
    # Verify anchor composition
    # ----------------------------------

    expected_anchor = (
        outputs["base_prediction"]
        + outputs["smooth_field"]
        + outputs["detail"]
    )

    assert torch.allclose(
        outputs["anchor"],
        expected_anchor,
        atol=1e-5,
    )

    print("[PASS] Anchor composition is correct")

    # ----------------------------------
    # Verify hard projection
    # ----------------------------------

    assert torch.allclose(
        outputs["velocity"],
        model.flow_refiner.frequency_projection(
            outputs["raw_velocity"]
        ),
        atol=1e-5,
    )

    print("[PASS] High-frequency projection is applied")

    print("\n================================")
    print("FARE-3D SMOKE TEST PASSED")
    print("================================")


if __name__ == "__main__":
    main()
