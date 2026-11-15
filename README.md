
# FARE-3D
### Frequency-Anchored Residual Evolution for 3D CBCT-to-synthetic-CT Synthesis

## 1. Project Overview

FARE-3D is a deep learning framework for generating synthetic CT (sCT) volumes from Cone-Beam CT (CBCT) scans.

The project aims to address major challenges in CBCT-to-CT synthesis, including:

- Scatter and beam-hardening artifacts.
- Intensity and HU inaccuracies.
- Loss of fine anatomical structures.
- Poor preservation of high-frequency details.
- Difficulty maintaining anatomical consistency during reconstruction.

The framework combines 3D convolutional networks, frequency-aware reconstruction, Vision Transformers and conditional flow matching.

## 2. Research Objective

The primary objective is to investigate whether separating low-frequency intensity correction from high-frequency anatomical refinement can improve CBCT-to-sCT reconstruction.

The framework is designed to:

1. Establish an anatomically consistent regression-based CT anchor.
2. Model smooth intensity corrections explicitly.
3. Preserve fine anatomical structures through frequency-aware processing.
4. Refine high-frequency residuals using conditional flow matching.
5. Prevent the generative refinement stage from directly modifying low-frequency HU information.

## 3. Proposed Architecture

FARE-3D consists of five major components.

### Stage A: 3D Regression Anchor

A 3D Residual U-Net predicts an initial synthetic CT volume from the input CBCT.

It provides the primary reconstruction and establishes the anatomical and intensity foundation for subsequent refinement.

### Smooth-Field Correction

A DCT-based smooth-field head predicts low-order spatial corrections.

It is designed to model broad intensity variations, including scatter-related cupping and low-frequency HU offsets.

### High-Frequency Detail Head

A dedicated detail head predicts high-frequency information.

A frequency projection constrains its output to reduce unintended low-frequency modifications.

### 3D Anatomy Encoder

A 3D Vision Transformer extracts global anatomical context from the CBCT volume.

Its tokens provide conditioning information to the refinement network.

### Artifact Severity Encoder

A convolutional encoder learns a latent representation of CBCT artifact characteristics.

This representation conditions the generative refinement process.

### Stage B: Conditional Flow-Matching Refiner

A compact 3D Diffusion Transformer predicts a velocity field for high-frequency residual evolution.

The refiner uses:
- Anatomy tokens.
- Artifact severity embeddings.
- Time embeddings.
- High-frequency residual inputs.

A frequency projection constrains the predicted velocity to the high-frequency subspace.

The final reconstruction is:

    Y_final = Y_anchor + P_H(R_flow)

where P_H denotes the high-frequency projection.

## 4. Mathematical Formulation

### Smooth-field correction

    B_smooth(x,y,z) =
        sum a_ijk * phi_i(x) * phi_j(y) * phi_k(z)

The low-order DCT basis represents smooth spatial intensity variations.

### High-frequency projection

    P_H(R) = R - L(R)

where L(R) is the selected low-frequency approximation.

### Final reconstruction

    Y_final = Y_A + P_H(R_flow)

This constrains the refinement stage from directly changing the low-frequency component of the anchor.

### Flow-matching objective

    L_FM = E[ ||v_theta(x_t,t,c) - (x_1-x_0)||^2 ]

where:

- x_0: initial high-frequency noise.
- x_1: target high-frequency residual.
- x_t: interpolated residual at time t.
- v_theta: predicted velocity field.
- c: conditioning information.

## 5. Dataset

The project uses the SynthRAD2025 Task 2 dataset.

The dataset contains paired CBCT and CT volumes for model development and evaluation.

All data preprocessing, volume alignment and intensity handling are performed while preserving the relevant medical-image geometry and intensity information.

Dataset structure and case splits are configured separately.

## 6. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Medical Imaging | SimpleITK, Nibabel |
| Numerical Computing | NumPy, SciPy |
| Architecture | 3D U-Net, 3D ViT, 3D DiT |
| Generative Modeling | Conditional Flow Matching |
| Frequency Processing | DCT, 3D Filtering |
| Visualization | Matplotlib |
| Evaluation | Scikit-image |
| Configuration | YAML |

## 7. Project Structure

    FARE-3D/
    ├── configs/
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── splits/
    ├── src/
    │   ├── data/
    │   ├── models/
    │   ├── losses/
    │   ├── training/
    │   ├── evaluation/
    │   └── utils/
    ├── scripts/
    ├── experiments/
    ├── checkpoints/
    ├── logs/
    ├── requirements.txt
    └── README.md

## 8. Experimental Strategy

The framework will be evaluated through progressively more complex experiments.

1. 3D U-Net regression baseline.
2. Frequency-aware Stage A anchor.
3. Anatomy and artifact conditioning.
4. Conditional flow-matching refinement.
5. Complete FARE-3D framework.

Ablation studies will investigate the contribution of individual components.

## 9. Evaluation Metrics

The following metrics will be considered:

- MAE: Mean Absolute Error.
- RMSE: Root Mean Square Error.
- PSNR: Peak Signal-to-Noise Ratio.
- SSIM: Structural Similarity Index.
- LPIPS: Learned Perceptual Image Patch Similarity.

Evaluation should be performed on reconstructed volumes in the appropriate original intensity scale.

## 10. Research Considerations

The following aspects require experimental validation:

- Effectiveness of the frequency projection.
- Preservation of low-frequency HU accuracy.
- Improvement in anatomical detail.
- Stability of flow-matching refinement.
- Computational efficiency of 3D attention.
- Generalization across different CBCT artifact patterns.

The proposed architecture and its potential improvements are research hypotheses that will be evaluated through controlled experiments and ablation studies.

## 11. Project Status

Current phase: Project structure and implementation.

Planned phases:

1. Complete implementation.
2. Integration and debugging.
3. Dataset preparation and validation.
4. Baseline training.
5. Full FARE-3D training.
6. Quantitative evaluation and ablation studies.
7. Research analysis and documentation.

---

**FARE-3D**  
Frequency-Anchored Residual Evolution for 3D CBCT-to-synthetic-CT Synthesis
