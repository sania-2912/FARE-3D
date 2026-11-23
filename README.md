# FARE-3D
### Frequency-Anchored Residual Evolution for 3D CBCT-to-synthetic-CT Synthesis

## 1. Project Overview

FARE-3D is a deep learning framework for generating synthetic CT (sCT) volumes from Cone-Beam CT (CBCT) scans.

The project investigates major challenges in CBCT-to-CT synthesis:

- Scatter and beam-hardening artifacts.
- Intensity and Hounsfield Unit (HU) inaccuracies.
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
5. Constrain the generative refinement stage from directly modifying low-frequency HU information.

## 3. Proposed Architecture

<img width="2880" height="1744" alt="fare3d_merged" src="https://github.com/user-attachments/assets/14826b61-45c1-4799-bc2e-fbbc12101944" />

FARE-3D consists of two main reconstruction stages and two conditioning encoders.

### Stage A: 3D Regression Anchor

A 3D Residual U-Net predicts an initial synthetic CT volume from the input CBCT.

It establishes the anatomical and intensity foundation for subsequent refinement.

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

A frequency projection constrains the predicted velocity to the intended high-frequency subspace.

The final reconstruction is:

\[
Y_{final}=Y_{anchor}+P_H(R_{flow})
\]

where \(P_H\) denotes the high-frequency projection.

## 4. Mathematical Formulation

### Smooth-field correction

\[
B_{smooth}(x,y,z)=
\sum a_{ijk}\phi_i(x)\phi_j(y)\phi_k(z)
\]

The low-order DCT basis represents smooth spatial intensity variations.

### High-frequency projection

\[
P_H(R)=R-\mathcal L(R)
\]

where \(\mathcal L(R)\) is the selected low-frequency approximation.

### Final reconstruction

\[
Y_{final}=Y_A+P_H(R_{flow})
\]

### Flow-matching objective

\[
\mathcal L_{FM}=
\mathbb E[
\|v_\theta(x_t,t,c)-(x_1-x_0)\|^2
]
\]

where:

- \(x_0\): initial high-frequency noise.
- \(x_1\): target high-frequency residual.
- \(x_t\): interpolated residual at time \(t\).
- \(v_\theta\): predicted velocity field.
- \(c\): conditioning information.

## 5. Dataset

The project uses the SynthRAD2025 Task 2 dataset.

The dataset contains paired CBCT and CT volumes for model development and evaluation.

The expected directory layout is:

```text
SynthRAD2025_Task2/
├── CBCT/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── CT/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

Nested case directories are also supported, provided that the relative directory and filenames match between CBCT and CT.

The project supports `.nii.gz`, `.nii`, `.mha` and `.mhd` medical image formats in its dataset discovery scripts.

**Important:** Keep the dataset locally. Do not upload medical imaging data to the public GitHub repository.

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

---

# 7. Installation and Setup

The following instructions use Windows PowerShell and VS Code.

### Step 1: Clone the repository

```bash
git clone https://github.com/sania-2912/FARE-3D.git
cd FARE-3D
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install PyTorch using the installation command appropriate for your CUDA version or CPU environment. Refer to the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

Verify PyTorch and GPU availability:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is available, check the GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')"
```

**Note:** The configured training patch size and 3D attention layers can require substantial GPU memory. Actual GPU compatibility and batch-size requirements must be validated before training.

---

# 8. Dataset Preparation

### Step 1: Place the dataset

Download SynthRAD2025 Task 2 and place the paired modality directories in the dataset root.

For example:

```text
D:/Datasets/SynthRAD2025_Task2/
├── CBCT/
└── CT/
```

### Step 2: Inspect the dataset

Run the dataset inspection script:

```bash
python scripts/inspect_dataset.py --cbct "D:/Datasets/SynthRAD2025_Task2/CBCT" --ct "D:/Datasets/SynthRAD2025_Task2/CT"
```

The script checks:

- Number of matching CBCT–CT pairs.
- Missing CT or CBCT files.
- Volume dimensions.
- Voxel spacing.
- Image origin and direction.
- Intensity statistics and percentiles.

The output report is saved by default to:

```text
data/dataset_inspection.csv
```

You can specify another output location:

```bash
python scripts/inspect_dataset.py --cbct "D:/Datasets/SynthRAD2025_Task2/CBCT" --ct "D:/Datasets/SynthRAD2025_Task2/CT" --output "data/dataset_inspection.csv"
```

Review the report before proceeding. Matching dimensions alone do not establish anatomical registration.

### Step 3: Create dataset splits

Generate training, validation and test splits:

```bash
python scripts/create_splits.py --root "D:/Datasets/SynthRAD2025_Task2" --output "data/splits/split.json" --seed 42
```

The script creates the following split:

| Split | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The split file contains paired CBCT and CT paths.

**Important:**

- Preserve any official dataset splits if they are provided and required by the benchmark.
- Do not use the test set during model training or hyperparameter selection.
- The split script matches CBCT and CT by relative paths and currently derives case IDs from filenames. Check for duplicate filenames in nested directories.

---

# 9. Model Training

FARE-3D uses a two-stage training strategy.

## Stage A: Train the regression anchor

Stage A trains:

- 3D Residual U-Net.
- Smooth-field correction head.
- High-frequency detail head.

Run:

```bash
python -m src.training.train_anchor --config configs/config.yaml --splits data/splits/split.json
```

The training script:

1. Loads the configuration.
2. Loads the training and validation splits.
3. Extracts paired 3D CBCT and CT patches.
4. Initializes the FARE-3D model.
5. Freezes Stage B components.
6. Optimizes the Stage A components.
7. Calculates reconstruction and frequency losses.
8. Saves model checkpoints.

Expected checkpoint directory:

```text
checkpoints/
└── anchor/
    ├── latest.pt
    ├── epoch_5.pt
    ├── epoch_10.pt
    └── ...
```

## Stage B: Train the conditional flow refiner

After Stage A training, use the saved anchor checkpoint to initialize Stage B.

Run:

```bash
python -m src.training.train_refiner --config configs/config.yaml --splits data/splits/split.json --anchor-checkpoint checkpoints/anchor/latest.pt
```

The training script:

1. Loads the pretrained Stage A checkpoint.
2. Freezes the anchor components.
3. Enables the anatomy encoder and severity encoder.
4. Enables the conditional flow-matching refiner.
5. Generates anchor reconstructions.
6. Calculates the high-frequency residual target.
7. Samples noise and interpolation time.
8. Predicts the velocity field.
9. Optimizes the flow-matching objective.
10. Saves refiner checkpoints.

Expected checkpoint directory:

```text
checkpoints/
└── refiner/
    ├── latest.pt
    ├── epoch_5.pt
    ├── epoch_10.pt
    └── ...
```

**Note:** These commands describe the current training-script interfaces. Before the first training run, verify model/configuration compatibility and the frozen-module training mode. The configuration values are initial research settings, not validated optimal hyperparameters.

---

# 10. Inference and Synthetic CT Generation

The evaluation module provides full-volume sliding-window inference through:

```python
generate_sct_full_volume()
```

It accepts a complete CBCT tensor with shape:

```text
[1, 1, D, H, W]
```

Example usage from Python:

```python
import torch

from src.evaluation.inference import generate_sct_full_volume

# Load your trained FARE-3D model and its checkpoint first.
model.eval()

# cbct_tensor must be preprocessed using the same
# intensity handling used during training.

sct = generate_sct_full_volume(
    model=model,
    cbct=cbct_tensor,
    patch_size=(64, 128, 128),
    overlap=0.5,
    inference_steps=8,
    device=device
)
```

The function:

1. Accepts a complete 3D CBCT volume.
2. Divides it into overlapping patches.
3. Generates an sCT prediction for each patch.
4. Blends predictions using Gaussian weights.
5. Reconstructs the complete output volume.

The final output has the same spatial dimensions as the original input tensor.

**Important:** A standalone command-line prediction script is not currently documented here. The example above is an inference API, not a complete executable prediction pipeline. Model checkpoint loading, input normalization, output denormalization and saving with the original medical-image geometry must be handled by the calling application.

---

# 11. Evaluation

The evaluation script calculates metrics for previously generated prediction volumes.

Prepare a JSON file containing prediction and reference CT paths:

```json
[
    {
        "case_id": "case_001",
        "prediction": "experiments/full_model/predictions/case_001_sct.nii.gz",
        "target": "D:/Datasets/SynthRAD2025_Task2/CT/case_001.nii.gz"
    },
    {
        "case_id": "case_002",
        "prediction": "experiments/full_model/predictions/case_002_sct.nii.gz",
        "target": "D:/Datasets/SynthRAD2025_Task2/CT/case_002.nii.gz"
    }
]
```

Save this file as:

```text
experiments/full_model/evaluation_pairs.json
```

Run evaluation:

```bash
python scripts/run_evaluation.py --pairs experiments/full_model/evaluation_pairs.json --output experiments/full_model/metrics.csv --data-range 4000
```

Replace `4000` with the appropriate intensity range for the volumes being evaluated. It is an example value, not a universal HU range.

The evaluation script:

- Loads prediction and reference volumes.
- Checks their physical geometry.
- Calculates MAE.
- Calculates RMSE.
- Calculates PSNR.
- Calculates SSIM.
- Saves per-case metrics and dataset-level mean metrics.

Results are saved to:

```text
experiments/full_model/metrics.csv
```

**Evaluation requirements:**

- Predictions and reference CT volumes must be in the same original intensity units.
- Prediction and target geometry must match.
- Use the test set only for final evaluation.
- The current evaluation script does not calculate LPIPS, DSC or dose-related metrics.

---

# 12. Configuration

Model and training settings are controlled through:

```text
configs/config.yaml
```

Important configuration groups:

| Section | Purpose |
|---|---|
| `project` | Project name, seed and experiment |
| `data` | Dataset paths, patch size and loading |
| `intensity` | Intensity handling and normalization |
| `model.anchor` | Stage A architecture |
| `model.anatomy_encoder` | 3D ViT settings |
| `model.severity_encoder` | Artifact conditioning |
| `model.refiner` | Stage B DiT and flow settings |
| `training` | Optimizer, epochs and GPU settings |
| `loss` | Reconstruction and flow loss weights |
| `evaluation` | Evaluation metric configuration |
| `output` | Checkpoint, log and experiment paths |

Update the dataset root and other environment-specific paths before training.

---

# 13. Project Structure

```text
FARE-3D/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── create_splits.py
│   ├── prepare_dataset.py
│   └── run_evaluation.py
│
├── experiments/
├── checkpoints/
├── logs/
│
├── requirements.txt
└── README.md
```

---

# 14. Experimental Strategy

The framework will be evaluated through progressively more complex experiments:

1. 3D U-Net regression baseline.
2. Frequency-aware Stage A anchor.
3. Anatomy and artifact conditioning.
4. Conditional flow-matching refinement.
5. Complete FARE-3D framework.

Ablation studies will investigate the contribution of individual components.

## 15. Evaluation Metrics

| Metric | Purpose |
|---|---|
| MAE | Mean absolute HU reconstruction error |
| RMSE | Penalizes larger intensity errors |
| PSNR | Reconstruction fidelity |
| SSIM | Structural similarity |
| LPIPS | Perceptual feature similarity |
| DSC | Anatomical segmentation agreement |
| Dose evaluation | Radiotherapy dose-calculation fidelity |

Evaluation should be performed on reconstructed volumes in the appropriate original intensity scale.

## 16. Research Considerations

The following aspects require experimental validation:

- Effectiveness of the frequency projection.
- Preservation of low-frequency HU accuracy.
- Improvement in anatomical detail.
- Stability of flow-matching refinement.
- Computational efficiency of 3D attention.
- Generalization across different CBCT artifact patterns.
- Full-volume inference consistency and patch-boundary effects.

The proposed architecture and its potential improvements are research hypotheses that will be evaluated through controlled experiments and ablation studies.

## 17. Project Status

**Current phase:** Project structure and implementation.

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
*Frequency-Anchored Residual Evolution for 3D CBCT-to-synthetic-CT Synthesis*
