# SBD-INR minimal reproducibility example

This folder contains a minimal, single-image evaluation example for SBD-INR.
It reconstructs the preprocessed Kodak image from a trained coordinate network
and reports bit-error rate (BER), PSNR, SSIM, and RMSE.

## Contents

```text
SBD-INR/
├── checkpoints/
│   ├── README.md
│   └── kodim23_sbd_inr.pth   # released checkpoint (add this file)
├── configs/
│   └── kodim23.json
├── data/
│   ├── README.md
│   └── kodim23.png
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   └── model.py
├── .gitignore
├── expected_results.json
├── README.md
└── requirements.txt
```

## Environment

Python 3.10 or newer is recommended. Create an isolated environment and install
the pinned dependencies:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If a different CUDA build of PyTorch is required, install the matching `torch`
and `torchvision` wheels from the official PyTorch instructions while keeping
their versions compatible.

## Checkpoint

Copy the checkpoint produced for `kodim23.png` by the SBD-INR `ours`
configuration to:

```text
checkpoints/kodim23_sbd_inr.pth
```

The repository currently expects the checkpoint dictionary saved by
`lossless_bitplane_inr.py`, including `model_state_dict`. For a public GitHub
release, commit the file directly if it is below GitHub's file-size limit;
otherwise use Git LFS or attach it to a GitHub Release and document the URL.

## Evaluation

Run from the repository root:

```bash
python src/evaluate.py
```

To force CPU inference or reduce peak memory:

```bash
python src/evaluate.py --device cpu --batch-pixels 1024
```

To save the reconstructed image:

```bash
python src/evaluate.py --save-reconstruction outputs/kodim23_reconstructed.png
```

The expected exact-reconstruction output is:

```text
Bit errors: 0/1572864
BER:        0.000000000000e+00
PSNR:      inf
SSIM:       1.000000
RMSE:       0.000000
```

The machine-readable values are also recorded in `expected_results.json`.

## Evaluation definition

The script follows the training data path used for this checkpoint:

1. Load `kodim23.png` as RGB.
2. Center-crop the 768 x 512 image to 512 x 512.
3. Resize to 256 x 256 using bilinear interpolation with antialiasing and round
   to 8-bit integers.
4. Query the 2D DB on coordinates ordered as `(x, y)` in `[-1, 1]`.
5. Predict eight bit planes ordered from LSB to MSB with FiLM-PBD.
6. Threshold each logit at zero, equivalent to applying sigmoid and using a
   probability threshold of 0.5.
7. Recombine the binary planes with weights `2^b` and compute metrics against
   the identically preprocessed target.

BER is the fraction of incorrect binary values over all pixels, bit planes, and
RGB channels. PSNR and RMSE are computed on 8-bit RGB integer values. SSIM is
computed with `channel_axis=2` and `data_range=255`.

## Scope

This is an inference-only reproducibility example for one image and one trained
checkpoint. It does not include training, dataset-wide evaluation, entropy
coding, or a claim about end-to-end compression rate.

