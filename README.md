# QIREN — Quantum Implicit Neural Representations

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30-green.svg)](https://pennylane.ai/)
[![TorchQuantum](https://img.shields.io/badge/TorchQuantum-0.1.7-blueviolet.svg)](https://github.com/mit-han-lab/torchquantum)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Implementation of the paper:

> **Quantum Implicit Neural Representations**
> *ICML 2024*

QIREN is a quantum machine learning framework that integrates parameterised quantum circuits (PQCs)
into implicit neural representations (INRs). It replaces or augments classical MLP layers with
hybrid quantum–classical building blocks, demonstrating improved signal representation across
image encoding, audio representation, and face generation tasks.

![](figure/2.png)

---

## Features

| Feature | Description |
|---|---|
| **Quantum backbone** | Hybrid quantum–classical layers via PennyLane (StronglyEntanglingLayers + RZ encoding + CZ entanglement), trained end-to-end with PyTorch autograd |
| **Classical baselines** | SIREN (sine activations), FFN (ReLU/tanh with residuals), and RFF (random Fourier features) — switch via `--model` |
| **Multi-task support** | Image and audio (1-D signal) representation via coordinate-based MLP |
| **GAN extension** | Quantum circuit features plugged into a StyleGAN2-ADA generator for face synthesis (`inr-gan/`) |
| **Differential operators** | Built-in gradient / Laplacian / Hessian utilities for physics-informed experiments |
| **Flexible architecture** | Hidden width, depth, quantum wire count, and noise level are all CLI-configurable |

---

## Installation

### 1. Clone and enter the repo

```bash
git clone https://github.com/Vibe-QUANTUM-LAB/QIREN.git
cd QIREN-main
```

### 2. Create an isolated environment (recommended)

```bash
conda create -n qiren python=3.10
conda activate qiren
```

### 3. Install dependencies

**For image / audio tasks (`qinr/`):**

```bash
pip install -r qinr/requirements.txt
```

**For face generation (`inr-gan/`):**

```bash
pip install -r inr-gan/requirements.txt
```

> **Note on versions:** PennyLane 0.30 is required for `qinr/`; the `inr-gan/` module uses
> TorchQuantum 0.1.7, which is tested against PyTorch 1.8–2.2.
> NumPy must stay on the 1.x series — both TorchQuantum and PennyLane have compatibility
> issues with NumPy 2.x.

---

## Quick Start

### Image representation

```bash
python qinr/train.py \
    --type image \
    --img_size 64 \
    --model hybridren \
    --epochs 301
```

### Audio representation

```bash
python qinr/train.py \
    --type sound \
    --model hybridren \
    --epochs 301
```

### Classical SIREN baseline

```bash
python qinr/train.py \
    --type image \
    --img_size 64 \
    --model siren \
    --epochs 301
```

### Face generation (INR-GAN)

```bash
cd inr-gan
python train.py \
    --outdir=./training-runs \
    --data=./data/FFHQ32x32.zip \
    --gpus=4
```

---

## Full CLI Reference — `qinr/train.py`

| Argument | Type | Default | Description |
|---|---|---|---|
| `--type` | `str` | `image` | Task type: `image` or `sound` |
| `--model` | `str` | `hybridren` | Model variant: `hybridren`, `siren`, `siren_bn`, `ffn`, `relu`, `tanh`, `relu+rff` |
| `--img_size` | `int` | `32` | Image resolution (image task only) |
| `--epochs` | `int` | `301` | Number of training epochs |
| `--lr` | `float` | `1e-4` | Initial learning rate |
| `--scheduler` | `str` | `cosine` | LR schedule: `cosine` or `linear` |
| `--hidden_dim` | `int` | `256` | Hidden layer width |
| `--n_layers` | `int` | `4` | Number of hidden layers |
| `--noise` | `float` | `0.0` | Noise level injected into quantum circuit |

---


### Module dependency diagram

```
qinr/train.py
    └── modules.py        (Hybridren / Siren / FFN)
        ├── PennyLane     (quantum circuit simulation)
        └── diff_operators.py

inr-gan/train.py
    └── training/
        ├── networks.py   (Generator / Discriminator)
        └── layers.py     (QuantumLayer / QuantumCircuitFeatures)
            └── TorchQuantum
```

---

## Quantum Architecture

### `qinr/` — PennyLane backend

The `Hybridren` model interleaves classical linear projections with quantum layers:

```
Input coords
    │
    ▼
Linear + BN
    │
    ▼
QuantumLayer (PennyLane)
  ├── RZ encoding of input features
  ├── StronglyEntanglingLayers (CZ entanglement)
  └── PauliZ measurement → output features
    │
    ▼
Linear + BN  →  … (repeated n_layers times)
    │
    ▼
Output signal value
```

### `inr-gan/` — TorchQuantum backend

`QuantumLayer` in `training/layers.py`:
- U3 gates for variational ansatz
- RZ gates for feature encoding
- CNOT rings for entanglement
- Joint ZI expectation values as output

---

## Evaluation

For face generation, evaluate FID on a trained checkpoint:

```bash
python inr-gan/calc_metrics.py \
    --metrics=fid50k_full \
    --data=./data/FFHQ32x32.zip \
    --network=./training-runs/<run>/network-snapshot-*.pkl \
    --gpus=4
```

---

## Acknowledgements

- Classical SIREN architecture:
  Sitzmann et al., *"Implicit Neural Representations with Periodic Activation Functions"*, NeurIPS 2020
  [[paper]](https://arxiv.org/abs/2006.09661) · [[code]](https://github.com/vsitzmann/siren)

- StyleGAN2-ADA baseline:
  Karras et al., *"Training Generative Adversarial Networks with Limited Data"*, NeurIPS 2020
  [[paper]](https://arxiv.org/abs/2006.06676) · [[code]](https://github.com/NVlabs/stylegan2-ada-pytorch)

- Quantum circuit simulation:
  Bergholm et al., *PennyLane* [[repo]](https://github.com/PennyLaneAI/pennylane)
  Han et al., *TorchQuantum* [[repo]](https://github.com/mit-han-lab/torchquantum)
