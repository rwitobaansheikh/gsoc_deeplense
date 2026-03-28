# GSoC DeepLense — Gravitational Lensing Classification

This repository contains two tasks submitted as part of the GSoC DeepLense application. Both tasks address **multi-class classification of simulated strong gravitational lensing images** into three categories: no substructure (`no`), spherical subhalo (`sphere`), and vortex substructure (`vort`). Images are single-channel 150×150 NumPy arrays.

---

## Tasks

### [`test1/`](test1/) — ResNet50 Transfer Learning

A **ResNet50** pretrained on ImageNet is fine-tuned for lensing classification. The first conv layer is replaced to accept single-channel input, early layers are frozen, and a custom dropout head is attached. Trained with differential learning rates, early stopping, and gradient clipping.

| Metric | Test Set |
|--------|----------|
| Accuracy | 89.27% |
| Macro AUC | 0.9783 |

→ [Full details](test1/README.md) · [Pretrained weights (Google Drive)](https://drive.google.com/file/d/1DmlbkZsKYoJcdq3XFf7OrcbB9larAazF/view?usp=drive_link)

---

### [`test7/`](test7/) — Physics-Informed Neural Network (PINN)

A **PINN** that embeds the gravitational lensing equation directly into the forward pass. A differentiable `GravitationalLensingLayer` first warps each image from the image plane to the source plane using a learnable Einstein radius (SIS model + learned residual correction), then passes the reconstructed image to a fine-tuned ResNet50 backbone. This physics prior provides an inductive bias that significantly improves classification of subtle substructure.

| Metric | Test Set | vs. Test 1 |
|--------|----------|------------|
| Accuracy | 92.48% | +3.21 pp |
| Macro AUC | 0.9881 | +0.0098 |
| Sphere recall | 83.32% | +6.28 pp |

→ [Full details](test7/README.md) · [Pretrained weights (Google Drive)](https://drive.google.com/file/d/1swvx4bzEEMCsOaR5D55bHlS_W94fgZj-/view?usp=drive_link)

---

## Repository Notes

- Datasets (`dataset/`) and model checkpoints (`model/`, `*.pth`) are excluded from the repository via `.gitignore`. Download links for the final weights are provided above.
- Each task folder contains a training notebook (`classify*.ipynb`) and a standalone evaluation notebook (`evaluate*.ipynb`).
