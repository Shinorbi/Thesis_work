# OCR Image Denoising Using Deep Learning

Undergraduate thesis, Department of Computer Science and Engineering, Shahjalal University of Science and Technology (SUST).

**Authors:** Aniruddha Halder, Loknath Banik Sagor
**Supervisor:** Ayesha Tasnim, Assistant Professor, Dept. of CSE, SUST

## Overview

Optical Character Recognition (OCR) accuracy on handwritten documents degrades sharply in the presence of image noise (scanning artifacts, paper distortion, uneven lighting, blur). This thesis addresses that problem in two phases:

1. **Preprocessing (denoising):** clean noisy handwritten document images *before* they reach the OCR engine.
2. **Postprocessing (spelling correction):** correct residual OCR misreads *after* text extraction, using sequence-to-sequence models trained on OCR error patterns.

The goal was to build a generalizable image-cleaning pipeline that improves OCR readability on inputs a human — or an OCR engine — would otherwise struggle with, and to combine it with a learned correction stage to recover clean text end-to-end.

## Dataset

- **2,152 handwritten document images**, collected across **14 different devices** to capture a realistic range of real-world noise types and capture conditions.
- Images were split into 40×40×3 patches for training/testing the denoising models.
- For the postprocessing stage, **14,000 words** were extracted from OCR output (via Google OCR) and mapped to their correct spelling to build a supervised error-correction dataset.

## Phase 1: Denoising

Five architectures were implemented and compared:

| Model | Approach |
|---|---|
| **DnCNN** | Residual learning CNN — predicts the noise residual rather than the clean image directly; 17 conv2D layers with batch normalization |
| **RIDNET** | Feature-attention-based residual network using cascading Enhanced Attention Modules (EAM) |
| **Autoencoder** | Convolutional encoder–decoder (64/128 filters), trained 80 epochs, MAE loss |
| **SRGAN** | Adapted SRGAN generator (no upscaling block, since the task is denoising, not super-resolution) with 3 residual blocks, VGG19-based perceptual loss + adversarial loss |
| **CycleGAN** | Unpaired image-to-image translation (noisy ↔ clean domains) with 9 residual blocks, adversarial + cycle-consistency loss |

### Results (PSNR, dB — higher is better)

| Noise Level | Noisy Image | DnCNN | RIDNET | Autoencoder |
|---|---|---|---|---|
| 15.0 | 27.40 | **38.91** | 26.62 | 33.20 |
| 20.0 | 24.94 | **38.35** | 27.53 | 32.78 |
| 25.0 | 23.01 | **37.06** | 28.25 | 30.47 |
| 30.0 | 21.45 | **33.98** | 28.33 | 26.34 |
| 40.0 | 18.98 | **27.32** | 26.03 | 26.00 |
| 45.0 | 17.98 | **24.83** | 24.38 | 22.18 |

**DnCNN was the best-performing model across every noise level tested**, consistently recovering the highest PSNR relative to the noisy input — a gain of 6–12 dB depending on noise severity.

## Phase 2: Postprocessing (OCR error correction)

OCR output — even after denoising — still contains misspelled or garbled words. Two sequence-to-sequence models were trained to correct these on the 14k-word dataset:

| Model | Recall | F1 | Precision | Accuracy |
|---|---|---|---|---|
| **ConvSeq2Seq** | 0.76 | **0.83** | 0.84 | **79.39%** |
| GRUSeq2Seq | 0.71 | 0.77 | 0.81 | 73.27% |

**ConvSeq2Seq outperformed the GRU-based model** on every metric, correctly recovering the intended word in ~79% of cases where the OCR engine had produced a misspelling.

## Pipeline

```
Noisy handwritten image
        │
        ▼
   DnCNN denoising  ──▶  cleaned image
        │
        ▼
     OCR (Google OCR)  ──▶  raw text (with residual errors)
        │
        ▼
  ConvSeq2Seq correction  ──▶  final corrected text
```

## Repository Contents

- `Thesis_SRGAN_pytorch.ipynb` — SRGAN-based denoising implementation (PyTorch)
- `Thesis_Autoencoder.ipynb` — Convolutional autoencoder denoising implementation (Keras)
- `Thesis_CycleGAN.ipynb` — CycleGAN unpaired denoising implementation
- Sample noisy/denoised image outputs

## Notes on Reproducing / Extending

- DnCNN and RIDNET training scripts are described in the thesis report but not yet included as standalone notebooks in this repo — being added.
- The postprocessing (ConvSeq2Seq/GRUSeq2Seq) training code and the 14k-word dataset are referenced in the thesis but not yet uploaded here — being added.
- Full methodology, related work, and references are available in the complete thesis report (PDF, on request).

## Citation

If you reference this work, please cite:

> Halder, A. & Banik Sagor, L. (2026). *OCR Image Denoising Using Deep Learning*. Undergraduate Thesis, Dept. of Computer Science and Engineering, Shahjalal University of Science and Technology. Supervised by Ayesha Tasnim.
