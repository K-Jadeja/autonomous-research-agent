# PRESENTATION: Lightweight Speech Enhancement Using Shallow Transformers

**Team:** Krishnasinh Jadeja (22BLC1211) · Kirtan Sondagar (22BLC1228) · Prabhu Kalyan Panda (22BLC1213)
**Guide:** Dr. Praveen Jaraut — VIT Chennai
**Date:** April 2026

---

## SLIDE 1–2: PROJECT OVERVIEW

### Slide 1: Title Slide
- **Title:** Lightweight Speech Enhancement Using Shallow Transformers
- **Subtitle:** Dual-Path Transformer vs Conv-Recurrent Network — A Comparative Study
- **Team Members:** Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)
- **Guide:** Dr. Praveen Jaraut
- **Institution:** School of Electronics Engineering, VIT Chennai

### Slide 2: Project At A Glance
- **What we built:** Two deep learning models that clean noisy audio — removing background noise while preserving the speaker's voice
- **Baseline:** CRN v2 (Conv-Recurrent Network) — 1.81M parameters
- **Proposed:** DPT v2 (Dual-Path Transformer) — 0.90M parameters (50% smaller)
- **Both models** operate on STFT spectrograms and predict a soft mask to separate speech from noise
- **The headline result:** DPT beats CRN on every quality metric while being half the size and using 33% less compute
- **Deployment:** Interactive Gradio web demo with spectrogram visualization and real-time audio playback
- **Timeline:** 4 reviews spanning Jan 2026 → Apr 2026

---

## SLIDE 3: PROBLEM STATEMENT & OBJECTIVES

### The Problem
- Speech signals in real-world environments are corrupted by additive noise (traffic, HVAC, crowd noise)
- This degrades intelligibility for hearing aids, voice assistants, and teleconference systems
- Traditional DSP filters (Wiener, spectral subtraction) require noise-type assumptions and fail in non-stationary conditions
- Existing deep learning models like CRNs use parameter-heavy LSTMs — too expensive for edge deployment (phones, hearing aids, IoT)

### Objectives
1. Design a **lightweight deep-learning model** (< 1M parameters) for single-channel speech enhancement
2. Operate in the **STFT magnitude domain** with lossless phase-preserving reconstruction
3. Achieve measurable improvement in **PESQ, STOI, and SI-SDR** over the noisy baseline
4. Run in **real-time on edge devices** (target latency < 15ms per frame)
5. Outperform the CRN baseline in both quality and computational efficiency

---

## SLIDE 4–5: SYSTEM ARCHITECTURE

### Slide 4: Overall System Pipeline
```
🎤 Noisy Waveform x(t)
    │
    ├──→ STFT (n_fft=512, hop=256, Hann window)
    │       │
    │       ├──→ |X(t,f)| Magnitude (257 × T)  ──→ Deep Learning Model ──→ Mask M(t,f) ∈ [0,1]
    │       │                                                                           │
    │       └──→ ∠X(t,f) Phase (257 × T)                              M ⊙ |X| ──────────┘
    │                                                                           │
    └──────────────────────────────────────────────────────────────────────────────→ ISTFT
    │                                                                              (with original phase)
    │
🔊 Enhanced Waveform ŝ(t)
```

**Key Design Choice:** STFT magnitude + original noisy phase → ISTFT is **mathematically lossless**. This avoids the catastrophic artifacts from Mel-spectrogram + GriffinLim that destroyed audio quality in earlier reviews.

### Slide 5: Two Model Architectures Side by Side

**Model A — CRN v2 (Baseline, 1.81M params):**
```
Input (1×257×T)
  → Conv2d 1→64 [stride-2 freq]    (129×T)
  → Conv2d 64→128 [stride-2 freq]  (65×T)
  → Conv2d 128→256 [stride-2 freq] (33×T)
  → 2-layer LSTM (256→256)         per frequency bin  ← SEQUENTIAL BOTTLENECK
  → ConvTranspose2d 256→128        (65×T)
  → ConvTranspose2d 128→64         (129×T)
  → ConvTranspose2d 64→32          (257×T)
  → Conv2d 32→1 + Sigmoid          mask output
```
- LSTM consumes 58% of all parameters
- Sequential processing limits parallelism

**Model B — DPT v2 (Proposed, 0.90M params):**
```
Input (1×257×T)
  → Conv2d 1→32 [stride-2 freq]    (129×T)
  → Conv2d 32→128 [stride-2 freq]  (65×T)
  → DualPathBlock ×2:
      → Freq Transformer (attend across 65 freq bins)      ← PARALLEL
      → Time Transformer (attend across T time frames)     ← PARALLEL
  → +Skip connection from encoder
  → Bilinear upsample                  (257×T)
  → Conv2d 128→64 → Conv2d 64→1 + Sigmoid  mask output
```
- DualPathBlocks consume 88% of params — all parallelizable attention
- Skip connection preserves low-level spectral features
- 4 attention heads, Pre-LayerNorm, FF dim 512

---

## SLIDE 6–10: COMPLETE IMPLEMENTATION

### Slide 6: Dataset & Data Pipeline
| Property | Value |
|---|---|
| Source | `earth16/libri-speech-noise-dataset` (Kaggle) |
| Train Pairs | 7,000 noisy + clean WAV pairs |
| Test Pairs | 105 noisy + clean WAV pairs |
| Sample Rate | 16 kHz, mono |
| Duration | 16–24 seconds per utterance |
| SNR Range | 5–20 dB additive noise |
| Training Crop | 3-second random segment (aligned) |
| Train/Val Split | 6,300 train / 700 validation |

**Critical Engineering Fix — Data Alignment Bug:**
- **The Bug:** v1 code called `_load_fix()` independently for noisy and clean files. Each call generated its own random crop position. With 16–24s files and 3s crops, only ~19% of pairs overlapped. The model trained on mismatched audio.
- **The Fix:** A single shared crop position for both noisy and clean in `__getitem__()`. Validated with correlation check (corr = 0.8578).
- **Impact:** Val loss dropped from ~0.10 → ~0.05. PESQ improvement went from ~0 (bug) to +0.529 (real).

### Slide 7: DualPathBlock — Core Innovation
```
class DualPathBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=512, dropout=0.1):
        self.freq_transformer = TransformerEncoderLayer(norm_first=True)
        self.time_transformer = TransformerEncoderLayer(norm_first=True)

    def forward(self, x):   # x: (B, 128, 65, T)
        # Frequency path: attend across 65 freq bins per time step
        x_f = reshape to (B×T, 65, 128) → freq_transformer → reshape back
        # Time path: attend across T time frames per freq bin
        x_t = reshape to (B×65, T, 128) → time_transformer → reshape back
        return x
```

**Why Dual-Path works:**
- **Freq-Transformer:** Models harmonic structures and spectral envelopes across frequency bins
- **Time-Transformer:** Captures temporal dynamics, onset/offset patterns, and speech rhythm
- Both use Pre-LayerNorm (stabilizes training), 4 attention heads, FF dimension 512
- Fully parallelizable — no sequential bottleneck like LSTM

### Slide 8: Training Configuration
| Hyperparameter | CRN v2 | DPT v2 |
|---|---|---|
| Batch Size | 16 | 8 |
| Max Epochs | 30 | 30 |
| Optimizer | Adam (lr=1e-3, wd=1e-5) | Adam (lr=1e-3, wd=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | ReduceLROnPlateau (factor=0.5, patience=5) |
| LR Warmup | — | 3 epochs (linear) |
| Early Stopping | patience=10 | patience=12 |
| Gradient Clipping | max_norm=5.0 | max_norm=5.0 |
| Loss Function | L1 on log1p(magnitude) | L1 on log1p(magnitude) |
| Hardware | Kaggle P100 (16 GB) | Kaggle P100 (16 GB) |

**Loss Function:**
L = (1/N) Σ || log(1 + |Ŝ(t,f)|) - log(1 + |S(t,f)|) ||
where Ŝ = M · |X| (masked noisy magnitude). The log1p compression gives equal weight to quiet and loud frequency bins.

### Slide 9: Performance Optimization & Waveform Reconstruction
**Optimizations applied:**
- Kaiming Normal weight initialization for stable training start
- Gradient clipping (max_norm=5.0) prevents exploding gradients
- Linear LR warmup (DPT only) prevents early instability with Transformers
- Batch Normalization + ReLU in CNN layers for fast convergence
- Pre-LayerNorm in Transformer blocks (more stable than Post-LayerNorm)

**Waveform Reconstruction (Evaluation):**
```
1. noisy_stft = STFT(noisy_wav, n_fft=512, hop=256)       # Complex STFT
2. noisy_mag  = |noisy_stft|                                # Magnitude (257 × T)
3. noisy_phase = ∠noisy_stft                                # Phase (preserved!)
4. mask = model(log1p(noisy_mag))                           # Forward pass
5. enhanced_mag = mask × noisy_mag                           # Apply mask
6. enhanced_stft = enhanced_mag × exp(j × noisy_phase)       # Combine with original phase
7. enhanced_wav = ISTFT(enhanced_stft)                       # LOSSLESS reconstruction
```

### Slide 10: Gradio Web Deployment
**Three deployment versions built:**

| Version | Port | Features |
|---|---|---|
| Simple Demo | 7860 | 1 tab, quick enhancement only |
| Enhanced | 7861 | 3 tabs, batch stats, model info |
| Premium | 7862 | Custom audio upload, full features, about tab |

**UI Features:**
- Pre-loaded test sample selector (105 samples)
- One-click inference with CRN v2 and DPT v2 simultaneously
- Spectrogram comparison: Noisy → CRN → DPT → Clean
- Mask visualization showing speech/noise separation
- Real-time audio playback of enhanced outputs
- Metrics display: PESQ, STOI, SI-SDR
- Latency benchmarking per model
- Professional dark theme, optimized rendering (~1.2s per visualization)

---

## SLIDE 11–15: RESULTS & TESTING

### Slide 11: Final Results — Head to Head
| Metric | Noisy Input | CRN v2 (1.81M) | DPT v2 (0.90M) | Improvement |
|---|---|---|---|---|
| **PESQ** (1–4.5) | 1.163 | 1.630 (+0.467) | **1.692 (+0.529)** | +45.5% over noisy |
| **STOI** (0–1) | 0.722 | 0.864 (+0.141) | **0.866 (+0.144)** | +19.9% over noisy |
| **SI-SDR** (dB) | -0.25 | 8.62 (+8.86) | **9.05 (+9.30)** | Noise reduced to ~1/8th |
| **Parameters** | — | 1,811,009 | **904,705** | 2× smaller |
| **GFLOPs** | — | 15.74 | **10.50** | 33% less compute |
| **Checkpoint** | — | 7.07 MB | **3.56 MB** | 50% smaller |
| **< 1M params?** | — | No | **Yes** | — |
| **Latency < 15ms?** | — | Pass | **Pass** | — |

**Key Finding:** DPT achieves superior performance on ALL metrics while using only 49.9% of CRN's parameters.

### Slide 12: Computational Cost Analysis
**GFLOPs per forward pass (3-second audio, input shape 1×1×257×188):**

| Metric | CRN v2 | DPT v2 | Advantage |
|---|---|---|---|
| Parameters | 1.811M | 0.905M | DPT is 50% smaller |
| GFLOPs | 15.74 | 10.50 | DPT uses 33% less compute |
| Checkpoint size | 7.07 MB | 3.56 MB | DPT is 50% lighter |

**Why this matters for deployment:**
- Edge devices (phones, hearing aids): fewer FLOPs = longer battery life
- Cloud servers: fewer FLOPs = lower inference cost per request
- Both models scale linearly with audio duration; DPT consistently uses less compute
- DPT's gap widens for longer clips because Transformer processes in parallel while LSTM is sequential

**Computational Cost Visualizations:**

![FLOPs Comparison](docs/chart1_flops_comparison.png)
*Parameter count (DPT is 50% smaller), GFLOPs per pass (DPT uses 33% less compute), compute efficiency per parameter, and the CRN-to-DPT ratio across all metrics.*

![FLOPs Scaling](docs/chart2_flops_scaling.png)
*Both models scale roughly linearly with audio duration. DPT consistently uses less compute at every duration.*

![Quality vs Efficiency](docs/chart3_quality_efficiency.png)
*All metrics normalized — higher = better. DPT (green) wins on both quality AND efficiency.*

![Efficiency Summary](inference_efficiency.png)
*Parameter count, inference latency, and peak RAM usage. DPT meets the <1M parameter goal.*

### Slide 13: Training Curves & Convergence
**CRN v2:**
| Epoch | Train Loss | Val Loss | LR |
|---|---|---|---|
| 1 | 0.0731 | 0.0682 | 1e-3 |
| 5 | 0.0594 | 0.0581 | 1e-3 |
| 10 | 0.0567 | 0.0559 | 1e-3 |
| 20 | 0.0553 | 0.0547 | 1e-3 |
| 30 (best) | 0.0539 | 0.0534 | 5e-4 |

**DPT v2:**
| Epoch | Train Loss | Val Loss | LR |
|---|---|---|---|
| 1 | 0.0609 | 0.0593 | warmup |
| 4 | 0.0564 | 0.0549 | 1e-3 |
| 10 | 0.0548 | 0.0534 | 1e-3 |
| 20 | 0.0533 | 0.0518 | 1e-3 |
| **29 (best)** | **0.0532** | **0.0513** | **1e-3** |

- DPT achieves lower best val loss (0.0513 vs 0.0534) with half the parameters
- DPT's LR never dropped (still actively learning at epoch 30)
- Warmup produced smoother initial descent vs CRN

![CRN Training Curves](crn_output/training_curves.png)
*CRN v2 training and validation loss curves — steady convergence, LR reduced at epoch 29.*

![DPT Training Curves](dpt_output/training_curves.png)
*DPT v2 training and validation loss curves — smoother warmup, lower best val loss at epoch 29.*

### Slide 14: Visual Spectrogram Comparison

![CRN Spectrogram Comparison](crn_output/spectrogram_comparison.png)
*CRN v2: Noisy input → CRN Enhanced → Clean Reference. Noticeable noise suppression with LSTM-based temporal modeling.*

![DPT Spectrogram Comparison](dpt_output/spectrogram_comparison.png)
*DPT v2: Noisy input → DPT Enhanced → Clean Reference. Sharper harmonic definition due to frequency-aware self-attention.*

**Key observations:**
- **Noisy Input:** Dense broadband noise fills all frequency bins; no clean harmonic structure visible
- **CRN Enhanced:** Noticeably cleaner — vertical striations (voiced speech frames) emerge, inter-harmonic noise attenuated
- **DPT Enhanced:** Sharper harmonic definition than CRN — inter-harmonic noise bands more consistently suppressed
- **Clean Reference:** Ground truth — strong narrow harmonic bands, clearly dark noise floor in pauses

![Predicted Masks](inference_masks.png)
*CRN mask (left) vs DPT mask (right). DPT produces sharper speech/noise boundaries → cleaner output with fewer artifacts.*

![Overall Comparison](inference_comparison.png)
*Left to right: Noisy input → CRN enhanced → DPT enhanced → Clean target.*

**Per-Sample PESQ (first 3 test samples):**
| Sample | Noisy PESQ | DPT Enhanced | Delta |
|---|---|---|---|
| Sample 1 | 1.114 | 1.751 | +0.637 |
| Sample 2 | 1.097 | 1.795 | +0.698 |
| Sample 3 | 1.106 | 1.632 | +0.526 |

### Slide 15: Comparison Across All Reviews & Standards Compliance
**Full project evolution (7 model versions):**
| Model | Review | PESQ | STOI | SI-SDR | Params | Status |
|---|---|---|---|---|---|---|
| CRN (Original) | R1 | ~3.10* | — | — | ~2M | Metrics estimated (fake) |
| CNN-Transformer (Mel) | R2 | 1.141 | 0.695 | -25.58 dB | 2.45M | Mel non-invertible |
| STFT-Transformer | R3 | 1.089 | 0.622 | -1.65 dB | 2.45M | Frequency collapse |
| CRN v1 | R3 | 1.144 | 0.336 | -41.03 dB | 1.81M | Crop misalignment bug |
| DPT v1 | R3 | 1.071 | 0.339 | -40.49 dB | 0.90M | Crop misalignment bug |
| **CRN v2** | **R3** | **1.630** | **0.864** | **8.62 dB** | **1.81M** | **Working** |
| **DPT v2** | **R3** | **1.692** | **0.866** | **9.05 dB** | **0.90M** | **Best** |

**Standards & Codes Complied:**
- **ITU-T P.862:** Perceptual Evaluation of Speech Quality (PESQ) — used for audio quality assessment
- **ITU-T P.10/G.100:** Vocabulary for performance and quality of service
- **ISO/IEC 25010:** Software Quality Models (efficiency and performance)
- **PEP 8:** Python code style standard for maintainable codebase
- **IEEE:** Evaluation methodology follows IEEE ICASSP/Interspeech publication standards for reproducible speech enhancement benchmarks

**Kaggle Notebook Registry:**
| Notebook | Kaggle ID | Version | PESQ |
|---|---|---|---|
| CRN Baseline (R1) | `kjadeja/baseline-crn-speechenhance` | v6 | ~3.10 (estimated) |
| CNN-Transformer (R2) | `kjadeja/review-2-cnn-transformer-speech-enhancement` | v5 | 1.141 |
| STFT Transformer (R3) | `kjadeja/review-3-stft-transformer-speech-enhancement` | v3 | 1.089 |
| **CRN v2** | `kjadeja/crn-v2-aligned-speech-enhancement` | **v2** | **1.630** |
| **DPT v2** | `kjadeja/dpt-v2-aligned-speech-enhancement` | **v2** | **1.692** |

---

## SLIDE 16–17: CONCLUSION & FUTURE SCOPE

### Slide 16: Conclusion
1. **DPT is the superior model** — scores higher on PESQ (+0.062), STOI (+0.002), and SI-SDR (+0.43 dB) compared to CRN, while using 50% fewer parameters
2. **Real-time capable** — inference latency well under 15ms target for both models
3. **Meets all project constraints** — DPT satisfies <1M parameter requirement (0.905M); CRN exceeds it by 81%
4. **Transformer advantage is real** — self-attention captures long-range dependencies in both frequency and time that LSTM misses, while being more parallelizable and parameter-efficient
5. **Phase-preserving STFT pipeline** resolved the catastrophic audio artifacts from earlier Mel-spectrogram approaches (SI-SDR went from -25.58 dB → 9.05 dB)
6. **Data alignment fix** was the critical engineering discovery — correcting a hidden bug that caused all v1 models to train on mismatched audio pairs
7. **Deployed and demonstrable** via interactive Gradio web interface with spectrogram visualization and real-time audio playback

### Slide 17: Future Scope
- **INT8 Quantization:** Apply `torch.quantization` for further 2–4× model compression, enabling deployment on microcontrollers and hearing aids
- **VAD Integration:** SileroVAD to detect speech frames and only enhance voiced segments — reduces unnecessary computation by ~40% during silence
- **Extended Evaluation:** Test on DEMAND and VCTK-DEMAND datasets for generalization across diverse noise conditions
- **On-Device Deployment:** Convert to ONNX/TFLite for mobile and embedded deployment (Android, iOS, Raspberry Pi)
- **Real-Time Streaming:** Adapt the model for frame-by-frame processing with look-ahead buffers for live teleconferencing
- **Multi-Noise Generalization:** Train on mixed noise scenarios (babble + traffic + wind simultaneously)
- **Hugging Face Spaces Deployment:** Public web demo accessible without local setup

---

## SLIDE 18: REFERENCES

1. Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation," *IEEE/ACM Trans. Audio, Speech, and Language Processing*, vol. 27, no. 8, pp. 1256–1266, 2019.

2. J. Chen, Q. Mao, and D. Liu, "Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation," in *Interspeech*, 2020.

3. X. Ren et al., "A Causal U-Net Based Neural Beamforming Network for Real-Time Multi-Channel Speech Enhancement," in *Interspeech*, 2021.

4. C. K. A. Reddy et al., "INTERSPEECH 2021 Deep Noise Suppression Challenge," in *Interspeech*, 2021.

5. A. W. Rix et al., "Perceptual Evaluation of Speech Quality (PESQ) — A New Method for Speech Quality Assessment of Telephone Networks and Codecs," in *ICASSP*, 2001.

6. C. H. Taal et al., "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech," *IEEE Trans. Audio, Speech, and Language Processing*, vol. 19, no. 7, pp. 2125–2136, 2011.

7. J. Le Roux et al., "SDR — Half-Baked or Well Done?," in *ICASSP*, 2019.

8. V. Panayotov et al., "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," in *ICASSP*, 2015.

9. K. Tan and D. Wang, "A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement," in *Interspeech*, 2018.

10. A. Vaswani et al., "Attention Is All You Need," in *NeurIPS*, 2017.

11. D. Yin, C. Luo, Z. Xiong, and W. Zeng, "PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network," in *AAAI*, 2020.
