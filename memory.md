_Use this proactively for long term memory — LAST UPDATED: Feb 26, 2026_

---

## Project: Lightweight Speech Enhancement Using Shallow Transformers

**Team:** Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)  
**Guide:** Dr. Praveen Jaraut  

## Architecture Target (Review 2)
- Input: Log-Mel spectrogram (128 mels, n_fft=512, hop=256)
- CNN Encoder: Conv2d 1→64→128→256 (3×3, BN, ReLU)
- Shallow Transformer: 2 layers, 4 heads, d_model=256
- CNN Decoder: Conv2d 256→128→64→1 + Sigmoid mask
- Mask applied: enhanced = mask × noisy_spec
- Target params: ~350K (vs CRN 2–5M)
- Target PESQ: ≥3.2 | Latency: <15ms

## Dataset
- `earth16/libri-speech-noise-dataset` on Kaggle (6.6GB, 7000 train + 105 test WAV pairs)
- Stored as `.7z` archives, extract with `p7zip-full`
- 16kHz, single-channel, SNR 5–20dB
- 3-second segments, random crop during training

## Phase Timeline
| Phase | Date | Target |
|---|---|---|
| Review 1 | Jan 21 ✅ | CRN Baseline — PESQ 3.1 |
| Review 2 | Feb 18 ✅ | CNN-Transformer — PESQ ≥3.2 |
| Review 3 | Mar 18 | VAD (SileroVAD) integration |
| Final | Apr 8 | Quantized model + Gradio |

## Kaggle Notebooks
- Baseline CRN: `kjadeja/baseline-crn-speechenhance` (v6, last run Feb 20 2026)
- Review 2 Transformer: to be created

## Key Technical Notes
- LSTM input bug in CRN (was 256×128=32768) → fixed by per-frequency LSTM in CRN Fixed
- Original CRN: PESQ/STOI was **ESTIMATED** via formula (fake!) — CRN Fixed has real ISTFT eval
- Training uses L1 loss on log-mel spectrogram domain
- Optimizer: Adam lr=1e-3, ReduceLROnPlateau scheduler (factor=0.5, patience=5)
- Grad clipping max_norm=5.0
- Early stopping patience=10
- `kjadeja/prevres` dataset used for checkpoint resuming

## Tools & Stack
- PyTorch, torchaudio, pesq, pystoi
- Kaggle T4 GPU (15GB VRAM)
- Kaggle MCP for notebook management

---

## Review 2 Notebook: `review2-transformer-speechenhance`
**Local file:** `d:\Workspace\kaggle agent\review2-transformer-speechenhance.ipynb`  
**Kaggle notebook:** `kjadeja/review-2-cnn-transformer-speech-enhancement` (ID: 110344085, v5)  
**Status:** COMPLETE — 25 epochs, PESQ=1.141 (**FAILED**, see Review 2 Results section)**

### Cell Map (25 total cells):
| # | Type | Name | Local Status |
|---|------|------|--------------|
| 1 | MD | Title | — |
| 2 | Code | Cell 1: Imports + device check | ✅ Executed (CPU) |
| 3 | MD | Dataset header | — |
| 4 | Code | Cell 2: Data extraction (Kaggle-only cells) | ⚠️ NOT executed (Kaggle only — uses `extract_path`) |
| 5 | MD | Dataset class header | — |
| 6 | Code | Cell 3: `SpeechEnhancementDataset` class | ✅ Executed |
| 7 | MD | Model header | — |
| 8 | Code | Cell 4: `PositionalEncoding` class | ✅ Executed |
| 9 | Code | Cell 5: `ShallowTransformerEnhancer` model + forward test | ✅ Executed — 2.45M params, shapes OK |
| 10 | Code | Cell 6: `get_attention_weights()` helper | ✅ Executed — (1,4,188,188) per layer |
| 11 | MD | Training Setup header | — |
| 12 | Code | Cell 7: Training config + dataset loading | ⚠️ NOT executed (needs `extract_path`) |
| 13 | MD | Training Loop header | — |
| 14 | Code | Cell 8: Full training loop | ⚠️ NOT executed (Kaggle only) |
| 15 | MD | Training Curves header | — |
| 16 | Code | Cell 9: Plot training curves | ⚠️ NOT executed (needs `history`) |
| 17 | MD | Evaluation header | — |
| 18 | Code | Cell 10: `mel_spec_to_waveform()` + `si_sdr()` + test | ✅ Executed — Griffin-Lim OK, SI-SDR 19.15 dB |
| 19 | Code | Cell 11: Full eval — PESQ/STOI/SI-SDR | ⚠️ NOT executed (needs `test_ds`) |
| 20 | Code | Cell 12: Eval summary + plots | ⚠️ NOT executed (needs `results`) |
| 21 | MD | Attention Visualisation header | — |
| 22 | Code | Cell 13: Attention visualisation | ✅ Executed (synthetic data fallback) |
| 23 | MD | CRN vs Transformer header | — |
| 24 | Code | Cell 14: CRN vs Transformer comparison + save JSON | ⚠️ NOT executed (needs `avg_pesq`) |
| 25 | Code | SMOKE TEST: mini training on synthetic data | ✅ Executed — loss 0.2147→0.0769 in 10 steps |

### Architecture Summary (ShallowTransformerEnhancer):
```
Input: (B, 128, T) log-mel
  → unsqueeze → CNN Encoder: Conv2d(1→64→128→256, 3×3, BN, ReLU)
  → permute + mean(freq dim) → (B, T, 256)
  → Linear pre_proj(256→256)
  → PositionalEncoding(d=256, sinusoidal)
  → TransformerEncoder: 2 layers × (4 heads, ff=1024, Pre-LN, dropout=0.1)
  → Linear post_proj(256→256)
  → unsqueeze+expand → permute → (B, 256, 128, T)
  → CNN Decoder: ConvBlock(256→128), ConvBlock(128→64), Conv2d(64→1)
  → Sigmoid → mask (B, 128, T)
Output: enhanced = mask × noisy_spec
Params: 2.45M (trainable)
```

### Training Config:
- Loss: L1Loss on log-mel spectrograms
- Optimizer: Adam lr=1e-3, weight_decay=1e-5
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Batch: 16, MaxEpoch: 25, EarlyStopping: patience=10
- Grad clip: max_norm=5.0
- Checkpoint: `transformer_best.pth`
- Init: Kaiming (Conv/Linear), ones/zeros (BN)
- Saves periodic checkpoints at epochs 5,10,15,20

### Evaluation Pipeline:
- waveform reconstruction: `expm1(mel_spec)` → `InverseMelScale(driver='gelsd')` → `GriffinLim(n_iter=32)`
- Metrics: PESQ (wb), STOI, SI-SDR
- Uses REAL waveform reconstruction (not estimated like R1)
- Test on 105 samples from `earth16/libri-speech-noise-dataset`

### What Variables `extract_path` Points To On Kaggle:
```python
extract_path = '/kaggle/working/extracted_data'
```
Dataset extraction uses `!7z x ... -o{extract_path}` from `earth16/libri-speech-noise-dataset`

### Known Gotchas:
- `InverseMelScale` fails on pure random tensors (ill-conditioned matrix) — fixed by using `driver='gelsd'` and `abs()+0.1` for tests
- `get_attention_weights()` extracts attention BEFORE passing through each layer's norm+ffn — this is intentional for correct pre-attn weights
- Cell 14 (comparison) uses `avg_pesq` from Cell 12 — must run in sequence
- Cell 4 (extraction) is Kaggle-only, hardcodes paths for the `earth16/libri-speech-noise-dataset` dataset
- `norm_first=True` in TransformerEncoderLayer = Pre-LN (more training stable than standard Post-LN)
- **MEL SPECTROGRAM IS NON-INVERTIBLE** — `InverseMelScale + GriffinLim(32)` introduces catastrophic artifacts (see Review 2 Results below)

---

## Review 2 Results — COMPLETED Feb 24, 2026

**Kaggle Notebook:** `kjadeja/review-2-cnn-transformer-speech-enhancement` (ID: 110344085, v5)  
**Hardware:** P100-PCIE-16GB GPU, 25 epochs completed (no early stopping triggered)

### Metrics (on 105 test samples):
| Metric | Noisy Input | Enhanced (Ours) | CRN Baseline |
|--------|------------|-----------------|--------------|
| PESQ   | 1.144       | **1.141**        | 3.10         |
| STOI   | 0.693       | **0.695**        | —            |
| SI-SDR | -0.82 dB    | **-25.58 dB**    | —            |

**Best epoch:** 25 | **Best val loss (L1):** 0.1485  
**Val loss progress:** 0.1764 (ep1) → 0.1485 (ep25) — model WAS learning

### Root Cause of Failure:
**The mel spectrogram is non-invertible.** The evaluation chain:
```
predicted mel mask × noisy mel → expm1 → InverseMelScale(driver='gelsd') → GriffinLim(n_iter=32)
```
introduces catastrophic phase artifacts. `InverseMelScale` is an approximate pseudo-inverse of the
mel filterbank (many-to-one mapping), and GriffinLim(32 iters) cannot adequately recover phase.
Result: SI-SDR degrades from -0.82 → -25.58 dB even though val loss improved.

**The CRN baseline uses complex STFT → ISTFT (phase-preserving), which explains its PESQ=3.10.**

### Output Files Saved on Kaggle:
- `transformer_best.pth` — best checkpoint (val=0.1485, ep25)
- `ckpt_ep5/10/15/20/25.pth` — periodic checkpoints (all 5 exist)
- `attention_weights.png` — self-attention heatmaps (2 layers × 4 heads)
- `training_curves.png` — train/val L1 loss over 25 epochs
- `review2_summary.json` — full metric JSON

---

## Review 3 Plan — TARGET: Fix Reconstruction Pipeline (Mar 18)

### Core Fix: Switch from Mel Spectrogram → STFT
**Old pipeline (broken):**
```
MelSpectrogram → mask → InverseMelScale → GriffinLim  ← lossy, non-invertible
```
**New pipeline:**
```
STFT (n_fft=512, hop=128) → magnitude → mask → ISTFT with original phase ← lossless!
```

### Architecture Changes for Review 3:
1. **Input:** STFT magnitude `|X(t,f)|` shape `(B, 257, T)` instead of mel `(B, 128, T)`
2. **Model:** Update CNN encoder input dim (1×257 instead of 1×128)
3. **Output:** Mask applied to STFT magnitude → ISTFT with noisy phase
4. **Loss:** L1 on STFT magnitude domain (or MultiResolutionSTFTLoss)
5. **Reconstruction:**
   ```python
   noisy_stft = torch.stft(noisy_wav, n_fft=512, hop_length=128, return_complex=True)
   mag = noisy_stft.abs()  # (B, 257, T)
   mask = model(mag)       # (B, 257, T) in [0,1]
   enhanced_stft = mask * noisy_stft  # apply mask to complex STFT
   enhanced_wav = torch.istft(enhanced_stft, n_fft=512, hop_length=128)  # PERFECT reconstruction
   ```
6. **Optional add: SileroVAD** — detect speech frames, only enhance those
7. **Optional: Move to waveform-level Dice/loss** (SISDR loss directly)

### Secondary Fixes:
- Reduce to `num_workers=0` (eliminates `^^` AssertionError spam from DataLoader cleanup)
- GriffinLim entirely removed (not needed with STFT approach)
- PESQ/STOI computed directly on `enhanced_wav` (ground truth waveform not just spectrogram)

### Kaggle Notebook Strategy for Review 3:
- Slug: `kjadeja/review-3-stft-transformer-speech-enhancement`
- Enable internet for dataset download (same `earth16/libri-speech-noise-dataset`)
- `enableGpu = true` (P100)
- Same `kaggle datasets download` CLI approach (dataset mounting is broken for slug-based saves)
- `kernelExecutionType = "SaveAndRunAll"`

### Known Working MCP Save Template (do NOT change these):
```json
{
  "hasId": true,
  "id": <integer ID>,
  "language": "python",
  "kernelType": "notebook",
  "enableGpu": true,
  "enableInternet": true,
  "kernelExecutionType": "SaveAndRunAll",
  "text": "<notebook JSON string>"
}
```
**WARNING:** `datasetDataSources` does NOT work when updating by ID. Use `kaggle datasets download` CLI instead.

---

## Kaggle Notebooks Registry
| Notebook | Kaggle Slug | ID | Status |
|----------|-------------|-----|--------|
| CRN Baseline (R1) | `kjadeja/baseline-crn-speechenhance` | — | COMPLETE, PESQ=3.10 (ESTIMATED — fake!) |
| CRN Fixed (R1) | `kjadeja/crn-baseline-fixed-stft-speech-enhancement` | 110475041 | COMPLETE, PESQ=1.144, **CROP BUG** |
| Transformer R2 | `kjadeja/review-2-cnn-transformer-speech-enhancement` | 110344085 | COMPLETE, PESQ=1.141 (**FAILED**) |
| STFT Transformer R3 | `kjadeja/review-3-stft-transformer-speech-enhancement` | 110421493 | v3 trained 25ep, eval COMPLETE |
| R3 Eval Only | `kjadeja/review-3-eval-stft-transformer` | 110458458 | Eval ran locally instead (Kaggle input mount issues) |
| R3 DPT v1 | `kjadeja/review-3-dpt-stft-speech-enhancement` | 110473038 | COMPLETE, PESQ=1.071, **CROP BUG** |
| **CRN v2** | `kjadeja/crn-v2-aligned-speech-enhancement` | 110524198 | **v2 COMPLETE** — PESQ=1.630 (+0.467) |
| **DPT v2** | `kjadeja/dpt-v2-aligned-speech-enhancement` | 110525753 | **v2 COMPLETE** — PESQ=1.692 (+0.529) |

---

## Review 3 — STFT Transformer (In Progress)

**Local file:** `d:\Workspace\kaggle agent\review3-stft-transformer.ipynb`  
**Kaggle:** `kjadeja/review-3-stft-transformer-speech-enhancement` (ID: 110421493)  
**Hardware:** P100-PCIE-16GB, Python 3.12, PyTorch 2.6+

### Version History:
| Version | Status | Error | Fix Applied |
|---------|--------|-------|-------------|
| v1 | ERROR | `total_mem` attr missing (PyTorch 2.6) | → `getattr(props, 'total_memory', ...)` |
| v2 | ERROR | `verbose` kwarg removed from `ReduceLROnPlateau` (Python 3.12) | → removed `verbose=False` |
| v3 | ERROR | Training ✅ (25 ep), Eval ❌ `torch.load weights_only=True` blocks numpy | → `weights_only=False` |
| v4 | RUNNING | All 4 fixes applied, retrains from scratch (mistake) | SaveAndRunAll wiped v3 outputs |

### R3 Eval: Ran locally Feb 24, 2026
- Downloaded `stft_transformer_best.pth` from v3 output (epoch 15, val_loss=0.1050)
- Downloaded test.7z + y_test.7z from dataset
- Evaluated 105 test samples on CPU
- Results saved to `review3_summary.json`

### R3 Architecture (STFTTransformerEnhancer):
```
Input: (B, 1, 257, T) ← log1p(STFT magnitude)
  → CNN Encoder: Conv2d 1→64→128→256 (3×3, BN, ReLU)
  → mean(freq dim) → (B, T, 256)
  → Linear(256→256) + PositionalEncoding(d=256)
  → 2-layer Pre-LN TransformerEncoder (4 heads, ff=1024, dropout=0.1)
  → Linear(256→256) → expand → (B, 256, 257, T)
  → CNN Decoder: Conv2d 256→128→64→1 + Sigmoid
Output: mask (B, 257, T) in [0, 1]
Reconstruction: enhanced_mag × exp(j·noisy_phase) → torch.istft → waveform
Params: 2,451,457 (2.45M)
```

### STFT Config:
- n_fft=512, hop_length=256, N_FREQ=257 bins, SR=16000, MAX_LEN=48000 (3s)
- STFT roundtrip error: 7.15e-07 (validated locally)

### R3 v3 Training Results (training completed, eval errored):
```
Ep01 tr=0.1066 va=0.1072 lr=1e-3  SAVED
Ep02 tr=0.1065 va=0.1065 lr=1e-3  SAVED
Ep05 tr=0.1063 va=0.1051 lr=1e-3  SAVED
Ep11 lr→5e-4
Ep15 tr=0.1058 va=0.1050 lr=5e-4  SAVED (BEST)
Ep21 lr→2.5e-4
Ep25 tr=0.1062 va=0.1065 lr=2.5e-4  Early stop (10/10 patience)
Total: 7291s (~2h), ~293s/epoch
```

### R3 Eval Results — COMPLETED Feb 24, 2026 (Local CPU Eval):
| Metric | Noisy Input | Enhanced (R3) | R2 Enhanced | R1 CRN |
|--------|------------|---------------|-------------|--------|
| PESQ   | 1.163       | **1.089**      | 1.141       | ~3.10  |
| STOI   | 0.722       | **0.622**      | 0.695       | —      |
| SI-SDR | -0.25 dB    | **-1.65 dB**   | -25.58 dB   | —      |

**R3 is WORSE than noisy input on all metrics:**
- PESQ: -0.074 (degraded)
- STOI: -0.1009 (degraded)
- SI-SDR: -1.40 dB (degraded)

**R3 vs R2:** SI-SDR massively improved (-1.65 vs -25.58 dB) due to STFT fix, but PESQ/STOI still bad.
The STFT reconstruction is lossless — the model itself is the problem.

---

## Review 3 DPT — Dual-Path Transformer Fix (In Progress)

**Kaggle:** `kjadeja/review-3-dpt-stft-speech-enhancement` (ID: 110473038)
**Status:** v1 RUNNING on P100 (SaveAndRunAll)
**URL:** https://www.kaggle.com/code/kjadeja/review-3-dpt-stft-speech-enhancement

### Architecture: DPTSTFTEnhancer
```
Input: (B, 1, 257, T) ← log1p(STFT magnitude)
  → CNN Encoder (stride 2×2 on freq): (B, 128, 65, T)
    Conv2d(1→32, k=3, s=(2,1), p=1) + BN + ReLU
    Conv2d(32→128, k=3, s=(2,1), p=1) + BN + ReLU
  → 2 × DualPathBlock:
    Freq-Transformer: TransformerEncoderLayer(128, 4, 512, norm_first=True)
      reshapes to (B*T, 65, 128) — attend across 65 freq sub-bands per time step
    Time-Transformer: TransformerEncoderLayer(128, 4, 512, norm_first=True)
      reshapes to (B*65, T, 128) — attend across T time frames per freq bin
  → Additive skip connection from encoder output
  → F.interpolate (bilinear) back to (B, 128, 257, T)
  → CNN Decoder:
    Conv2d(128→64, k=3, p=1) + BN + ReLU
    Conv2d(64→1, k=1) + Sigmoid
Output: mask (B, 257, T) in [0, 1]
Reconstruction: mask × noisy_mag → exp(j·noisy_phase) → ISTFT → waveform
Params: 904,705 (0.90M) — 63% smaller than R3v1's 2.45M
```

### Architecture Breakdown:
- Encoder: 37,632 (4.2%)
- DPT blocks: 793,088 (87.7%)
- Decoder: 73,985 (8.2%)

### Training Config:
- batch_size=8, MAX_EPOCHS=30, patience=12
- Adam lr=1e-3, weight_decay=1e-5
- 3-epoch linear LR warmup, then ReduceLROnPlateau(factor=0.5, patience=5)
- Grad clip: max_norm=5.0
- L1 loss on log1p(magnitude) — same as R3v1 for fair comparison
- STFT: n_fft=512, hop_length=256, N_FREQ=257, SR=16000, MAX_LEN=48000

### Why DPT Fixes R3v1:
- R3v1: `mean(dim=2)` collapses 257 freq bins → zero frequency resolution → near-identity mask
- DPT: Alternating freq+time transformers give FULL spectral+temporal resolution
- Validated locally: mask range [0.14, 0.72] (not identity!) vs R3v1 which was stuck near 0.5
- Model is 63% SMALLER but has far more representational power for spectral masking

### Files:
- `build_r3_dpt_nb.py` — notebook generator script
- `kaggle_r3_dpt_nb.json` — Kaggle push payload
- `dpt_nb_text.txt` — extracted notebook text (26,655 chars)
- `validate_cells.py` — syntax validation script

### DPT Results — COMPLETED Feb 24, 2026
**Training:** 30 epochs (full run, no early stop), best_ep=30, best_val=0.1006, time=12916s (~3.6h)
**Val loss:** 0.1031 (ep1) → 0.1006 (ep30) — only 2.4% improvement, LR dropped to 5e-4 at ep29

| Metric | Noisy | DPT Enhanced | Delta |
|--------|-------|-------------|-------|
| PESQ | 1.109 | **1.071** | **-0.039** |
| STOI | 0.218 | **0.339** | **+0.121** |
| SI-SDR | -43.34 dB | **-40.49 dB** | **+2.84 dB** |

**DPT FAILED on PESQ — still degrades speech quality.** STOI and SI-SDR improved but PESQ got worse.
Despite fixing the frequency bottleneck, the DPT cannot learn meaningful spectral masking with this loss function and training setup.

---

---

## CRN Baseline Fixed — STFT-Based (COMPLETE)

**Kaggle:** `kjadeja/crn-baseline-fixed-stft-speech-enhancement` (ID: 110475041)
**Status:** v1 COMPLETE
**URL:** https://www.kaggle.com/code/kjadeja/crn-baseline-fixed-stft-speech-enhancement

### Why This Was Needed:
The original CRN baseline (`kjadeja/baseline-crn-speechenhance`) had 5 critical issues:
1. **Fake eval metrics** — PESQ estimated via `pesq_noisy + (improvement_ratio - 1.0) * 0.5`, not real waveforms
2. **Mel spectrogram pipeline** — non-invertible, no ISTFT possible
3. **`mean(dim=2)` frequency collapse** — same bottleneck as R3v1
4. **Training code cells commented out** — messy duplicate cells
5. **Dead/commented code** throughout

The "PESQ ~3.10" from the original CRN was **NEVER REAL** — it was a formula estimate.

### Architecture: CRNBaseline (Fixed)
```
Input: (B, 1, 257, T) ← log1p(STFT magnitude)
  → CNN Encoder (stride (2,1) on freq):
    Conv2d(1→64, k=3, s=(2,1), p=1) + BN + ReLU     → (B, 64, 129, T)
    Conv2d(64→128, k=3, s=(2,1), p=1) + BN + ReLU   → (B, 128, 65, T)
    Conv2d(128→256, k=3, s=(2,1), p=1) + BN + ReLU  → (B, 256, 33, T)
  → Per-frequency LSTM: reshape (B,256,33,T) → (B*33, T, 256) → LSTM(256, 256, 2 layers, dropout=0.1)
  → CNN Decoder (transposed conv):
    ConvT2d(256→128, k=3, s=(2,1)) + BN + ReLU      → (B, 128, 66, T)
    ConvT2d(128→64, k=3, s=(2,1)) + BN + ReLU       → (B, 64, 132, T)
    ConvT2d(64→32, k=3, s=(2,1)) + BN + ReLU        → (B, 32, 264, T)
  → F.interpolate bilinear to (B, 32, 257, T) [fallback for freq mismatch]
  → Conv2d(32→1, k=1) + Sigmoid → mask (B, 257, T)
Output: mask × noisy_mag → exp(j·noisy_phase) → ISTFT → waveform
Params: 1,811,009 (1.81M)
```

### Architecture Breakdown:
- Encoder:  370,560 (20.5%)
- LSTM:   1,052,672 (58.1%)
- Decoder:  387,777 (21.4%)

### Training Config:
- batch=16, 25 epochs, patience=10
- Adam lr=1e-3, weight_decay=1e-5
- ReduceLROnPlateau(factor=0.5, patience=5)
- Grad clip: max_norm=5.0
- L1 loss on log1p(magnitude)
- STFT: n_fft=512, hop_length=256, N_FREQ=257

### Key Differences from Original CRN:
- STFT (lossless ISTFT) instead of Mel (non-invertible)
- Per-frequency LSTM: `(B*33, T, 256)` — each freq sub-band gets temporal modeling
- Proper encoder-decoder with stride-2 convolutions + transposed convolutions
- Real PESQ/STOI/SI-SDR via ISTFT waveform reconstruction
- Saves checkpoint as `crn_baseline_best.pth`
- Outputs `crn_baseline_summary.json` with all metrics

### Files:
- `build_crn_fixed.py` — notebook generator script
- `kaggle_crn_fixed_nb.json` — Kaggle push payload
- `crn_nb_text.txt` — extracted notebook text (26,083 chars)
- `validate_crn.py` — syntax validation script
- `test_crn_model.py` — local forward-pass test

### CRN Fixed Results — COMPLETED Feb 24, 2026
**Training:** 25 epochs (no early stop), best_ep=22, best_val=0.1009, time=6765s (~1.9h)
**Val loss:** 0.1065 (ep1) → 0.1009 (ep22) — only 5.3% improvement, LR dropped to 5e-4 at ep13

| Metric | Noisy | CRN Fixed | Delta |
|--------|-------|-----------|-------|
| PESQ | 1.126 | **1.144** | **+0.017** |
| STOI | 0.215 | **0.336** | **+0.121** |
| SI-SDR | -44.04 dB | **-41.03 dB** | **+3.01 dB** |

**CRN Fixed is the ONLY model that improves PESQ over noisy baseline.** But the improvement is tiny (+0.017).
Note: Noisy STOI=0.215 and SI-SDR=-44dB are suspiciously different from R3v1 local eval (STOI=0.722, SI-SDR=-0.25dB).
This suggests the test data is being loaded differently (possibly different random crops or padding).

---

---

## v2 NOTEBOOKS — ALIGNED CROP FIX (Pushed Feb 25, 2026)

### ROOT CAUSE DISCOVERY — Crop Misalignment Bug
**THE BUG:** In v1 notebooks (CRN Fixed, DPT, R3), `_load_fix()` was called INDEPENDENTLY for noisy and clean files. Each call generated its OWN random crop position via `torch.randint`. Since WAV files are 16-24 seconds long (NOT 1 second as dataset description misleadingly stated) and crops are 3 seconds, the noisy and clean waveforms came from COMPLETELY DIFFERENT time positions.

**Evidence of the bug in v1 results:**
- Noisy STOI = 0.215 (should be ~0.7-0.8 for aligned pairs)
- Noisy SI-SDR = -44 dB (should be ~-0.25 to 5 dB for aligned pairs)
- Loss plateau at ~0.100 for ALL models (can't learn from mismatched pairs)

**THE FIX:** `__getitem__` now loads BOTH files, computes ONE random start position, and crops BOTH at the same position. Test mode uses `start=0` for deterministic evaluation.

### Actual File Lengths (verified):
- Min duration: 16.600s, Max: 24.525s, Mean: 16.912s
- Clean files same: Min 16.600s, Max 24.525s, Mean 16.913s
- All at 16kHz sample rate

### CRN v2 — Aligned Crop Fix — COMPLETE
**Kaggle:** `kjadeja/crn-v2-aligned-speech-enhancement` (ID: 110524198, v2)
**Status:** COMPLETE — 30 epochs, best_ep=30, best_val=0.0534, time=8041s (~2.2h)
**URL:** https://www.kaggle.com/code/kjadeja/crn-v2-aligned-speech-enhancement
- Architecture: CRNBaseline (1,811,009 params) — CNN Encoder + per-freq LSTM + CNN Decoder
- Aligned crop fix + correlation sanity check (corr=0.8578, PASSED)
- batch=16, 30 epochs, patience=10, LR=1e-3 (dropped to 5e-4 at ep29)
- L1 loss on log1p(magnitude)
- v1 failed with `torchaudio.info` error → v2.1 fix applied

| Metric | Noisy | CRN v2 Enhanced | Delta |
|--------|-------|----------------|-------|
| PESQ | 1.163 | **1.630** | **+0.467** |
| STOI | 0.722 | **0.864** | **+0.141** |
| SI-SDR | -0.25 dB | **8.62 dB** | **+8.86 dB** |

### DPT v2 — Aligned Crop Fix — COMPLETE
**Kaggle:** `kjadeja/dpt-v2-aligned-speech-enhancement` (ID: 110525753, v2)
**Status:** COMPLETE — 30 epochs, best_ep=29, best_val=0.0513, time=12917s (~3.6h)
**URL:** https://www.kaggle.com/code/kjadeja/dpt-v2-aligned-speech-enhancement
- Architecture: DPTSTFTEnhancer (904,705 params) — CNN Encoder + Dual-Path Transformer + CNN Decoder
- Aligned crop fix + correlation sanity check (corr=0.8578, PASSED)
- batch=8, 30 epochs, patience=12, warmup=3 epochs, LR=1e-3 (stayed at 1e-3 entire run)
- v1 failed with `torchaudio.info` error → v2.1 fix applied

| Metric | Noisy | DPT v2 Enhanced | Delta |
|--------|-------|----------------|-------|
| PESQ | 1.163 | **1.692** | **+0.529** |
| STOI | 0.722 | **0.866** | **+0.144** |
| SI-SDR | -0.25 dB | **9.05 dB** | **+9.30 dB** |

### Key Results Summary:
- **DPT v2 beats CRN v2** on ALL metrics with HALF the parameters (0.9M vs 1.8M)
- Val loss halved from v1's 0.10 plateau → 0.05 — confirms crop misalignment was THE root cause
- Noisy baseline now CORRECT: STOI=0.722, SI-SDR=-0.25dB (vs v1 broken: STOI=0.215, SI-SDR=-44dB)
- Alignment check: noisy-clean correlation = 0.8578 (both notebooks)
- PESQ ~1.6-1.7 is a realistic result for this dataset/model size — NOT the fake 3.10 from R1

### Build Files:
- `build_crn_v2.py` — 683 lines, notebook generator with aligned crop fix (v2.1: torchaudio fix at line 163)
- `build_dpt_v2.py` — 693 lines, notebook generator with aligned crop fix (v2.1: torchaudio fix at line 161)
- `crn_v2_nb_text.txt` — 28,324 chars, 19 cells (regenerated Feb 26)
- `dpt_v2_nb_text.txt` — 28,465 chars, 19 cells (regenerated Feb 26)
- `smoke_test.py` — local validation harness (tests imports, dataset class, model class, forward pass, loss, SI-SDR)

### v2.1 Fix — torchaudio.info Removed (Feb 26, 2026)
**Error:** `AttributeError: module 'torchaudio' has no attribute 'info'`
**Root cause:** Kaggle's newer torchaudio removed the `info()` function entirely.
**Fix:** Replaced `torchaudio.info(fp)` with `torchaudio.load(fp)` + `wav.shape[-1]` for frame count.
```python
# OLD (broken):
info = torchaudio.info(fp)
dur = info.num_frames / info.sample_rate

# NEW (working):
wav, sr = torchaudio.load(fp)
num_frames = wav.shape[-1]
dur = num_frames / sr
```
**Validation:** Local smoke test passes all 6 checks for both CRN and DPT.

### PyTorch/torchaudio Kaggle Compat Issues (Complete List):
1. `torch.cuda.get_device_properties(0).total_mem` → use `getattr` fallback for `total_memory`/`total_mem`
2. `ReduceLROnPlateau(verbose=False)` → `verbose` param removed in newer PyTorch — don't use it
3. `torch.load()` defaults to `weights_only=True` → explicitly use `weights_only=False`
4. **`torchaudio.info()` removed** → use `torchaudio.load()` + `wav.shape[-1]` for frame count

---

### v1 vs v2 Loss Comparison (proves crop fix was the root cause):
- **v1 (misaligned):** val loss plateau at ~0.100 for ALL models → near-identity masks → PESQ degraded
- **v2 (aligned):** val loss dropped to ~0.05 → real spectral masks → PESQ improved by +0.5
- Val loss improvement: ~50% reduction (0.10 → 0.05) proves data quality was the bottleneck

### OBSOLETE — Critical Observation (v1 only, fixed in v2):
_These observations from R3 v1 are now explained by the crop misalignment bug:_
- Val loss barely moved (0.1072 → 0.1050) because model was learning from mismatched pairs
- Near-identity masks were the model's rational response to unrelated noisy/clean inputs
- All architectures plateauing at same loss = data bug, not architecture problem

### Architecture Bottleneck Identified:
- `mean(dim=2)` collapses ALL 257 frequency bins into a single vector per time step
- Transformer operates only on temporal dimension — zero frequency resolution
- `expand(-1, -1, n_freq, -1)` copies same feature to all freq bins — decoder must infer freq info from spatial position alone
- This limits the model to time-dependent global gain, not spectral masking

### PyTorch 2.6 / Python 3.12 Compatibility Issues Found:
1. `torch.cuda.get_device_properties(0).total_mem` → use `total_memory` (or `getattr` fallback)
2. `ReduceLROnPlateau(verbose=False)` → `verbose` parameter REMOVED in Python 3.12
3. `torch.load()` defaults to `weights_only=True` → blocks numpy globals in checkpoints
4. `val_loss` as `np.mean()` result = numpy float → fails `weights_only=True`. Fix: save as `float(va_loss)`

### Output Files Saved on Kaggle (v3):
- `stft_transformer_best.pth` — best checkpoint (ep15, val=0.1050)
- `ckpt_ep5.pth`, `ckpt_ep10.pth`, `ckpt_ep15.pth`, `ckpt_ep20.pth`, `ckpt_ep25.pth`
- `training_curves.png` — train/val L1 loss over 25 epochs
