_Use this proactively for short term memory and to track the processes — LAST UPDATED: Feb 25, 2026_

---

## Session Log: Feb 25, 2026 (Continued) — R3 DPT Notebook Pushed

### What Was Done:
1. ✅ Diagnosed R3v1 failure: `mean(dim=2)` collapses frequency → near-identity mask
2. ✅ Designed Dual-Path Transformer (DPT) architecture: 904K params (0.90M)
3. ✅ Created `build_r3_dpt_nb.py` — generates 19-cell notebook (11 code + 8 markdown)
4. ✅ Built notebook: 26,655 chars, validated all 11 code cells via ast.parse
5. ✅ Validated model locally: shapes OK, mask range [0.14, 0.72] (not identity)
6. ✅ Pushed to Kaggle: `kjadeja/review-3-dpt-stft-speech-enhancement` (ID: 110473038, v1)
7. ⏳ Training RUNNING on P100 (SaveAndRunAll)

### Architecture Decision:
Chose Option A: Fix the architecture (Dual-Path Transformer) over:
- Option B: Pure CNN (U-Net/CRN) — would miss capstone's transformer requirement
- Option C: Abandon transformer approach — not acceptable for capstone
- Option D: Hybrid LSTM + transformer — too complex

### DPT Key Design Choices:
- d_model=128, nhead=4, dim_ff=512, num_dp_blocks=2
- CNN encoder stride (2,1) × 2 on freq → F'=65 bins (manageable attention matrices)
- Time dimension preserved (T=188 for 3s audio)
- Additive skip from encoder to post-DP blocks (residual learning)
- F.interpolate bilinear upsample from 65→257 freq bins
- 3-epoch LR warmup (new for transformer stability)

### Expected Timeline:
- Training: ~4-6 hours on P100
- Eval: ~30 min (105 test samples)
- Target: PESQ > 1.163 (noisy baseline) — any improvement validates the architecture

### Next Steps:
1. Monitor training completion
2. Download results / checkpoint
3. If PESQ > noisy baseline → success, update capstone docs
4. If still failing → may need to try pure CNN approach or deeper DPT

---

## Session Log: Feb 24, 2026 (Continued) — R3 Eval Complete

### What Was Done This Session:
1. ✅ Created eval-only notebook `kjadeja/review-3-eval-stft-transformer` (ID: 110458458)
2. ✅ Pushed v1-v3 to Kaggle (v1: no inputs mounted, v2: syntax error, v3: KeyError on `ckpt['model']`)
3. ✅ Root cause of Kaggle eval failure: v4 re-ran training and only produced `ckpt_ep5.pth` (raw state_dict, no wrapper dict). The v3 best checkpoint was overwritten.
4. ✅ Downloaded v3's `stft_transformer_best.pth` directly from Kaggle (epoch 15, val=0.1050, 11.9 MB)
5. ✅ Downloaded test.7z (40.3 MB) + y_test.7z (36.1 MB) from `earth16/libri-speech-noise-dataset`
6. ✅ **Ran full eval locally on CPU** — 105 test samples
7. ✅ Saved `review3_summary.json`
8. ✅ Updated memory.md

### R3 Eval Results (LOCAL, CPU):
| Metric | Noisy | Enhanced (R3) | Δ |
|--------|-------|---------------|---|
| PESQ   | 1.163 | **1.089**     | **-0.074** |
| STOI   | 0.722 | **0.622**     | **-0.1009** |
| SI-SDR | -0.25 dB | **-1.65 dB** | **-1.40 dB** |

**All metrics WORSE than noisy input.** The model actually degrades speech quality.

### Review Comparison Table:
| Review | Pipeline | PESQ | STOI | SI-SDR | Params |
|--------|----------|------|------|--------|--------|
| R1 CRN | Mel (estimated) | ~3.10 | — | — | ~2.5M |
| R2 Transformer | Mel+GriffinLim | 1.141 | 0.695 | -25.58 dB | 2.45M |
| R3 Transformer | STFT+ISTFT | 1.089 | 0.622 | -1.65 dB | 2.45M |
| Noisy baseline | — | 1.163 | 0.722 | -0.25 dB | — |

### Key Insights:
1. **STFT fix worked for reconstruction** — SI-SDR went from -25.58 → -1.65 dB (massive improvement)
2. **The model itself is the problem** — it degrades audio instead of enhancing it
3. **Architecture bottleneck confirmed**: `mean(dim=2)` collapses 257 freq bins → transformer has zero frequency resolution → decoder cannot produce meaningful spectral masks
4. **Loss barely improved** (0.1072 → 0.1050, 2%) because the model converges to near-identity mask
5. **Next steps needed**: Fix frequency handling — either keep freq dimension through transformer, or use CRN-style architecture with STFT pipeline

### Files Created/Modified:
- `run_eval_local.py` — local eval script (soundfile backend, CPU)
- `review3_summary.json` — eval metrics JSON
- `build_eval_nb.py` — Kaggle eval notebook generator
- `inspect_output.py` — helper to inspect Kaggle output files
- `ckpt_dl/stft_transformer_best.pth` — downloaded checkpoint
- `data/test/`, `data/y_test/` — test audio files (105 pairs)

---

## Session Log: Feb 25, 2026 — Review 3 Training Results + v4 Push

### What Was Done This Session:
1. ✅ Created `review3-stft-transformer.ipynb` locally (18 cells, STFT-based pipeline)
2. ✅ Validated locally via `validate_r3.py` (STFT roundtrip error: 7.15e-07, 2.45M params)
3. ✅ Pushed to Kaggle as `kjadeja/review-3-stft-transformer-speech-enhancement` (ID: 110421493)
4. ✅ Debugged 3 PyTorch/Python 3.12 compat issues across v1-v3:
   - v1: `total_mem` → `total_memory` (getattr fallback)
   - v2: `verbose=False` removed from ReduceLROnPlateau (Python 3.12)
   - v3: `torch.load weights_only=True` blocks numpy globals in checkpoint
5. ✅ v3 training COMPLETED: 25 epochs, best_ep=15, best_val=0.1050, ~7291s
6. ✅ Analyzed training results: loss barely moved (0.1072→0.1050, 2% improvement)
7. ✅ Diagnosed architecture bottleneck (mean pooling destroys frequency info)
8. ✅ Fixed all 4 bugs + pushed v4 (will get PESQ/STOI/SI-SDR metrics)
9. ✅ Updated memory.md with R3 results + compat issues

### R3 v3 Training Results:
- Val loss: 0.1072 → 0.1050 (only 2% improvement, concerning)
- Compare R2: 0.1764 → 0.1485 (16% improvement)
- Each epoch: ~293s, total: ~7291s (~2 hours)
- LR schedule: 1e-3 → 5e-4 (ep11) → 2.5e-4 (ep21)
- Early stopping triggered at epoch 25 (patience 10 exhausted)

### Architecture Bottleneck Analysis:
The model barely learned because `mean(dim=2)` collapses all 257 frequency bins into
one vector per time step. The transformer only models temporal patterns, not spectral ones.
The decoder receives frequency-uniform features and must reconstruct a frequency-dependent
mask purely from its CNN weight patterns — this severely limits capacity.

### v4 Status: RUNNING on Kaggle
- All 4 fixes applied (getattr, no verbose, float(va_loss), weights_only=False)
- Will retrain (same architecture) + eval → get PESQ/STOI/SI-SDR
- Expected runtime: ~2.5 hours (2h training + 30min eval)
- URL: https://www.kaggle.com/code/kjadeja/review-3-stft-transformer-speech-enhancement

---

## Session Log: Feb 24, 2026 — Review 2 Results & Diagnosis

### What Was Done This Session:
1. ✅ Confirmed notebook `kjadeja/review-2-cnn-transformer-speech-enhancement` (v5) STATUS: COMPLETE
2. ✅ Extracted training metrics from Kaggle notebook log (3.5MB NDJSON)
3. ✅ Diagnosed root cause of failure (mel spectrogram non-invertibility)
4. ✅ Updated memory.md with full results + Review 3 fix plan
5. ✅ Updated learningsandprogress.md (this file)

### Review 2 Final Results (FAILED — needs fix):
| Metric | Noisy | Enhanced | CRN Baseline | Target |
|--------|-------|----------|--------------|--------|
| PESQ   | 1.144 | **1.141** | 3.10        | ≥ 3.2  |
| STOI   | 0.693 | **0.695** | —           | —      |
| SI-SDR | -0.82 | **-25.58 dB** | —       | —      |

Training ran all 25 epochs — val loss improved (0.1764 → 0.1485) — model IS learning.
But the reconstruction pipeline is catastrophically broken.

### Root Cause Diagnosis:
The mel filterbank is a MANY-TO-ONE mapping — there is no exact inverse.
`InverseMelScale(driver='gelsd')` is a least-squares pseudo-inverse that introduces
high-frequency noise, and `GriffinLim(n_iter=32)` cannot recover phase well.

When PESQ measures waveform quality, it sees the GriffinLim artifacts as severe distortion.
SI-SDR going from -0.82 dB to -25.58 dB confirms the reconstruction is adding NOISE.

**The CRN baseline PESQ=3.10 is valid because CRN uses STFT → sigmoid mask → ISTFT (no GriffinLim).**

### Review 3 Fix (IMMEDIATE PRIORITY):
Switch from MelSpectrogram to STFT. Apply mask to complex STFT. Use torch.istft for reconstruction.
- Input shape changes from (B, 128, T) to (B, 257, T) [n_fft=512 → 257 bins]
- Reconstruction is PERFECT (no information loss)
- This is what CRN does and why it works

See memory.md "Review 3 Plan" section for full architecture spec.

---

## Session Log: Feb 23, 2026 — Kaggle Upload & Training Run

### What Was Done:
1. ✅ Read `Project2.pdf` — understood Review 2 requirements
2. ✅ Read `kjadeja/baseline-crn-speechenhance` via Kaggle MCP — understood CRN architecture
3. ✅ Created `review2-transformer-speechenhance.ipynb` locally (25 cells, 14 code + 11 markdown)
4. ✅ Validated all locally-runnable cells on CPU:
   - Cell 2 (imports): ✅ runs OK
   - Cell 3 (dataset class): ✅ defined OK
   - Cell 4 (PositionalEncoding): ✅ (1,188,256) output shape correct
   - Cell 5 (ShallowTransformerEnhancer): ✅ 2.45M params, forward pass correct
   - Cell 6 (attention extraction): ✅ returns (1,4,188,188) per layer
   - Cell 10 (Griffin-Lim waveform): ✅ 128×188 spec → 2.99s wav, SI-SDR 19.15 dB
   - Cell 13 (attention viz): ✅ 2 layers × 4 heads renders correctly
   - SMOKE TEST (Cell 25): ✅ loss 0.2147 → 0.0769 in 10 synthetic steps
5. ✅ Saved clean notebook to Kaggle as `kjadeja/review-2-cnn-transformer-speech-enhancement`
6. ✅ Debugged dataset mounting (5 versions):
   - v1-v2: Dataset mounting via `datasetDataSources` doesn't work when updating by ID
   - v3: paths wrong after 7z extraction
   - v4: `num_workers=2` caused `^^` AssertionError spam between epochs
   - v5: Used `kaggle datasets download` CLI with internet enabled — WORKED
7. ✅ v5 ran to completion on P100-PCIE-16GB GPU

### Debugging Notes (for future sessions):
- `^^` spam = `AssertionError: can only test a child process` from `num_workers=2` DataLoader
  cleanup between epochs — **harmless** but annoying. Fix: `num_workers=0`
- `datasetDataSources` in `mcp_kaggle_save_notebook` DOES NOT WORK when updating by integer ID
  (only works for new notebook creation by slug). Always use `kaggle datasets download` CLI instead.
- 7z archives extract into nested subdirs. The `find_wav_dir()` helper auto-detects the actual
  wav directory path.
- `mcp_kaggle_save_notebook` with `hasId=true` + integer `id` = update existing notebook.
  Without `hasId`, uses slug to create new notebook.

---

## Key Architecture Reference (ShallowTransformerEnhancer):
```
Input: (B, 128, T) log-mel
  → CNN Encoder: Conv2d(1→64→128→256, 3×3, BN, ReLU)
  → mean(freq) → (B, T, 256) → linear(256→256)
  → PositionalEncoding(sinusoidal, d=256)
  → 2× TransformerEncoderLayer(d=256, heads=4, ff=1024, Pre-LN, dropout=0.1)
  → linear(256→256) → reshape → CNN Decoder(256→128→64→1)
  → Sigmoid → mask × noisy_mel
Params: 2,450,945 (2.45M)
```

## Phase Timeline:
| Phase | Date | Status | PESQ |
|-------|------|--------|------|
| Review 1 | Jan 21 | ✅ DONE | 3.10 (CRN baseline) |
| Review 2 | Feb 18 | ✅ DONE | 1.141 (**FAILED** — wrong pipeline) |
| Review 3 | Mar 18 | ⏳ NEXT | Target ≥ 3.2 (STFT fix) |
| Final | Apr 8 | ⏳ | Quantized + Gradio demo |

## NEXT SESSION — Immediate Actions:
```
1. Read memory.md to restore context
2. Create new notebook for Review 3 (STFT-based)
3. Architecture change: MelSpectrogram → STFT (n_fft=512, hop=128)
4. Input dim: 128 → 257 frequency bins
5. Reconstruction: mask × complex STFT → torch.istft
6. Target PESQ ≥ 3.2
7. Kaggle slug: kjadeja/review-3-stft-transformer-speech-enhancement
```
