_Use this proactively for short term memory and to track the processes — LAST UPDATED: Feb 24, 2026_

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
