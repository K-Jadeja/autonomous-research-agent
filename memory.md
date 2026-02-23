_Use this proactively for long term memory — LAST UPDATED: Feb 23, 2026_

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
- LSTM input bug in CRN (was 256×128=32768) → fixed by mean-pooling freq dim before LSTM
- PESQ/STOI in evaluation is ESTIMATED (no waveform reconstruction, no Griffin-Lim)
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
**Kaggle target slug:** `kjadeja/review2-transformer-speechenhance`  
**Status:** LOCAL VALIDATION COMPLETE. Not yet saved/run on Kaggle.

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

### Next Steps:
1. **Save notebook to Kaggle** using Kaggle MCP (`mcp_kaggle_save_notebook`)
2. Create a Kaggle notebook session and run cells 1→14 interactively via MCP
3. Verify PESQ ≥ 3.2 after training
4. Save final results and update `review2_summary.json`
5. Update this memory with final results

### Kaggle MCP Save Parameters (to use):
```json
{
  "slug": "review2-transformer-speechenhance",
  "language": "python",
  "kernelType": "notebook",
  "isPrivate": false,
  "enableGpu": true,
  "enableInternet": true,
  "datasetDataSources": ["earth16/libri-speech-noise-dataset"],
  "kernelExecutionType": "SaveAndRunAll"
}
```
