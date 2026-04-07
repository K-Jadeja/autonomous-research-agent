# Speech Enhancement: CRN vs Dual-Path Transformer (DPT)

## Project Outcomes & Computational Analysis

**Team:** Krishnasinh Jadeja (22BLC1211) · Kirtan Sondagar (22BLC1228) · Prabhu Kalyan Panda (22BLC1213)
**Guide:** Dr. Praveen Jaraut — VIT Chennai

---

## What This Project Does

We built two speech enhancement models that clean noisy audio — removing background noise while preserving the speaker's voice. Both models operate on STFT spectrograms and predict a soft mask that separates speech from noise.

| | CRN v2 (Baseline) | DPT v2 (Proposed) |
|---|---|---|
| **Architecture** | CNN Encoder → LSTM → CNN Decoder | CNN Encoder → Dual-Path Transformer → CNN Decoder |
| **Idea** | Recurrent layers capture temporal patterns | Self-attention captures both frequency and time patterns |
| **Input** | STFT magnitude (257 x T) | STFT magnitude (257 x T) |
| **Output** | Soft mask → enhanced spectrogram | Soft mask → enhanced spectrogram |

---

## The Big Result

**DPT beats CRN on every metric while being half the size and using a third less compute.**

### Audio Quality Metrics (averaged over 105 test samples)

| Metric | Noisy Input | CRN v2 | DPT v2 | What it means |
|---|---|---|---|---|
| **PESQ** (1–4.5) | 1.163 | 1.630 | **1.692** | Perceptual speech quality — higher = sounds better to humans |
| **STOI** (0–1) | 0.723 | 0.864 | **0.866** | Short-time intelligibility — higher = more words understandable |
| **SI-SDR** (dB) | -0.25 | 8.62 | **9.05** | Signal-to-noise improvement — higher = cleaner signal |

**What this means:**
- PESQ goes from 1.163 (barely intelligible) to 1.692 (noticeably cleaner) — a **+0.529** improvement
- STOI jumps from 0.723 to 0.866 — roughly **14% more words** become understandable
- SI-SDR improves by **~9.3 dB** — the noise power is reduced to about 1/8th of its original level

### Single Sample Visualization

![Spectrogram Comparison](inference_comparison.png)

*Left to right: Noisy input → CRN enhanced → DPT enhanced → Clean target. Notice how DPT suppresses more background noise (darker regions) while keeping speech patterns intact.*

---

## Computational Cost Comparison

This is where DPT really shines. We measured GFLOPs (billion floating-point operations) per forward pass using the `thop` profiler.

### The Numbers (3-second audio, input shape 1×1×257×188)

| Metric | CRN v2 | DPT v2 | CRN/DPT Ratio |
|---|---|---|---|
| **Parameters** | 1.811M | **0.905M** | 2.00x |
| **GFLOPs** | 15.74 | **10.50** | 1.50x |
| **Checkpoint size** | 7.07 MB | **3.56 MB** | 1.99x |
| **Params < 1M goal?** | No (over budget) | **Yes** | — |
| **Latency < 15ms target?** | Pass | **Pass** | — |

**What GFLOPs means:** One GFLOP = 1 billion floating-point math operations (multiply + add). A forward pass through CRN requires ~15.7 billion such operations. DPT only needs ~10.5 billion — that's **33% less compute**.

**Why this matters for deployment:**
- On edge devices (phones, hearing aids, IoT), fewer FLOPs = longer battery life
- On cloud servers, fewer FLOPs = lower inference cost per request
- DPT's 0.9M parameters fit within the project's <1M constraint; CRN doesn't

![FLOPs Comparison](docs/chart1_flops_comparison.png)

*Left to right: Parameter count (DPT is 50% smaller), GFLOPs per pass (DPT uses 33% less compute), compute efficiency per parameter, and the CRN-to-DPT ratio across all metrics.*

### How Compute Scales with Audio Length

![FLOPs Scaling](docs/chart2_flops_scaling.png)

*Both models scale roughly linearly with audio duration. DPT consistently uses less compute at every duration. The gap widens slightly for longer clips because the LSTM in CRN processes each time step sequentially, while the Transformer processes in parallel.*

### Quality vs Efficiency

![Quality vs Efficiency](docs/chart3_quality_efficiency.png)

*All metrics normalized so higher = better. "Params (inv.)" and "GFLOPs (inv.)" are inverted because fewer is better. DPT (green) wins on both quality AND efficiency — you don't usually see that tradeoff.*

---

## Model Architecture Comparison

### CRN (Conv-Recurrent Network)
```
Input (1×257×T)
  → Conv2d 1→64 [stride-2 freq]    (129×T)
  → Conv2d 64→128 [stride-2 freq]  (65×T)
  → Conv2d 128→256 [stride-2 freq] (33×T)
  → 2-layer LSTM (256→256)         per frequency bin
  → ConvTranspose2d 256→128        (65×T)
  → ConvTranspose2d 128→64         (129×T)
  → ConvTranspose2d 64→32          (257×T)
  → Conv2d 32→1 + Sigmoid          mask output
```
- 1,811,009 parameters
- The LSTM (58% of params) processes each frequency bin independently over time
- Sequential nature limits parallelism

### DPT (Dual-Path Transformer)
```
Input (1×257×T)
  → Conv2d 1→32 [stride-2 freq]    (129×T)
  → Conv2d 32→128 [stride-2 freq]  (65×T)
  → DualPathBlock ×2:
      → Freq Transformer (attend across 65 freq bins)
      → Time Transformer (attend across T time frames)
  → +Skip connection
  → Bilinear upsample                  (257×T)
  → Conv2d 128→64 → Conv2d 64→1 + Sigmoid  mask output
```
- 904,705 parameters (50% fewer)
- Each DualPathBlock does: attention across frequency, then attention across time
- Skip connection preserves low-level features from the encoder
- Fully parallelizable — no sequential bottleneck like LSTM

---

## Predicted Masks

The models output a soft mask M(f,t) ∈ [0,1] for each frequency-time cell:
- **1.0** = keep this (speech)
- **0.0** = suppress this (noise)

![Masks](inference_masks.png)

*CRN's mask (left) vs DPT's mask (right). DPT produces sharper speech/noise boundaries, which translates to cleaner output with fewer artifacts.*

---

## Inference Efficiency

![Efficiency](inference_efficiency.png)

*Parameter count, inference latency, and peak RAM usage during a single forward pass. DPT meets the <1M parameter goal; CRN exceeds it by 81%.*

---

## Key Takeaways

1. **DPT is the better model.** It scores higher on PESQ (+0.062), STOI (+0.002), and SI-SDR (+0.43 dB) compared to CRN.

2. **DPT is much more efficient.** 50% fewer parameters, 33% fewer FLOPs, 50% smaller checkpoint file.

3. **DPT meets the project constraint.** The <1M parameter requirement is satisfied by DPT (0.905M) but not by CRN (1.811M).

4. **Both models run in real-time.** Inference latency is well under the 15ms target for both models.

5. **The Transformer advantage is real.** Self-attention captures long-range dependencies in both frequency and time that the LSTM misses, while being more parallelizable and parameter-efficient.

---

## Files in This Project

| File | Purpose |
|---|---|
| `flops_comparison.ipynb` | FLOPs profiling notebook with all charts |
| `inference_demo.ipynb` | Full inference demo with audio playback |
| `docs/chart1_flops_comparison.png` | Computational cost bar charts |
| `docs/chart2_flops_scaling.png` | GFLOPs vs audio duration plot |
| `docs/chart3_quality_efficiency.png` | Quality-efficiency profile |
| `inference_efficiency.png` | Params/latency/RAM comparison |
| `inference_comparison.png` | Spectrogram comparison |
| `inference_masks.png` | Predicted mask visualization |
| `inference_metrics.png` | Quality metrics bar charts |
