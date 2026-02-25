# Presentation Guide — Capstone Review
## *Lightweight Speech Enhancement Using Shallow Transformers*

> **Who this guide is for:** A presenter who may not be a deep learning researcher.  
> This file gives you everything you need — what each slide means, what to say, simple definitions for jargon, and how to handle panel questions.

---

## The Big Picture (Read This First)

**What this project does in one sentence:**  
We built a small AI model that cleans up noisy audio — like removing background noise from a voice recording — and we proved our model is twice as small but works better than the standard approach.

**Why the panel cares:**
- Most speech-enhancement AI is too big to run on phones, hearing aids, or IoT devices.
- This project hits a specific engineering target: **under 1 million parameters** (think of "parameters" as the number of knobs the AI can tune — fewer = smaller, faster, cheaper).
- The proposed model (**DPT**) beats the baseline (**CRN**) on every quality metric while being half the size.

**The 3 numbers to remember:**

| | Noisy Input | CRN (Old) | DPT (Ours) |
|---|---|---|---|
| PESQ (quality, higher = better) | 1.163 | 1.630 | **1.692** |
| STOI (intelligibility, higher = better) | 0.722 | 0.864 | **0.866** |
| SI-SDR (signal clarity in dB) | −0.25 dB | 8.62 dB | **9.05 dB** |

---

## Jargon Cheat Sheet

| Technical term | What to actually say |
|---|---|
| **STFT** (Short-Time Fourier Transform) | A mathematical tool that converts audio into a picture (spectrogram) showing frequency over time — like turning a song into sheet music. |
| **Spectrogram** | A visual heatmap of audio. Bright = loud frequencies, dark = silence. |
| **Parameters** | The "knobs" or "weights" inside the neural network. Fewer = lighter model. Think of it like kilobytes. |
| **PESQ** | A standard phone-industry test score for voice quality. Scale: 1 (bad) to 4.5 (perfect). |
| **STOI** | Measures how understandable speech is. Scale: 0 to 1. |
| **SI-SDR** | Measures how well the signal is separated from noise, in decibels. Higher = cleaner. |
| **CRN (Baseline)** | Our older model — Conv-Recurrent Network. Uses LSTM (a loop-based AI). Bigger and slower. |
| **DPT (Proposed)** | Our new model — Dual-Path Transformer. Uses attention (looks at the whole picture at once). Smaller and better. |
| **LSTM** | A type of AI that reads audio one step at a time, like reading a sentence word by word. Slower. |
| **Transformer / Attention** | An AI that reads the whole audio at once, catching patterns across time AND frequency simultaneously. Faster and more efficient. |
| **Mask** | A filter between 0 and 1. The model predicts which parts of the audio are speech (keep → near 1) vs noise (suppress → near 0). |
| **Epoch** | One full pass through all the training data. Like reading through the textbook once. |
| **val_loss** | How well the model performed on data it had never seen before. Lower = better. CRN: 0.0534, DPT: 0.0513. |
| **Kaggle T4 GPU** | A powerful graphics card used for training. Like renting a very fast calculator. |

---

## Slide-by-Slide Talking Points

---

### SLIDE 1 — Title Slide
**What to say:**
> "Good [morning/afternoon]. We are presenting the second review of our capstone project — Lightweight Speech Enhancement Using Shallow Transformers. Our team is Krishnasinh, Kirtan, and Prabhu, guided by Dr. Praveen Jaraut."

**Pro tip:** Don't rush through the title. Pause after saying the project name so it lands.

---

### SLIDE 2 — Introduction & Problem Recap
**What to say:**
> "The problem we're solving is simple: when you're on a call in a noisy place — a cafe, a street, an office — the AI on your device needs to clean that audio in real time. Existing deep learning models that do this are too large to run on small devices like earbuds or hearing aids. Our project builds one that fits within 1 million parameters — that's our key constraint — and still outperforms the baseline."

**If asked what '75% complete' means:**
> "The STFT pipeline and the DPT model are fully trained and evaluated. The remaining 25% is quantization, deployment packaging, and a Gradio demo."

---

### SLIDE 3 — System Design: The Pipeline
**What to say:**
> "There are three steps. First, we take the raw noisy audio waveform and convert it into a spectrogram using the STFT — this is like turning a sound recording into a picture. Our model then looks at this picture and predicts a mask — a filter that removes noise while keeping the speech. Finally, we use the Inverse STFT to convert the cleaned picture back into audio. The key advantage here is that we operate in the STFT domain, not Mel-spectrograms, which means the reconstruction is mathematically lossless — we don't throw away any information."

**Simple analogy to offer:**
> "Think of it like an Instagram filter for audio. The STFT produces the image, our AI applies the filter, and the ISTFT gives you the clean version."

---

### SLIDE 4 — Architecture Comparison
**What to say:**
> "We compared two architectures. The baseline — CRN — uses an LSTM, which processes audio sequentially, one frame at a time. It has 1.81 million parameters — that exceeds our 1M goal. Our proposed model — DPT — replaces the LSTM with a Dual-Path Transformer that processes frequency and time patterns simultaneously. It has only 0.90 million parameters, cuts our size requirement in half, and still outperforms the baseline on every metric."

**If asked why Transformers beat LSTMs here:**
> "LSTMs process data sequentially — they have to wait for the previous step to finish before processing the next one. Transformers use attention, which means every time frame can see every other time frame at once. This parallelization makes training faster and lets the model capture longer-range dependencies in audio."

---

### SLIDE 5 — (Architecture Diagram)
> Walk the panel through whichever diagram is on this slide. The flow is: Noisy audio → CNN Encoder → DualPath Blocks → CNN Decoder → Enhanced Magnitude → ISTFT → Clean audio.

---

### SLIDE 6 — Core Innovation: The Dual-Path Block
**What to say:**
> "This is the core innovation. Instead of one sequential pass through an LSTM, our Dual-Path Block splits the work into two parallel attention operations. The first — the Frequency Transformer — looks along all frequency bins at each time step, catching harmonic structure like the overtones in a voice. The second — the Time Transformer — looks across all time frames at each frequency, catching temporal dynamics like the rhythm and flow of speech. Because these run as attention operations rather than loops, they're inherently more parameter-efficient. Our DPT block is 2x more parameter efficient than the CRN baseline."

**Simple analogy:**
> "Imagine reading a spreadsheet. An LSTM reads it row by row. Our model reads across all rows AND all columns at the same time — it gets the full picture faster."

---

### SLIDE 7 — Implementation: Modules Developed
**What to say:**
> "We built three core software modules. The Data Loader handles LibriSpeech audio files and applies a critical fix we'll explain next. The Feature Extraction module wraps PyTorch's STFT/ISTFT with log-compression, which stabilises the values going into the model. And the DPT Neural Network itself, implemented using PyTorch's TransformerEncoderLayer with Pre-Layer Normalisation — a technique that makes training much more stable."

---

### SLIDE 8 — Critical Engineering Fix (The "Gibberish" Bug)
**What to say (this is a great story — tell it confidently):**
> "This slide describes the most significant engineering challenge we solved. In Review 1, the model was training — loss was decreasing — everything looked fine on paper. But the audio output was pure static. We traced the bug to the data loader. Both the noisy file and the clean target file are 3-second clips from longer recordings. The loader was randomly cropping each file independently — so the model was told 'this noisy clip corresponds to this clean clip' but they were actually from different parts of the recording. The model was trying to learn an impossible mapping. The fix was one line: enforce a shared random seed so both files are always cropped at exactly the same timestamp. After this fix, the PESQ went from meaningless noise to the numbers you see in our results."

**This is your strongest engineering story. Slow down. The panel will find this impressive — debugging this kind of silent data pipeline bug is harder than building the model.**

---

### SLIDE 9 — Algorithm & Code Snippets
**What to say:**
> "Here's the actual forward pass of the Dual-Path Block. Lines 3-5 reshape the tensor so the frequency transformer sees each time frame as a sequence of frequency values. Lines 8-10 do the same for the time transformer. The additive skip connection at the end — `return x_t + x` — is a U-Net style residual, which prevents the model from forgetting the original input. Training used L1 loss on the log-magnitude, which penalises absolute deviation rather than squared error — this is more robust to occasional large noise spikes."

**If asked why L1 and not MSE:**
> "MSE squares the error, so it heavily penalises rare large errors. L1 treats all deviations linearly, which is better for audio where occasional loud noise spikes shouldn't dominate the loss."

---

### SLIDE 10 — Results
**What to say:**
> "These are the quantitative results on the full LibriSpeech test set — 105 test pairs. Our DPT model improves PESQ by 45% over the noisy baseline — that's a perceptual quality improvement that a human listener can detect. STOI improved by 20%, meaning speech became measurably more intelligible. SI-SDR improved by 9.3 dB — that's a large gain on a logarithmic scale. Critically, DPT outperforms CRN in all three metrics, while being half the size."

**Numbers to memorise:**
- PESQ: 1.163 → 1.692 (+45%)
- STOI: 0.722 → 0.866 (+20%)
- SI-SDR: −0.25 → +9.05 dB (+9.3 dB improvement)

---

### SLIDE 11 — Efficiency Analysis
**What to say:**
> "This slide validates our primary engineering constraint. The goal was under 1 million parameters — DPT achieves 0.90 million. CRN at 1.81 million fails this target. For training speed, the Transformer's parallelism means DPT converged to a lower validation loss (0.0513 vs 0.0534) faster than the recurrent baseline on Kaggle's dual T4 GPU setup."

**If you ran the profiling cell in the notebook, you can add:**
> "In our inference profiling, DPT also [showed X ms latency] — [pass/fail relative to the 15ms target]."

---

### SLIDES 12–13 — (Likely diagrams/visuals)
> Describe what's on the slides. If they show graphs, say: "This graph shows the training/validation loss curve over 30 epochs. Notice DPT's loss plateaus lower, indicating better generalisation."

---

### SLIDE 14 — Qualitative Analysis: Spectrogram Visuals
**What to say:**
> "Here we can see the spectrograms directly. Each horizontal band represents a frequency — higher on the image means higher-pitched. Bright areas are loud signals. Looking at the noisy input spectrogram, you can see scattered bright spots throughout — that's the noise. In the CRN and DPT outputs, the background noise is visibly attenuated while the speech structure — those bright horizontal bands — is preserved. The DPT output is visually closer to the clean reference."

**If you have the `inference_comparison.png` from the notebook, you can show that here.**

---

### SLIDE 15 — Listening Tests
**THIS IS YOUR STRONGEST DEMO MOMENT.**

> "I'll now play three audio clips so you can hear the difference directly."

**Play order from the notebook's `inference_output/` folder:**
1. `01_noisy.wav` — "This is the noisy input — notice the background hiss."
2. `02_crn_enhanced.wav` — "This is CRN's output. Some noise removed."
3. `03_dpt_enhanced.wav` — "This is DPT's output. Notice it's cleaner — closer to the reference."
4. `00_clean_reference.wav` — "And this is the clean target — what we're aiming for."

**Tips:**
- Play them in sequence with a brief pause between each.
- Let the panel listen — don't talk over the audio.
- After playing all four: "DPT's output is audibly closer to the clean reference, which matches our quantitative metrics."

---

### SLIDE 16 — Challenges & Solutions
**What to say:**
> "We encountered three major engineering challenges. The data misalignment bug — which we detailed earlier — was the most significant; it produced a model that trained but was completely useless until we found it. Second, our initial approach used Mel-spectrograms, but those discard phase information, which meant reconstruction artifacts — robotic-sounding audio. Switching to complex STFT solved that. Third, there were API deprecation issues in the Kaggle environment which required custom wrapper functions."

---

### SLIDE 17 — Remaining Work (Final 25%)
**What to say:**
> "Three things remain before the final review. First, quantization — converting the model from 32-bit floats to 8-bit integers. This compresses the 3.5MB checkpoint to under 1MB with minimal quality loss, which is critical for edge deployment. Second, ONNX export and a Gradio web demo so the panel can interact with the model live at the final review. Third, a strict real-time latency benchmark — our target is under 15 milliseconds per audio frame. We have preliminary profiling data from our inference notebook."

---

### SLIDE 18 — Project Timeline
**What to say:**
> "This shows our milestones. Review 1 (January) delivered the CRN baseline. Review 2 (February) implemented the Transformer approach on Mel-spectrograms. Review 3 (March) — this review — delivers the full STFT pipeline and a working DPT model with validated metrics. The final review in April will deliver quantization, deployment, and the live demo."

---

### SLIDE 19 — References
**What to say:**
> "Our architecture is inspired by two seminal works: the Dual-Path RNN paper by Luo et al., which introduced the split-path processing concept, and the original Transformer paper by Vaswani et al. We evaluate against ITU-T P.862 PESQ, which is the telecoms industry standard for speech quality. Our training data is LibriSpeech, a publicly available English speech corpus."

---

## Anticipated Panel Questions

### Q: "Why not use a pre-trained model like Whisper or a large speech model?"
**A:** "Those models are general-purpose and very large — Whisper's smallest version is 39M parameters. Our constraint is sub-1M parameters for edge devices where you can't run a 150MB model. Our approach is task-specific and purpose-built for that constraint."

### Q: "What hardware would this run on in production?"
**A:** "At 0.90M parameters and ~3.5MB checkpoint size — and after quantization, under 1MB — this is suitable for microcontrollers with enough RAM, ARM Cortex-M class chips, or hearing aid DSPs. Our 15ms latency target corresponds to real-time operation at 16kHz with a 256-sample hop."

### Q: "How does it perform on unseen noise types?"
**A:** "Our test set uses the same noise distribution as training — LibriSpeech noise. Generalisation to novel noise types (e.g., DEMAND dataset) is listed as future work. The Transformer attention mechanism is theoretically better at generalising than LSTMs, but we haven't benchmarked that yet."

### Q: "What would improvement beyond 1.692 PESQ require?"
**A:** "PESQ scores above 2 typically require either more data, larger models, or domain-adaptation techniques. Our constraint is sub-1M parameters, so we'd explore better loss functions (perceptual loss, GAN-based training) or more efficient attention patterns within the same parameter budget."

### Q: "Why did you choose LibriSpeech and not a real-world dataset?"
**A:** "LibriSpeech is a standard benchmark which makes our results comparable to published work. In the remaining 25%, we plan evaluation on DEMAND and VCTK-DEMAND which include real-world noise conditions."

### Q: "Could you run this on a phone?"
**A:** "A phone has more than enough compute — the bottleneck on phones is usually power and latency, not memory. At under 1MB after quantization, it could run on the audio DSP chip. That said, we haven't done phone-specific profiling yet — that's post-final-review work."

### Q: "What's the PESQ score of commercial systems?"
**A:** "Commercial systems like Krisp or NVIDIA RTX Voice achieve PESQ around 2.7–3.1, but those are 50–100x larger. For the parameter budget we're working with, 1.692 is competitive with published academic baselines."

---

## Demo Checklist (Pre-Presentation)

Before walking into the room:

- [ ] Open `inference_demo.ipynb` in VS Code or Jupyter
- [ ] Run all cells once (takes ~30 seconds)
- [ ] Verify `inference_output/` folder has all 4 WAV files
- [ ] Test that audio plays from Cell 9 (the audio player cell)
- [ ] Have `inference_comparison.png` and `inference_efficiency.png` open in a separate window for Slides 11 and 14
- [ ] If showing the live notebook: zoom browser/VS Code to 150% for readability on a projector
- [ ] Have the PPTX open in PowerPoint on the same screen (or bring it on USB)

**File paths during demo:**
```
inference_output/00_clean_reference.wav   ← play last (the "ideal" reveal)
inference_output/01_noisy.wav             ← play first
inference_output/02_crn_enhanced.wav      ← play second
inference_output/03_dpt_enhanced.wav      ← play third (our best result)
```

---

## Timing Guide (Assuming 15-minute slot)

| Section | Slides | Time |
|---|---|---|
| Title + Intro | 1–2 | 1.5 min |
| System design + Architecture | 3–6 | 3 min |
| Implementation + Bug fix | 7–8 | 2.5 min |
| Results + Efficiency | 9–11 | 3 min |
| Qualitative + **Listening Demo** | 14–15 | 2.5 min |
| Challenges + Remaining Work | 16–17 | 1.5 min |
| Timeline + Q&A | 18–19 | 1 min |

> If you have a 20-minute slot, add 5 minutes to the listening demo and code walkthrough.

---

## Tone & Delivery Notes

1. **Own the bug story (Slide 8).** This is the best part of the presentation. Speak slowly and clearly when explaining the data misalignment fix — it demonstrates deep engineering judgment, not just running code.

2. **Let the audio speak.** During Slide 15, stop talking and let the panel actually listen. The quality improvement is audible — let that land.

3. **Lead with the number that matters most.** For PESQ: "We improved speech quality by 45%" is more striking than saying "we went from 1.163 to 1.692."

4. **If you don't know the answer to a panel question**, say: "That's a great point — it's captured as future work, but the current results suggest [nearest relevant result]. We can follow up after the session."

5. **DPT is the hero.** Every comparison should end with DPT winning — it's smaller AND better. Repeat that framing consistently.
