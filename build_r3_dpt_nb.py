#!/usr/bin/env python3
"""Build Review 3 DPT (Dual-Path Transformer) notebook for Kaggle push."""
import json, textwrap

# ─── Cell sources ─────────────────────────────────────────────────────────

md_title = textwrap.dedent("""\
# Review 3: Dual-Path Transformer STFT Speech Enhancement

**Key Fix from R3-v1:** Replaced single-path transformer (which collapsed frequency via
`mean(dim=2)`) with a **Dual-Path Transformer (DPT)** that alternately processes frequency
and time dimensions — giving the model full spectral resolution.

**Why R3-v1 failed (PESQ=1.089, *worse* than noisy 1.163):**
The `mean(dim=2)` pooling destroyed all 257 frequency bins. The transformer operated only on
temporal patterns and converged to a near-identity mask. Enhanced ≈ noisy.

**DPT architecture (this notebook):**
```
Input: (B, 1, 257, T) ← log1p(STFT magnitude)
  → CNN Encoder (freq stride 2×2): (B, 128, 65, T)
  → 2 × DualPathBlock:
       Freq-Transformer: attend across 65 freq sub-bands per time step
       Time-Transformer: attend across T time frames per freq bin
  → Skip + Interpolate to (B, 128, 257, T)
  → CNN Decoder + Sigmoid → mask (B, 257, T)
Reconstruction: mask × noisy_mag → ISTFT with noisy phase → waveform
```
**Params:** ~900K (lightweight) | **Pipeline:** STFT → DPT mask → ISTFT (lossless)

**Team:** Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)
**Guide:** Dr. Praveen Jaraut — VIT Bhopal Capstone""")

# ------------------------------------------------------------------
cell1_imports = textwrap.dedent("""\
# ============================================================================
# Cell 1: Install deps + Imports + Config
# ============================================================================
!pip install pesq==0.0.4 pystoi -q

import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import glob, os, json, time, warnings
warnings.filterwarnings('ignore')
from pesq import pesq as pesq_metric
from pystoi import stoi as stoi_metric

torch.manual_seed(42)
np.random.seed(42)

# STFT config (same as all reviews)
N_FFT      = 512
HOP_LENGTH = 256
N_FREQ     = N_FFT // 2 + 1   # 257
SR         = 16000
MAX_LEN    = 48000             # 3 s

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram/1e9:.1f}GB')
print(f'STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, freq={N_FREQ}')
print('Imports OK')""")

# ------------------------------------------------------------------
md_dataset = "## Dataset: LibriSpeech-Noise\nDownload & extract `earth16/libri-speech-noise-dataset` (7000 train + 105 test WAV pairs)."

cell2_data = textwrap.dedent("""\
# ============================================================================
# Cell 2: Dataset download & extraction (Kaggle)
# ============================================================================
import subprocess, zipfile

data_base = '/kaggle/working/data'
dl_tmp    = '/kaggle/working/dl_tmp'
os.makedirs(data_base, exist_ok=True)
os.makedirs(dl_tmp, exist_ok=True)
done_flag = os.path.join(data_base, '.done')

if os.path.exists(done_flag):
    print('Dataset already extracted, skipping')
else:
    mounted = '/kaggle/input/libri-speech-noise-dataset'
    if os.path.isdir(mounted) and len(os.listdir(mounted)) > 0:
        src = mounted
        print(f'Using mounted dataset at {src}')
    else:
        print('Dataset not mounted, downloading via kaggle API...')
        subprocess.run(['kaggle', 'datasets', 'download',
                        'earth16/libri-speech-noise-dataset', '-p', dl_tmp], check=True)
        zf = os.path.join(dl_tmp, 'libri-speech-noise-dataset.zip')
        if os.path.exists(zf):
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(dl_tmp)
            os.remove(zf)
        src = dl_tmp
        print(f'Downloaded to {src}')

    subprocess.run(['apt-get', 'install', '-y', 'p7zip-full'], capture_output=True)
    for arch in ['train.7z', 'y_train.7z', 'test.7z', 'y_test.7z']:
        fp = os.path.join(src, arch)
        if os.path.exists(fp):
            print(f'Extracting {arch}...')
            subprocess.run(['7z', 'x', fp, f'-o{data_base}', '-y'], capture_output=True)
    open(done_flag, 'w').close()

def find_wav_dir(base, name):
    for root, dirs, files in os.walk(base):
        if os.path.basename(root) == name and any(f.endswith('.wav') for f in files):
            return root
    return None

noisy_train = find_wav_dir(data_base, 'train')
clean_train = find_wav_dir(data_base, 'y_train')
noisy_test  = find_wav_dir(data_base, 'test')
clean_test  = find_wav_dir(data_base, 'y_test')

for tag, d in [('noisy_train', noisy_train), ('clean_train', clean_train),
               ('noisy_test', noisy_test), ('clean_test', clean_test)]:
    n = len(glob.glob(os.path.join(d, '*.wav'))) if d else 0
    print(f'  {tag}: {d} ({n} files)')""")

# ------------------------------------------------------------------
md_stft_ds = "## STFT Dataset Class\nLoads WAV pairs → STFT → returns magnitude, phase, waveforms."

cell3_dataset = textwrap.dedent("""\
# ============================================================================
# Cell 3: STFTSpeechDataset
# ============================================================================
class STFTSpeechDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sr=SR, max_len=MAX_LEN):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        assert len(self.noisy_files) == len(self.clean_files)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.max_len = max_len
        self.window = torch.hann_window(n_fft)

    def __len__(self):
        return len(self.noisy_files)

    def _load_fix(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav[0]
        if wav.shape[0] > self.max_len:
            start = torch.randint(0, wav.shape[0] - self.max_len, (1,)).item()
            wav = wav[start:start + self.max_len]
        elif wav.shape[0] < self.max_len:
            wav = F.pad(wav, (0, self.max_len - wav.shape[0]))
        return wav

    def __getitem__(self, idx):
        noisy_wav = self._load_fix(self.noisy_files[idx])
        clean_wav = self._load_fix(self.clean_files[idx])
        noisy_stft = torch.stft(noisy_wav, self.n_fft, self.hop_length,
                                window=self.window, return_complex=True)
        clean_stft = torch.stft(clean_wav, self.n_fft, self.hop_length,
                                window=self.window, return_complex=True)
        return {
            'noisy_mag':   noisy_stft.abs(),
            'clean_mag':   clean_stft.abs(),
            'noisy_phase': torch.angle(noisy_stft),
            'noisy_wav':   noisy_wav,
            'clean_wav':   clean_wav,
        }

print(f'STFTSpeechDataset defined (n_fft={N_FFT}, hop={HOP_LENGTH})')""")

# ------------------------------------------------------------------
md_model = textwrap.dedent("""\
## Model: Dual-Path Transformer (DPT)

**Core innovation:** Instead of collapsing frequency bins with `mean(dim=2)` (R3-v1),
the DPT processes **both** frequency and time dimensions via alternating transformer blocks:

- **Freq-Transformer:** For each time step, attend across 65 frequency sub-bands
- **Time-Transformer:** For each frequency sub-band, attend across T time frames

This gives the model **full spectral resolution** for frequency-selective noise masking.

```
CNN Encoder (stride-2 on freq): (B,1,257,T) → (B,128,65,T)
  → DualPathBlock #1: FreqTransformer(65,128) + TimeTransformer(T,128)
  → DualPathBlock #2: FreqTransformer(65,128) + TimeTransformer(T,128)
  → Skip + Interpolate → (B,128,257,T)
  → CNN Decoder + Sigmoid → mask (B,257,T)
```""")

cell4_model = textwrap.dedent("""\
# ============================================================================
# Cell 4: Dual-Path Transformer Model
# ============================================================================
class DualPathBlock(nn.Module):
    \"\"\"One dual-path block: frequency-transformer + time-transformer.\"\"\"
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.freq_transformer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.time_transformer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)

    def forward(self, x):
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        # --- Frequency path: attend across freq bins for each time step ---
        x_f = x.permute(0, 3, 2, 1).reshape(B * T, F, C)   # (BT, F, C)
        x_f = self.freq_transformer(x_f)                      # (BT, F, C)
        x = x_f.reshape(B, T, F, C).permute(0, 3, 2, 1)      # (B, C, F, T)
        # --- Time path: attend across time for each freq bin ---
        x_t = x.permute(0, 2, 3, 1).reshape(B * F, T, C)   # (BF, T, C)
        x_t = self.time_transformer(x_t)                      # (BF, T, C)
        x = x_t.reshape(B, F, T, C).permute(0, 3, 1, 2)      # (B, C, F, T)
        return x


class DPTSTFTEnhancer(nn.Module):
    \"\"\"Dual-Path Transformer for STFT-based Speech Enhancement.\"\"\"
    def __init__(self, n_freq=257, d_model=128, nhead=4, num_dp_blocks=2,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        self.n_freq = n_freq
        # CNN Encoder: downsample frequency by 4x (257 → 65)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, d_model, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
        )
        # Dual-Path blocks
        self.dp_blocks = nn.ModuleList([
            DualPathBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_dp_blocks)
        ])
        # CNN Decoder: upsample freq back
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, n_freq, T)
        B, _, F_orig, T_orig = x.shape
        h = self.encoder(x)           # (B, d_model, F', T)
        skip = h                       # encoder skip
        for block in self.dp_blocks:
            h = block(h)
        h = h + skip                   # residual
        h = F.interpolate(h, size=(F_orig, T_orig),
                          mode='bilinear', align_corners=False)
        return self.decoder(h).squeeze(1)   # (B, F_orig, T)


# ---- Quick test ----
model = DPTSTFTEnhancer(n_freq=N_FREQ).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'DPTSTFTEnhancer: {total_params:,} params ({total_params/1e6:.2f}M)')

with torch.no_grad():
    dummy = torch.randn(2, 1, N_FREQ, 188).to(device)
    out = model(dummy)
    print(f'Input: {dummy.shape} -> Mask: {out.shape}')
    assert out.shape == (2, N_FREQ, 188)
    assert out.min().item() >= 0 and out.max().item() <= 1
    print('Forward pass OK')

# Architecture breakdown
enc_p = sum(p.numel() for p in model.encoder.parameters())
dp_p  = sum(p.numel() for p in model.dp_blocks.parameters())
dec_p = sum(p.numel() for p in model.decoder.parameters())
print(f'  Encoder:     {enc_p:>8,} ({enc_p/total_params*100:.1f}%)')
print(f'  DPT blocks:  {dp_p:>8,} ({dp_p/total_params*100:.1f}%)')
print(f'  Decoder:     {dec_p:>8,} ({dec_p/total_params*100:.1f}%)')""")

# ------------------------------------------------------------------
cell5_utils = textwrap.dedent("""\
# ============================================================================
# Cell 5: SI-SDR utility
# ============================================================================
def si_sdr(estimate, reference):
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    dot = torch.sum(ref * est)
    s_target = dot * ref / (torch.sum(ref**2) + 1e-8)
    e_noise  = est - s_target
    return 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + 1e-8) + 1e-8)

print('si_sdr defined')""")

# ------------------------------------------------------------------
md_training = textwrap.dedent("""\
## Training
L1 loss on log-magnitude, Adam lr=1e-3 with 3-epoch linear warmup, then ReduceLROnPlateau.
Batch=8 (for GPU memory headroom with dual-path attention). 30 epochs, patience=12.""")

cell6_setup = textwrap.dedent("""\
# ============================================================================
# Cell 6: Training setup
# ============================================================================
MAX_EPOCHS    = 30
LR            = 1e-3
BATCH         = 8
PATIENCE      = 12
WARMUP_EPOCHS = 3
CKPT          = 'dpt_stft_best.pth'

# Re-init model fresh
model = DPTSTFTEnhancer(n_freq=N_FREQ).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total_params:,}')

# Datasets
full_train = STFTSpeechDataset(noisy_train, clean_train)
n_val   = int(0.1 * len(full_train))
n_train = len(full_train) - n_val
train_ds, val_ds = torch.utils.data.random_split(
    full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
test_ds = STFTSpeechDataset(noisy_test, clean_test)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.5, patience=5)

print(f'Train:{n_train} Val:{n_val} Test:{len(test_ds)} | BS={BATCH} LR={LR}')
print(f'Warmup: {WARMUP_EPOCHS} ep, then ReduceLROnPlateau(patience=5, factor=0.5)')""")

# ------------------------------------------------------------------
cell7_train = textwrap.dedent("""\
# ============================================================================
# Cell 7: Training loop with warmup
# ============================================================================
history = {'train_loss': [], 'val_loss': []}
best_val = float('inf')
patience_ctr = 0
t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    # --- LR warmup ---
    if epoch <= WARMUP_EPOCHS:
        warmup_lr = LR * epoch / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

    # --- Train ---
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f'Ep{epoch}/{MAX_EPOCHS}', leave=False):
        noisy_mag = batch['noisy_mag'].to(device)
        clean_mag = batch['clean_mag'].to(device)
        inp  = torch.log1p(noisy_mag).unsqueeze(1)   # (B, 1, 257, T)
        mask = model(inp)                              # (B, 257, T)
        enhanced_mag = mask * noisy_mag
        loss = F.l1_loss(torch.log1p(enhanced_mag), torch.log1p(clean_mag))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validate ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            noisy_mag = batch['noisy_mag'].to(device)
            clean_mag = batch['clean_mag'].to(device)
            inp  = torch.log1p(noisy_mag).unsqueeze(1)
            mask = model(inp)
            enhanced_mag = mask * noisy_mag
            loss = F.l1_loss(torch.log1p(enhanced_mag), torch.log1p(clean_mag))
            val_losses.append(loss.item())

    tr_loss = np.mean(train_losses)
    va_loss = np.mean(val_losses)
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)

    if epoch > WARMUP_EPOCHS:
        scheduler.step(va_loss)

    elapsed = time.time() - t0
    lr_now  = optimizer.param_groups[0]['lr']
    line = f'Ep{epoch:02d} tr={tr_loss:.4f} va={va_loss:.4f} lr={lr_now:.1e} [{elapsed:.0f}s]'

    if va_loss < best_val:
        best_val = va_loss
        patience_ctr = 0
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                     'val_loss': float(va_loss)}, CKPT)
        print(f'{line}  SAVED best={va_loss:.4f}')
    else:
        patience_ctr += 1
        print(f'{line}  no improve ({patience_ctr}/{PATIENCE})')

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'ckpt_ep{epoch}.pth')

    if patience_ctr >= PATIENCE:
        print(f'Early stopping at epoch {epoch}')
        break

best_ep = history['val_loss'].index(min(history['val_loss'])) + 1
print(f'\\nDONE best_ep={best_ep} best_val={best_val:.4f} time={time.time()-t0:.0f}s')""")

# ------------------------------------------------------------------
md_results = "## Results"

cell8_curves = textwrap.dedent("""\
# ============================================================================
# Cell 8: Training curves
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
eps = range(1, len(history['train_loss']) + 1)
ax.plot(eps, history['train_loss'], 'b-o', label='Train', ms=3)
ax.plot(eps, history['val_loss'], 'r-s', label='Val', ms=3)
ax.axvline(best_ep, color='g', ls='--', alpha=0.7, label=f'Best (ep{best_ep})')
if WARMUP_EPOCHS > 0:
    ax.axvline(WARMUP_EPOCHS, color='orange', ls=':', alpha=0.7, label='Warmup end')
ax.set_xlabel('Epoch')
ax.set_ylabel('L1 Loss (log-magnitude)')
ax.set_title('Review 3: DPT-STFT Enhancer Training')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print('Saved training_curves.png')""")

# ------------------------------------------------------------------
md_eval = "## Evaluation\nPESQ / STOI / SI-SDR on 105 test samples with ISTFT waveform reconstruction."

cell9_eval = textwrap.dedent("""\
# ============================================================================
# Cell 9: Evaluation — PESQ / STOI / SI-SDR
# ============================================================================
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'Loaded: epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f}')

window_eval = torch.hann_window(N_FFT).to(device)

pesq_noisy_list, pesq_enh_list   = [], []
stoi_noisy_list, stoi_enh_list   = [], []
sisdr_noisy_list, sisdr_enh_list = [], []

for i in tqdm(range(len(test_ds)), desc='Eval'):
    s           = test_ds[i]
    noisy_mag   = s['noisy_mag'].unsqueeze(0).to(device)
    noisy_phase = s['noisy_phase'].unsqueeze(0).to(device)
    clean_np    = s['clean_wav'].numpy()
    noisy_np    = s['noisy_wav'].numpy()

    with torch.no_grad():
        inp     = torch.log1p(noisy_mag).unsqueeze(1)
        mask    = model(inp)
        enh_mag = (mask * noisy_mag).squeeze(0)

    enh_stft = enh_mag * torch.exp(1j * noisy_phase.squeeze(0))
    enh_wav  = torch.istft(enh_stft, N_FFT, HOP_LENGTH,
                           window=window_eval, length=MAX_LEN)
    enh_np   = enh_wav.cpu().numpy()

    try:
        pesq_noisy_list.append(pesq_metric(SR, clean_np, noisy_np, 'wb'))
        pesq_enh_list.append(  pesq_metric(SR, clean_np, enh_np,   'wb'))
    except Exception as e:
        print(f'  PESQ err {i}: {e}')

    stoi_noisy_list.append(stoi_metric(clean_np, noisy_np, SR, extended=False))
    stoi_enh_list.append(  stoi_metric(clean_np, enh_np,   SR, extended=False))

    c_t = torch.from_numpy(clean_np).float()
    n_t = torch.from_numpy(noisy_np).float()
    e_t = torch.from_numpy(enh_np).float()
    sisdr_noisy_list.append(si_sdr(n_t, c_t).item())
    sisdr_enh_list.append(  si_sdr(e_t, c_t).item())

avg = lambda lst: float(np.mean(lst)) if lst else 0.0
avg_pesq_n,  avg_pesq_e  = avg(pesq_noisy_list),  avg(pesq_enh_list)
avg_stoi_n,  avg_stoi_e  = avg(stoi_noisy_list),  avg(stoi_enh_list)
avg_sisdr_n, avg_sisdr_e = avg(sisdr_noisy_list), avg(sisdr_enh_list)

print(f'\\nResults on {len(test_ds)} test files:')
print(f'  PESQ  : noisy={avg_pesq_n:.3f}  enhanced={avg_pesq_e:.3f}  d={avg_pesq_e-avg_pesq_n:+.3f}')
print(f'  STOI  : noisy={avg_stoi_n:.3f}  enhanced={avg_stoi_e:.3f}  d={avg_stoi_e-avg_stoi_n:+.4f}')
print(f'  SI-SDR: noisy={avg_sisdr_n:.2f}dB  enh={avg_sisdr_e:.2f}dB  d={avg_sisdr_e-avg_sisdr_n:+.2f}dB')""")

# ------------------------------------------------------------------
md_viz = "## Visualization\nSpectrogram comparison: noisy vs enhanced vs clean, plus predicted mask."

cell10_viz = textwrap.dedent("""\
# ============================================================================
# Cell 10: Spectrogram comparison
# ============================================================================
sample = test_ds[0]
noisy_mag_s = sample['noisy_mag'].unsqueeze(0).to(device)

with torch.no_grad():
    inp_s  = torch.log1p(noisy_mag_s).unsqueeze(1)
    mask_s = model(inp_s)
    enh_mag_s = (mask_s * noisy_mag_s).squeeze(0).cpu()

noisy_spec = sample['noisy_mag'].numpy()
clean_spec = sample['clean_mag'].numpy()
enh_spec   = enh_mag_s.numpy()
mask_np    = mask_s.squeeze(0).cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, spec, title in [
    (axes[0,0], np.log1p(noisy_spec), 'Noisy Input'),
    (axes[0,1], np.log1p(clean_spec), 'Clean Target'),
    (axes[1,0], np.log1p(enh_spec),   'Enhanced (DPT)'),
    (axes[1,1], mask_np,              'Predicted Mask'),
]:
    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Time frame')
    ax.set_ylabel('Frequency bin')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Review 3 DPT: Spectrogram Comparison (Test Sample 0)', fontsize=14)
plt.tight_layout()
plt.savefig('spectrogram_comparison.png', dpi=150)
plt.show()
print('Saved spectrogram_comparison.png')""")

# ------------------------------------------------------------------
cell11_summary = textwrap.dedent("""\
# ============================================================================
# Cell 11: Summary JSON + comparison table
# ============================================================================
summary = {
    'review': 3,
    'model': 'DPTSTFTEnhancer',
    'approach': 'Dual-Path Transformer',
    'pipeline': 'STFT (n_fft=512, hop=256) -> DPT mask -> ISTFT',
    'params': total_params,
    'checkpoint': {'epoch': int(ckpt['epoch']), 'val_loss': float(ckpt['val_loss'])},
    'test_samples': len(test_ds),
    'metrics': {
        'pesq_noisy':        round(avg_pesq_n, 3),
        'pesq_enhanced':     round(avg_pesq_e, 3),
        'stoi_noisy':        round(avg_stoi_n, 4),
        'stoi_enhanced':     round(avg_stoi_e, 4),
        'sisdr_noisy_dB':    round(avg_sisdr_n, 2),
        'sisdr_enhanced_dB': round(avg_sisdr_e, 2),
    },
    'comparison': {
        'R1_CRN_PESQ': 3.10,
        'R2_TransMel_PESQ': 1.141,
        'R3v1_TransSTFT_PESQ': 1.089,
        'R3_DPT_PESQ': round(avg_pesq_e, 3),
        'R3_DPT_SISDR': round(avg_sisdr_e, 2),
    },
    'history': history,
}
with open('review3_dpt_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved review3_dpt_summary.json')

W = 80
print('\\n' + '='*W)
print(f'{"Review":>12} {"Architecture":>20} {"PESQ":>8} {"STOI":>7} {"SI-SDR":>10} {"Params":>10}')
print('='*W)
print(f'{"R1 CRN":>12} {"LSTM+STFT":>20} {"~3.10":>8} {"---":>7} {"---":>10} {"~2.5M":>10}')
print(f'{"R2 Trans":>12} {"CNN+Trans+Mel":>20} {1.141:>8.3f} {0.695:>7.3f} {-25.58:>9.2f}dB {"2.45M":>10}')
print(f'{"R3v1 Trans":>12} {"CNN+Trans+STFT":>20} {1.089:>8.3f} {0.622:>7.3f} {-1.65:>9.2f}dB {"2.45M":>10}')
print(f'{"R3 DPT":>12} {"DualPath+STFT":>20} {avg_pesq_e:>8.3f} {avg_stoi_e:>7.3f} {avg_sisdr_e:>9.2f}dB {total_params/1e6:>8.2f}M')
print('='*W)
noisy_line = f'Noisy baseline: PESQ={avg_pesq_n:.3f}  STOI={avg_stoi_n:.3f}  SI-SDR={avg_sisdr_n:.2f}dB'
dp = avg_pesq_e - avg_pesq_n
ds = avg_stoi_e - avg_stoi_n
dd = avg_sisdr_e - avg_sisdr_n
dpt_line = f'DPT vs noisy: dPESQ={dp:+.3f}  dSTOI={ds:+.4f}  dSI-SDR={dd:+.2f}dB'
print(noisy_line)
print(dpt_line)""")

# ─── Assemble notebook ───────────────────────────────────────────────────

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "id": f"md{len(cells):02d}",
                  "metadata": {"trusted": True}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "id": f"c{len(cells):02d}", "metadata": {"trusted": True},
                  "outputs": [], "source": source})

md(md_title)
code(cell1_imports)
md(md_dataset)
code(cell2_data)
md(md_stft_ds)
code(cell3_dataset)
md(md_model)
code(cell4_model)
code(cell5_utils)
md(md_training)
code(cell6_setup)
code(cell7_train)
md(md_results)
code(cell8_curves)
md(md_eval)
code(cell9_eval)
md(md_viz)
code(cell10_viz)
code(cell11_summary)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_text = json.dumps(nb)
print(f"Cells: {len(cells)}, Notebook text: {len(nb_text):,} chars")

# ─── Kaggle push payload ─────────────────────────────────────────────────

payload = {
    "text": nb_text,
    "language": "python",
    "kernelType": "notebook",
    "enableGpu": True,
    "enableInternet": True,
    "kernelExecutionType": "SaveAndRunAll",
    "datasetDataSources": ["earth16/libri-speech-noise-dataset"],
}

with open("kaggle_r3_dpt_nb.json", "w") as f:
    json.dump(payload, f, indent=2)

print("Written: kaggle_r3_dpt_nb.json")
