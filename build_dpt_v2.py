"""
Build DPT-v2 notebook with the CRITICAL crop alignment fix.

Same fix as CRN v2 — __getitem__ uses ONE shared random crop for both noisy and clean.
Architecture: Dual-Path Transformer (Shallow Transformer for capstone title).
"""
import json, textwrap

cells = []

def md(src):
    cells.append({"cell_type": "markdown",
                  "id": f"md{len(cells):02d}",
                  "metadata": {"trusted": True},
                  "source": src})

def code(src):
    cells.append({"cell_type": "code",
                  "execution_count": None,
                  "id": f"c{len(cells):02d}",
                  "metadata": {"trusted": True},
                  "outputs": [],
                  "source": src})

# ═══════════════════════════════════════════════════════════════════════════
# Cell 0: Title
# ═══════════════════════════════════════════════════════════════════════════
md(r"""# Review 3: Dual-Path Transformer v2 — Aligned-Crop STFT Speech Enhancement

## Critical Bug Fix from v1
**v1 had a fatal data-loading bug:** `_load_fix()` was called independently for noisy and clean files —
each call generated its OWN random crop position.  Since WAV files are 16–24 seconds long and we crop
3-second segments, the noisy and clean waveforms came from **completely different time positions** in the
utterance.  The model was training on mismatched (noisy segment A, clean segment B) pairs.

**Evidence:** v1 noisy baseline had STOI = 0.218 and SI-SDR = −43.34 dB — indicating the "pairs" were
essentially unrelated audio.

**Fix:** `__getitem__` now computes **one** random start position and applies it to **both** files.
For test evaluation, `start = 0` (deterministic) so metrics are fully reproducible.

**Architecture:** Dual-Path Transformer (DPT) — "Lightweight Speech Enhancement Using Shallow Transformers"
```
CNN Encoder (stride-2 on freq): (B, 1, 257, T) → (B, 128, 65, T)
  → DualPathBlock ×2:
      FreqTransformer: attend across 65 freq bins per time step
      TimeTransformer: attend across T time steps per freq bin
  → Skip + Interpolate → (B, 128, 257, T)
  → CNN Decoder + Sigmoid → mask (B, 257, T)
Reconstruction: mask × noisy_mag → ISTFT with noisy phase → waveform
```

**Team:** Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)
**Guide:** Dr. Praveen Jaraut — VIT Bhopal Capstone""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 1: Imports + Config
# ═══════════════════════════════════════════════════════════════════════════
code(r"""# ============================================================================
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
import glob, os, json, time, warnings, textwrap
warnings.filterwarnings('ignore')
from pesq import pesq as pesq_metric
from pystoi import stoi as stoi_metric

torch.manual_seed(42)
np.random.seed(42)

# STFT config
N_FFT      = 512
HOP_LENGTH = 256
N_FREQ     = N_FFT // 2 + 1   # 257
SR         = 16000
MAX_LEN    = 48000             # 3 s crop from 16-24 s files

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram/1e9:.1f}GB')
print(f'STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, freq={N_FREQ}')
print(f'MAX_LEN={MAX_LEN} ({MAX_LEN/SR:.1f}s crop)')
print('Imports OK')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 2: Dataset download
# ═══════════════════════════════════════════════════════════════════════════
md("## Dataset: LibriSpeech-Noise\n"
   "Download & extract `earth16/libri-speech-noise-dataset` (7000 train + 105 test WAV pairs).\n\n"
   "**Files are 16–24 seconds** at 16 kHz.  We crop 3-second aligned segments for training.")

code(r"""# ============================================================================
# Cell 2: Dataset download & extraction
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
    print(f'  {tag}: {d} ({n} files)')

# Sanity: check file durations (use torchaudio.load — .info() removed in newer versions)
sample_files = sorted(glob.glob(os.path.join(noisy_test, '*.wav')))[:3]
for fp in sample_files:
    wav, sr = torchaudio.load(fp)
    num_frames = wav.shape[-1]
    dur = num_frames / sr
    print(f'  Sample: {os.path.basename(fp)} -> {dur:.2f}s ({num_frames} frames, sr={sr})')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 3: STFT Dataset (FIXED aligned crops) — IDENTICAL to CRN v2
# ═══════════════════════════════════════════════════════════════════════════
md("## STFT Dataset (ALIGNED Crops — CRITICAL FIX)\n\n"
   "**v1 bug:** `_load_fix()` called independently → noisy from time A, clean from time B.\n\n"
   "**v2 fix:** `__getitem__` loads BOTH, picks ONE random start, crops BOTH at same position.\n"
   "Test mode uses `start=0` for deterministic evaluation.")

code(r"""# ============================================================================
# Cell 3: STFTSpeechDataset — ALIGNED crops (v2 fix)
# ============================================================================
class STFTSpeechDataset(Dataset):
    '''STFT dataset with ALIGNED random crops for noisy/clean pairs.

    CRITICAL FIX: v1 generated independent random crops for noisy and clean,
    meaning the model trained on mismatched audio segments.  Now we use ONE
    shared crop position for both files.
    '''
    def __init__(self, noisy_dir, clean_dir, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sr=SR, max_len=MAX_LEN, test_mode=False):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        assert len(self.noisy_files) == len(self.clean_files), \
            f'Mismatch: {len(self.noisy_files)} noisy vs {len(self.clean_files)} clean'
        for nf, cf in zip(self.noisy_files[:3], self.clean_files[:3]):
            assert os.path.basename(nf) == os.path.basename(cf), \
                f'Name mismatch: {os.path.basename(nf)} vs {os.path.basename(cf)}'
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.max_len = max_len
        self.test_mode = test_mode
        self.window = torch.hann_window(n_fft)

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_wav, sr_n = torchaudio.load(self.noisy_files[idx])
        clean_wav, sr_c = torchaudio.load(self.clean_files[idx])
        noisy_wav = noisy_wav[0]
        clean_wav = clean_wav[0]

        if sr_n != self.sr:
            noisy_wav = torchaudio.functional.resample(noisy_wav, sr_n, self.sr)
        if sr_c != self.sr:
            clean_wav = torchaudio.functional.resample(clean_wav, sr_c, self.sr)

        min_len = min(noisy_wav.shape[0], clean_wav.shape[0])
        noisy_wav = noisy_wav[:min_len]
        clean_wav = clean_wav[:min_len]

        # ── CRITICAL FIX: ONE shared crop for both ──
        if min_len > self.max_len:
            if self.test_mode:
                start = 0
            else:
                start = torch.randint(0, min_len - self.max_len, (1,)).item()
            noisy_wav = noisy_wav[start:start + self.max_len]
            clean_wav = clean_wav[start:start + self.max_len]
        elif min_len < self.max_len:
            pad = self.max_len - min_len
            noisy_wav = F.pad(noisy_wav, (0, pad))
            clean_wav = F.pad(clean_wav, (0, pad))

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

# Quick sanity check
_ds = STFTSpeechDataset(noisy_test, clean_test, test_mode=True)
_s = _ds[0]
print(f'STFTSpeechDataset v2 (aligned crops)')
print(f'  noisy_mag:  {_s["noisy_mag"].shape}')
print(f'  clean_mag:  {_s["clean_mag"].shape}')
print(f'  noisy_wav:  {_s["noisy_wav"].shape}')

_corr = torch.corrcoef(torch.stack([_s['noisy_wav'], _s['clean_wav']]))[0,1].item()
print(f'  noisy-clean correlation: {_corr:.4f}  (should be > 0.3 if aligned)')
assert _corr > 0.1, f'Correlation too low ({_corr:.4f}) — crops may still be misaligned!'
print('  ALIGNMENT CHECK PASSED')
del _ds, _s""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 4: DPT Model
# ═══════════════════════════════════════════════════════════════════════════
md("## Model: Dual-Path Transformer (DPT)\n\n"
   "Same architecture as v1 — the bug was in data loading, not the model.\n\n"
   "**Core innovation:** alternating frequency and time transformer blocks give\n"
   "the model full spectral resolution for frequency-selective noise masking.\n\n"
   "```\n"
   "CNN Encoder (stride-2 on freq): (B,1,257,T) → (B,128,65,T)\n"
   "  → DualPathBlock #1: FreqTransformer(65,128) + TimeTransformer(T,128)\n"
   "  → DualPathBlock #2: FreqTransformer(65,128) + TimeTransformer(T,128)\n"
   "  → Skip + Interpolate → (B,128,257,T)\n"
   "  → CNN Decoder + Sigmoid → mask (B,257,T)\n"
   "```")

code(r"""# ============================================================================
# Cell 4: Dual-Path Transformer Model (same architecture as v1)
# ============================================================================
class DualPathBlock(nn.Module):
    '''One dual-path block: frequency-transformer + time-transformer.'''
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.freq_transformer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.time_transformer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)

    def forward(self, x):
        B, C, Fr, T = x.shape
        # Frequency path: attend across freq bins for each time step
        x_f = x.permute(0, 3, 2, 1).reshape(B * T, Fr, C)
        x_f = self.freq_transformer(x_f)
        x = x_f.reshape(B, T, Fr, C).permute(0, 3, 2, 1)
        # Time path: attend across time for each freq bin
        x_t = x.permute(0, 2, 3, 1).reshape(B * Fr, T, C)
        x_t = self.time_transformer(x_t)
        x = x_t.reshape(B, Fr, T, C).permute(0, 3, 1, 2)
        return x


class DPTSTFTEnhancer(nn.Module):
    '''Dual-Path Transformer for STFT-based Speech Enhancement.'''
    def __init__(self, n_freq=257, d_model=128, nhead=4, num_dp_blocks=2,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        self.n_freq = n_freq
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, d_model, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True),
        )
        self.dp_blocks = nn.ModuleList([
            DualPathBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_dp_blocks)
        ])
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
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B, _, F_orig, T_orig = x.shape
        h = self.encoder(x)
        skip = h
        for block in self.dp_blocks:
            h = block(h)
        h = h + skip
        h = F.interpolate(h, size=(F_orig, T_orig),
                          mode='bilinear', align_corners=False)
        return self.decoder(h).squeeze(1)


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

enc_p = sum(p.numel() for p in model.encoder.parameters())
dp_p  = sum(p.numel() for p in model.dp_blocks.parameters())
dec_p = sum(p.numel() for p in model.decoder.parameters())
print(f'  Encoder:     {enc_p:>8,} ({enc_p/total_params*100:.1f}%)')
print(f'  DPT blocks:  {dp_p:>8,} ({dp_p/total_params*100:.1f}%)')
print(f'  Decoder:     {dec_p:>8,} ({dec_p/total_params*100:.1f}%)')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 5: SI-SDR utility
# ═══════════════════════════════════════════════════════════════════════════
code(r"""# ============================================================================
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

# ═══════════════════════════════════════════════════════════════════════════
# Cell 6: Training setup
# ═══════════════════════════════════════════════════════════════════════════
md("## Training\n"
   "L1 loss on log-magnitude. Adam lr=1e-3 with 3-epoch warmup, then ReduceLROnPlateau.\n"
   "**Now with aligned crops** — model sees matched (noisy, clean) pairs for the first time!")

code(r"""# ============================================================================
# Cell 6: Training setup
# ============================================================================
MAX_EPOCHS    = 30
LR            = 1e-3
BATCH         = 8
PATIENCE      = 12
WARMUP_EPOCHS = 3
CKPT          = 'dpt_v2_best.pth'

model = DPTSTFTEnhancer(n_freq=N_FREQ).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total_params:,}')

# Training dataset: test_mode=False (random aligned crops)
full_train = STFTSpeechDataset(noisy_train, clean_train, test_mode=False)
n_val   = int(0.1 * len(full_train))
n_train = len(full_train) - n_val
train_ds, val_ds = torch.utils.data.random_split(
    full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

# Test dataset: test_mode=True (deterministic crop at start=0)
test_ds = STFTSpeechDataset(noisy_test, clean_test, test_mode=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.5, patience=5)

print(f'Train:{n_train} Val:{n_val} Test:{len(test_ds)} | BS={BATCH} LR={LR}')
print(f'Warmup: {WARMUP_EPOCHS} ep, then ReduceLROnPlateau(patience=5, factor=0.5)')
print(f'Train crops: RANDOM aligned | Test crops: DETERMINISTIC (start=0)')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 7: Training loop
# ═══════════════════════════════════════════════════════════════════════════
code(r"""# ============================================================================
# Cell 7: Training loop with warmup
# ============================================================================
history = {'train_loss': [], 'val_loss': []}
best_val = float('inf')
patience_ctr = 0
t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    # LR warmup
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
        inp  = torch.log1p(noisy_mag).unsqueeze(1)
        mask = model(inp)
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
print(f'\nDONE best_ep={best_ep} best_val={best_val:.4f} time={time.time()-t0:.0f}s')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 8: Training curves
# ═══════════════════════════════════════════════════════════════════════════
md("## Results")

code(r"""# ============================================================================
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
ax.set_title('DPT v2 (Aligned Crops) — Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print('Saved training_curves.png')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 9: Evaluation
# ═══════════════════════════════════════════════════════════════════════════
md("## Evaluation\n"
   "PESQ / STOI / SI-SDR on 105 test samples.\n"
   "**Deterministic:** test_mode=True → crop at start=0 → same segment every run.")

code(r"""# ============================================================================
# Cell 9: Evaluation — PESQ / STOI / SI-SDR
# ============================================================================
torch.manual_seed(42)

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

    if i < 3:
        print(f'  [{i}] PESQ: {pesq_noisy_list[-1]:.3f}->{pesq_enh_list[-1]:.3f}  '
              f'STOI: {stoi_noisy_list[-1]:.3f}->{stoi_enh_list[-1]:.3f}  '
              f'SI-SDR: {sisdr_noisy_list[-1]:.2f}->{sisdr_enh_list[-1]:.2f}dB')

avg = lambda lst: float(np.mean(lst)) if lst else 0.0
avg_pesq_n,  avg_pesq_e  = avg(pesq_noisy_list),  avg(pesq_enh_list)
avg_stoi_n,  avg_stoi_e  = avg(stoi_noisy_list),  avg(stoi_enh_list)
avg_sisdr_n, avg_sisdr_e = avg(sisdr_noisy_list), avg(sisdr_enh_list)

print(f'\n{"="*70}')
print(f'Results on {len(test_ds)} test files (ALIGNED deterministic crops):')
print(f'  PESQ  : noisy={avg_pesq_n:.3f}  enhanced={avg_pesq_e:.3f}  Δ={avg_pesq_e-avg_pesq_n:+.3f}')
print(f'  STOI  : noisy={avg_stoi_n:.3f}  enhanced={avg_stoi_e:.3f}  Δ={avg_stoi_e-avg_stoi_n:+.4f}')
print(f'  SI-SDR: noisy={avg_sisdr_n:.2f}dB  enh={avg_sisdr_e:.2f}dB  Δ={avg_sisdr_e-avg_sisdr_n:+.2f}dB')
print(f'{"="*70}')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 10: Spectrogram visualization
# ═══════════════════════════════════════════════════════════════════════════
md("## Visualization\nSpectrogram comparison: noisy vs enhanced vs clean, plus predicted mask.")

code(r"""# ============================================================================
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
    (axes[1,0], np.log1p(enh_spec),   'Enhanced (DPT v2)'),
    (axes[1,1], mask_np,              'Predicted Mask'),
]:
    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Time frame')
    ax.set_ylabel('Frequency bin')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('DPT v2 (Aligned Crops): Spectrogram Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('spectrogram_comparison.png', dpi=150)
plt.show()
print('Saved spectrogram_comparison.png')""")

# ═══════════════════════════════════════════════════════════════════════════
# Cell 11: Summary JSON
# ═══════════════════════════════════════════════════════════════════════════
code(r"""# ============================================================================
# Cell 11: Summary JSON + comparison table
# ============================================================================
summary = {
    'model': 'DPT_v2_AlignedCrops',
    'architecture': 'Dual-Path Transformer (Shallow Transformer)',
    'critical_fix': 'Aligned random crops for noisy/clean pairs',
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
    'history': history,
}
with open('dpt_v2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved dpt_v2_summary.json')

W = 70
print(f'\n{"="*W}')
print(f'{"Metric":>10} {"Noisy":>12} {"DPT v2":>12} {"Delta":>12}')
print(f'{"="*W}')
print(f'{"PESQ":>10} {avg_pesq_n:>12.3f} {avg_pesq_e:>12.3f} {avg_pesq_e-avg_pesq_n:>+12.3f}')
print(f'{"STOI":>10} {avg_stoi_n:>12.3f} {avg_stoi_e:>12.3f} {avg_stoi_e-avg_stoi_n:>+12.4f}')
print(f'{"SI-SDR":>10} {avg_sisdr_n:>11.2f}dB {avg_sisdr_e:>11.2f}dB {avg_sisdr_e-avg_sisdr_n:>+11.2f}dB')
print(f'{"="*W}')
print(f'\nv1 (misaligned) noisy baseline was: STOI=0.218, SI-SDR=-43.34dB (broken)')
print(f'v2 (aligned) noisy baseline is: STOI={avg_stoi_n:.3f}, SI-SDR={avg_sisdr_n:.2f}dB (correct)')""")

# ═══════════════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_text = json.dumps(nb, ensure_ascii=False)
print(f"Cells: {len(cells)}")
print(f"Notebook text: {len(nb_text):,} chars")

with open("dpt_v2_nb_text.txt", "w", encoding="utf-8") as f:
    f.write(nb_text)
print(f"Written: dpt_v2_nb_text.txt")
