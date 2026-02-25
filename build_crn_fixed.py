"""
Build a FIXED CRN baseline notebook:
- STFT-based (not mel) for lossless ISTFT reconstruction
- Real PESQ/STOI/SI-SDR evaluation on actual waveforms
- Proper checkpoint saving
- Clean structure matching our other review notebooks
"""
import json

cells = []

def md(id_, src):
    cells.append({"cell_type": "markdown", "id": id_,
                  "metadata": {"trusted": True}, "source": src})

def code(id_, src):
    cells.append({"cell_type": "code", "execution_count": None, "id": id_,
                  "metadata": {"trusted": True}, "outputs": [], "source": src})

# ── Cell 0: Title ──
md("md00", r"""# CRN Baseline — STFT Speech Enhancement (Fixed)

**What changed from the original CRN baseline:**
1. Switched from **Mel spectrogram** (non-invertible) → **STFT** (lossless ISTFT)
2. Real **PESQ / STOI / SI-SDR** on actual waveforms (no more "estimated" metrics)
3. Proper CRN architecture: CNN encoder → LSTM (per-frequency-bin) → CNN decoder → sigmoid mask
4. Checkpoint saving works correctly

**Architecture:**
```
Input: (B, 1, 257, T) ← log1p(STFT magnitude)
  → CNN Encoder (3 layers, stride-2 on freq): (B, 256, 33, T)
  → Reshape → LSTM(256, hidden=256, 2 layers) across time
  → CNN Decoder + Sigmoid → mask (B, 257, T)
Reconstruction: mask × noisy_mag → ISTFT with noisy phase → waveform
```

**Team:** Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)
**Guide:** Dr. Praveen Jaraut — VIT Bhopal Capstone""")

# ── Cell 1: Imports ──
code("c01", r"""# ============================================================================
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

# STFT config (same as all reviews — consistent comparison)
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

# ── Cell 2: Dataset header ──
md("md02", """## Dataset: LibriSpeech-Noise
Download & extract `earth16/libri-speech-noise-dataset` (7000 train + 105 test WAV pairs).""")

# ── Cell 3: Dataset extraction ──
code("c03", r"""# ============================================================================
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
    print(f'  {tag}: {d} ({n} files)')""")

# ── Cell 4: STFT Dataset ──
md("md04", """## STFT Dataset
Same STFT config as R2/R3 for fair comparison. Returns magnitude, phase, waveforms.""")

code("c05", r"""# ============================================================================
# Cell 3: STFTSpeechDataset
# ============================================================================
class STFTSpeechDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sr=SR, max_len=MAX_LEN):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        assert len(self.noisy_files) == len(self.clean_files), \
            f'Mismatch: {len(self.noisy_files)} noisy vs {len(self.clean_files)} clean'
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

# ── Cell 5: CRN Model ──
md("md06", "## CRN Model (Fixed)\n\n"
   "**Key fixes from original:**\n"
   "1. **STFT input** (257 freq bins) instead of mel (128) — enables lossless ISTFT reconstruction\n"
   "2. **Per-frequency LSTM**: reshapes to `(B*F', T, C)` so LSTM models temporal dynamics "
   "for each frequency sub-band independently — instead of collapsing frequency with `mean(dim=2)`\n"
   "3. **Proper CNN freq downsampling/upsampling** with stride-2 convolutions + transposed convolutions\n\n"
   "```\n"
   "CNN Encoder: (B, 1, 257, T)\n"
   "  -> Conv2d(1->64, stride=(2,1)) -> (B, 64, 129, T)\n"
   "  -> Conv2d(64->128, stride=(2,1)) -> (B, 128, 65, T)\n"
   "  -> Conv2d(128->256, stride=(2,1)) -> (B, 256, 33, T)\n"
   "LSTM: reshape to (B*33, T, 256) -> LSTM(256, 256, 2 layers) -> (B*33, T, 256)\n"
   "CNN Decoder: (B, 256, 33, T)\n"
   "  -> ConvT2d(256->128, stride=(2,1)) -> (B, 128, 65, T)\n"
   "  -> ConvT2d(128->64, stride=(2,1)) -> (B, 64, 129, T)\n"
   "  -> ConvT2d(64->32, stride=(2,1)) -> (B, 32, 257, T)\n"
   "  -> Conv2d(32->1, 1x1) + Sigmoid -> mask (B, 1, 257, T)\n"
   "```")

code("c07", "# ============================================================================\n"
"# Cell 4: CRN Baseline Model (Fixed -- STFT based)\n"
"# ============================================================================\n"
"class CRNBaseline(nn.Module):\n"
'    """CRN for STFT-based speech enhancement."""\n'
"    def __init__(self, n_freq=257):\n"
"        super().__init__()\n"
"        self.n_freq = n_freq\n"
"\n"
"        # CNN Encoder: downsample frequency with stride-2\n"
"        self.enc1 = nn.Sequential(\n"
"            nn.Conv2d(1, 64, kernel_size=3, stride=(2, 1), padding=1),\n"
"            nn.BatchNorm2d(64), nn.ReLU(inplace=True))\n"
"        self.enc2 = nn.Sequential(\n"
"            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),\n"
"            nn.BatchNorm2d(128), nn.ReLU(inplace=True))\n"
"        self.enc3 = nn.Sequential(\n"
"            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),\n"
"            nn.BatchNorm2d(256), nn.ReLU(inplace=True))\n"
"\n"
"        # LSTM: processes each frequency sub-band across time\n"
"        self.lstm = nn.LSTM(\n"
"            input_size=256, hidden_size=256, num_layers=2,\n"
"            batch_first=True, dropout=0.1)\n"
"\n"
"        # CNN Decoder: upsample frequency back with transposed convolutions\n"
"        self.dec3 = nn.Sequential(\n"
"            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),\n"
"            nn.BatchNorm2d(128), nn.ReLU(inplace=True))\n"
"        self.dec2 = nn.Sequential(\n"
"            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),\n"
"            nn.BatchNorm2d(64), nn.ReLU(inplace=True))\n"
"        self.dec1 = nn.Sequential(\n"
"            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),\n"
"            nn.BatchNorm2d(32), nn.ReLU(inplace=True))\n"
"\n"
"        # Final 1x1 conv + sigmoid mask\n"
"        self.mask_conv = nn.Sequential(\n"
"            nn.Conv2d(32, 1, kernel_size=1),\n"
"            nn.Sigmoid())\n"
"\n"
"        self._init_weights()\n"
"\n"
"    def _init_weights(self):\n"
"        for m in self.modules():\n"
"            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):\n"
"                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')\n"
"                if m.bias is not None:\n"
"                    nn.init.zeros_(m.bias)\n"
"            elif isinstance(m, nn.BatchNorm2d):\n"
"                nn.init.ones_(m.weight)\n"
"                nn.init.zeros_(m.bias)\n"
"            elif isinstance(m, nn.LSTM):\n"
"                for name, param in m.named_parameters():\n"
"                    if 'weight' in name:\n"
"                        nn.init.xavier_normal_(param)\n"
"                    elif 'bias' in name:\n"
"                        nn.init.zeros_(param)\n"
"\n"
"    def forward(self, x):\n"
"        # x: (B, 1, n_freq, T)\n"
"        B, _, F_orig, T_orig = x.shape\n"
"\n"
"        # Encode (downsample freq: 257->129->65->33)\n"
"        e1 = self.enc1(x)     # (B, 64, 129, T)\n"
"        e2 = self.enc2(e1)    # (B, 128, 65, T)\n"
"        e3 = self.enc3(e2)    # (B, 256, 33, T)\n"
"\n"
"        # LSTM: per-frequency-bin processing across time\n"
"        B2, C, Fenc, T = e3.shape\n"
"        # Reshape: (B, C, Fenc, T) -> (B*Fenc, T, C)\n"
"        lstm_in = e3.permute(0, 2, 3, 1).reshape(B2 * Fenc, T, C)\n"
"        lstm_out, _ = self.lstm(lstm_in)  # (B*Fenc, T, C)\n"
"        # Reshape back: (B*Fenc, T, C) -> (B, C, Fenc, T)\n"
"        h = lstm_out.reshape(B2, Fenc, T, C).permute(0, 3, 1, 2)  # (B, 256, 33, T)\n"
"\n"
"        # Decode (upsample freq: 33->65->129->257)\n"
"        d3 = self.dec3(h)      # (B, 128, 65, T)\n"
"        d2 = self.dec2(d3)     # (B, 64, 129, T)\n"
"        d1 = self.dec1(d2)     # (B, 32, 257, T)\n"
"\n"
"        # Crop/pad to match original freq dimension\n"
"        if d1.shape[2] != F_orig:\n"
"            d1 = F.interpolate(d1, size=(F_orig, T_orig), mode='bilinear', align_corners=False)\n"
"\n"
"        mask = self.mask_conv(d1).squeeze(1)  # (B, n_freq, T)\n"
"        return mask\n"
"\n"
"\n"
"# Quick test\n"
"model = CRNBaseline(n_freq=N_FREQ).to(device)\n"
"total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
"print(f'CRNBaseline: {total_params:,} params ({total_params/1e6:.2f}M)')\n"
"\n"
"with torch.no_grad():\n"
"    dummy = torch.randn(2, 1, N_FREQ, 188).to(device)\n"
"    mask = model(dummy)\n"
"    print(f'Input: {dummy.shape} -> Mask: {mask.shape}')\n"
"    assert mask.shape == (2, N_FREQ, 188), f'Shape mismatch: {mask.shape}'\n"
"    assert mask.min().item() >= 0 and mask.max().item() <= 1\n"
"    print(f'Mask range: [{mask.min().item():.4f}, {mask.max().item():.4f}]')\n"
"    print('Forward pass OK')\n"
"\n"
"# Architecture breakdown\n"
"enc_p = sum(p.numel() for n, p in model.named_parameters() if 'enc' in n)\n"
"lstm_p = sum(p.numel() for n, p in model.named_parameters() if 'lstm' in n)\n"
"dec_p = sum(p.numel() for n, p in model.named_parameters() if 'dec' in n or 'mask_conv' in n)\n"
"print(f'  Encoder:  {enc_p:>10,} ({enc_p/total_params*100:.1f}%)')\n"
"print(f'  LSTM:     {lstm_p:>10,} ({lstm_p/total_params*100:.1f}%)')\n"
"print(f'  Decoder:  {dec_p:>10,} ({dec_p/total_params*100:.1f}%)')")

# ── Cell 6: SI-SDR utility ──
code("c08", r"""# ============================================================================
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

# ── Cell 7: Training header ──
md("md09", """## Training
L1 loss on log-magnitude, Adam lr=1e-3 with ReduceLROnPlateau.
Batch=16, 25 epochs, patience=10.""")

# ── Cell 8: Training setup ──
code("c10", r"""# ============================================================================
# Cell 6: Training setup
# ============================================================================
MAX_EPOCHS    = 25
LR            = 1e-3
BATCH         = 16
PATIENCE      = 10
CKPT          = 'crn_baseline_best.pth'

# Re-init model fresh
model = CRNBaseline(n_freq=N_FREQ).to(device)
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

print(f'Train:{n_train} Val:{n_val} Test:{len(test_ds)} | BS={BATCH} LR={LR}')""")

# ── Cell 9: Training loop ──
code("c11", r"""# ============================================================================
# Cell 7: Training loop
# ============================================================================
history = {'train_loss': [], 'val_loss': []}
best_val = float('inf')
patience_ctr = 0
t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
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

# ── Cell 10: Training curves ──
md("md12", "## Results")

code("c13", r"""# ============================================================================
# Cell 8: Training curves
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
eps = range(1, len(history['train_loss']) + 1)
ax.plot(eps, history['train_loss'], 'b-o', label='Train', ms=3)
ax.plot(eps, history['val_loss'], 'r-s', label='Val', ms=3)
ax.axvline(best_ep, color='g', ls='--', alpha=0.7, label=f'Best (ep{best_ep})')
ax.set_xlabel('Epoch')
ax.set_ylabel('L1 Loss (log-magnitude)')
ax.set_title('CRN Baseline (Fixed STFT) — Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print('Saved training_curves.png')""")

# ── Cell 11: Eval header ──
md("md14", """## Evaluation
**REAL** PESQ / STOI / SI-SDR on 105 test samples via ISTFT waveform reconstruction.
No more estimated metrics!""")

# ── Cell 12: Evaluation ──
code("c15", r"""# ============================================================================
# Cell 9: Evaluation — REAL PESQ / STOI / SI-SDR
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

print(f'\nResults on {len(test_ds)} test files:')
print(f'  PESQ  : noisy={avg_pesq_n:.3f}  enhanced={avg_pesq_e:.3f}  d={avg_pesq_e-avg_pesq_n:+.3f}')
print(f'  STOI  : noisy={avg_stoi_n:.3f}  enhanced={avg_stoi_e:.3f}  d={avg_stoi_e-avg_stoi_n:+.4f}')
print(f'  SI-SDR: noisy={avg_sisdr_n:.2f}dB  enh={avg_sisdr_e:.2f}dB  d={avg_sisdr_e-avg_sisdr_n:+.2f}dB')""")

# ── Cell 13: Spectrogram visualization ──
md("md16", """## Visualization
Spectrogram comparison: noisy vs enhanced vs clean, plus predicted mask.""")

code("c17", r"""# ============================================================================
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
    (axes[1,0], np.log1p(enh_spec),   'Enhanced (CRN)'),
    (axes[1,1], mask_np,              'Predicted Mask'),
]:
    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Time frame')
    ax.set_ylabel('Frequency bin')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('CRN Baseline: Spectrogram Comparison (Test Sample 0)', fontsize=14)
plt.tight_layout()
plt.savefig('spectrogram_comparison.png', dpi=150)
plt.show()
print('Saved spectrogram_comparison.png')""")

# ── Cell 14: Summary JSON ──
code("c18", r"""# ============================================================================
# Cell 11: Summary JSON + comparison table
# ============================================================================
summary = {
    'review': 'R1_CRN_Fixed',
    'model': 'CRNBaseline',
    'approach': 'Conv-Recurrent Network (STFT-based)',
    'pipeline': 'STFT (n_fft=512, hop=256) -> CRN mask -> ISTFT',
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
with open('crn_baseline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved crn_baseline_summary.json')

W = 80
print('\n' + '='*W)
print(f'{"Model":>15} {"Pipeline":>20} {"PESQ":>8} {"STOI":>7} {"SI-SDR":>10} {"Params":>10}')
print('='*W)
print(f'{"Noisy":>15} {"---":>20} {avg_pesq_n:>8.3f} {avg_stoi_n:>7.3f} {avg_sisdr_n:>9.2f}dB {"---":>10}')
print(f'{"CRN (fixed)":>15} {"LSTM+STFT":>20} {avg_pesq_e:>8.3f} {avg_stoi_e:>7.3f} {avg_sisdr_e:>9.2f}dB {total_params/1e6:>8.2f}M')
print('='*W)
dp = avg_pesq_e - avg_pesq_n
ds = avg_stoi_e - avg_stoi_n
dd = avg_sisdr_e - avg_sisdr_n
print(f'CRN vs noisy: dPESQ={dp:+.3f}  dSTOI={ds:+.4f}  dSI-SDR={dd:+.2f}dB')""")

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

text = json.dumps(nb, ensure_ascii=False)
print(f"Cells: {len(cells)}")
print(f"Notebook text: {len(text):,} chars")

with open("kaggle_crn_fixed_nb.json", "w", encoding="utf-8") as f:
    json.dump({"text": text}, f)
print("Written: kaggle_crn_fixed_nb.json")

# Also write raw text for push
with open("crn_nb_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"Written: crn_nb_text.txt ({len(text):,} chars)")
