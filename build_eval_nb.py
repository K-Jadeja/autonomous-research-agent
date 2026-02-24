"""
Build eval-only notebook for Review 3.
- Mounts v3 checkpoint via kernelDataSources
- Mounts dataset via datasetDataSources (no 7z extraction needed for mounted data)
- Skips all training
- Runs: eval (PESQ/STOI/SI-SDR) + attention viz + summary JSON
"""
import json, textwrap

# ── CELLS ───────────────────────────────────────────────────────────────────

cells = []

def md(src): cells.append({"cell_type":"markdown","id":f"md{len(cells):02d}","metadata":{"trusted":True},"source":src})
def code(src): cells.append({"cell_type":"code","execution_count":None,"id":f"c{len(cells):02d}","metadata":{"trusted":True},"outputs":[],"source":textwrap.dedent(src).strip()})

# ── 0: header ────────────────────────────────────────────────────────────────
md("""# Review 3 — Eval Only: STFT-Transformer Speech Enhancement

**Purpose:** Load trained checkpoint from Review 3 training run and compute metrics.  
**Skips:** All training. Loads `stft_transformer_best.pth` from kernel output.  
**Metrics:** PESQ, STOI, SI-SDR on 105 test pairs + attention visualization.

**Team:** Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)  
**Guide:** Dr. Praveen Jaraut — VIT Bhopal Capstone""")

# ── 1: install / imports ─────────────────────────────────────────────────────
code("""
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
import glob, os, json, warnings
warnings.filterwarnings('ignore')
from pesq import pesq as pesq_metric
from pystoi import stoi as stoi_metric

torch.manual_seed(42)
np.random.seed(42)

# STFT config (must match training)
N_FFT      = 512
HOP_LENGTH = 256
N_FREQ     = N_FFT // 2 + 1   # 257
SR         = 16000
MAX_LEN    = 48000             # 3 s

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    props = torch.cuda.get_device_properties(0)
    vram  = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram/1e9:.1f} GB')
print(f'STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, freq_bins={N_FREQ}')

# Checkpoint: try mounted input first, fall back to downloading kernel output
CKPT_DIR_MOUNTED = '/kaggle/input/review-3-stft-transformer-speech-enhancement'
CKPT_DOWNLOAD    = '/kaggle/working/ckpt_dl'
CKPT_PATH = os.path.join(CKPT_DIR_MOUNTED, 'stft_transformer_best.pth')

if os.path.exists(CKPT_PATH):
    print(f'Checkpoint found (mounted): {CKPT_PATH}')
    print('Files in input dir:', sorted(os.listdir(CKPT_DIR_MOUNTED)))
else:
    print('Mounted checkpoint not found — downloading via kaggle kernels output ...')
    import subprocess
    os.makedirs(CKPT_DOWNLOAD, exist_ok=True)
    r = subprocess.run(
        ['kaggle', 'kernels', 'output',
         'kjadeja/review-3-stft-transformer-speech-enhancement',
         '-p', CKPT_DOWNLOAD],
        capture_output=True, text=True)
    print(r.stdout[-500:] if r.stdout else '(no stdout)')
    print(r.stderr[-300:] if r.stderr else '')
    # Find the checkpoint in downloaded files
    candidates = [os.path.join(CKPT_DOWNLOAD, f)
                  for f in os.listdir(CKPT_DOWNLOAD) if 'best' in f and f.endswith('.pth')]
    if not candidates:
        candidates = [os.path.join(CKPT_DOWNLOAD, f)
                      for f in os.listdir(CKPT_DOWNLOAD) if f.endswith('.pth')]
    print('Downloaded files:', os.listdir(CKPT_DOWNLOAD))
    assert candidates, f'No .pth files found in {CKPT_DOWNLOAD}'
    CKPT_PATH = candidates[0]
    print(f'Using checkpoint: {CKPT_PATH}')

print(f'CKPT_PATH = {CKPT_PATH}')
print(f'Exists: {os.path.exists(CKPT_PATH)}')
""")

# ── 2: test data ─────────────────────────────────────────────────────────────
md("## Test Data Setup\nExtracts test/y_test splits from the mounted LibriSpeech-Noise dataset.")

code("""
# ============================================================================
# Cell 2: Extract test data from mounted dataset
# ============================================================================
import subprocess, zipfile

DATA_BASE  = '/kaggle/working/data'
os.makedirs(DATA_BASE, exist_ok=True)
done_flag  = os.path.join(DATA_BASE, '.done_test')

if os.path.exists(done_flag):
    print('Test data already extracted, skipping')
else:
    # Install p7zip
    subprocess.run(['apt-get', 'install', '-y', 'p7zip-full'], capture_output=True)

    # Try mounted dataset first
    ds_mounted = '/kaggle/input/libri-speech-noise-dataset'
    dl_tmp     = '/kaggle/working/dl_tmp'
    os.makedirs(dl_tmp, exist_ok=True)

    if os.path.isdir(ds_mounted) and len(os.listdir(ds_mounted)) > 0:
        src = ds_mounted
        print(f'Using mounted dataset at {src}')
    else:
        print('Dataset not mounted — downloading via kaggle API ...')
        subprocess.run(['kaggle', 'datasets', 'download',
                        'earth16/libri-speech-noise-dataset', '-p', dl_tmp], check=True)
        zf = os.path.join(dl_tmp, 'libri-speech-noise-dataset.zip')
        if os.path.exists(zf):
            with zipfile.ZipFile(zf) as z:
                z.extractall(dl_tmp)
            os.remove(zf)
        src = dl_tmp

    # Extract ONLY test archives (skip 6 GB train data)
    for arch in ['test.7z', 'y_test.7z']:
        fp = os.path.join(src, arch)
        if os.path.exists(fp):
            print(f'Extracting {arch} ...')
            r = subprocess.run(['7z', 'x', fp, f'-o{DATA_BASE}', '-y'], capture_output=True)
            print(r.stdout.decode()[-300:] if r.stdout else '(no stdout)')
        else:
            print(f'WARNING: {fp} not found')

    open(done_flag, 'w').close()
    print('Extraction complete')

# Locate directories
def find_wav_dir(base, name):
    for root, dirs, files in os.walk(base):
        if os.path.basename(root) == name and any(f.endswith('.wav') for f in files):
            return root
    return None

noisy_test = find_wav_dir(DATA_BASE, 'test')
clean_test = find_wav_dir(DATA_BASE, 'y_test')
for tag, d in [('noisy_test', noisy_test), ('clean_test', clean_test)]:
    n = len(glob.glob(os.path.join(d, '*.wav'))) if d else 0
    print(f'  {tag}: {d} ({n} files)')
""")

# ── 3: dataset class ─────────────────────────────────────────────────────────
code("""
# ============================================================================
# Cell 3: STFTSpeechDataset
# ============================================================================
class STFTSpeechDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sr=SR, max_len=MAX_LEN):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        assert len(self.noisy_files) == len(self.clean_files)
        self.n_fft = n_fft; self.hop_length = hop_length
        self.sr = sr; self.max_len = max_len
        self.window = torch.hann_window(n_fft)

    def __len__(self): return len(self.noisy_files)

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

test_ds = STFTSpeechDataset(noisy_test, clean_test)
print(f'Test samples: {len(test_ds)}')
""")

# ── 4: model ─────────────────────────────────────────────────────────────────
code("""
# ============================================================================
# Cell 4: STFTTransformerEnhancer (must match training architecture exactly)
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])

class STFTTransformerEnhancer(nn.Module):
    def __init__(self, n_freq=257, d_model=256, nhead=4, num_layers=2,
                 dim_ff=1024, dropout=0.1):
        super().__init__()
        self.n_freq = n_freq
        self.encoder   = nn.Sequential(ConvBlock(1,64), ConvBlock(64,128), ConvBlock(128,256))
        self.pre_proj  = nn.Linear(256, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.post_proj = nn.Linear(d_model, 256)
        self.decoder   = nn.Sequential(
            ConvBlock(256,128), ConvBlock(128,64),
            nn.Conv2d(64,1,3,1,1), nn.Sigmoid())

    def forward(self, x):
        enc  = self.encoder(x)                                    # (B,256,F,T)
        feat = enc.mean(dim=2).permute(0,2,1)                    # (B,T,256)
        feat = self.pos_enc(self.pre_proj(feat))
        feat = self.post_proj(self.transformer(feat))             # (B,T,256)
        feat = feat.permute(0,2,1).unsqueeze(2)                  # (B,256,1,T)
        feat = feat.expand(-1,-1,self.n_freq,-1)                 # (B,256,F,T)
        return self.decoder(feat).squeeze(1)                     # (B,F,T)

model = STFTTransformerEnhancer(n_freq=N_FREQ).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model: {total_params:,} params ({total_params/1e6:.2f}M)')
""")

# ── 5: utilities ──────────────────────────────────────────────────────────────
code("""
# ============================================================================
# Cell 5: SI-SDR + attention weight utilities
# ============================================================================
def si_sdr(estimate, reference):
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    dot = torch.sum(ref * est)
    s_target = dot * ref / (torch.sum(ref**2) + 1e-8)
    e_noise   = est - s_target
    return 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + 1e-8) + 1e-8)

def get_attention_weights(mdl, x):
    mdl.eval()
    weights = []
    with torch.no_grad():
        enc  = mdl.encoder(x)
        feat = enc.mean(dim=2).permute(0,2,1)
        feat = mdl.pos_enc(mdl.pre_proj(feat))
        for layer in mdl.transformer.layers:
            normed = layer.norm1(feat)
            _, w   = layer.self_attn(normed, normed, normed,
                                     need_weights=True, average_attn_weights=False)
            weights.append(w.cpu())
            feat = feat + layer.dropout1(
                layer.self_attn(normed, normed, normed, need_weights=False)[0])
            normed2 = layer.norm2(feat)
            feat = feat + layer.linear2(
                F.dropout(layer.activation(layer.linear1(normed2)), p=0.0, training=False))
    return weights

print('Utilities defined')
""")

# ── 6: load checkpoint + eval ─────────────────────────────────────────────────
md("## Evaluation\nLoad best checkpoint → compute PESQ / STOI / SI-SDR on all 105 test pairs.")

code("""
# ============================================================================
# Cell 6: Load checkpoint + full evaluation
# ============================================================================
assert os.path.exists(CKPT_PATH), f'Checkpoint not found: {CKPT_PATH}'
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'Loaded checkpoint: epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f}')

window_gpu = torch.hann_window(N_FFT).to(device)

pesq_noisy_list, pesq_enh_list  = [], []
stoi_noisy_list, stoi_enh_list  = [], []
sisdr_noisy_list, sisdr_enh_list = [], []

for i in tqdm(range(len(test_ds)), desc='Eval'):
    s           = test_ds[i]
    noisy_mag   = s['noisy_mag'].unsqueeze(0).to(device)    # (1,257,T)
    noisy_phase = s['noisy_phase'].unsqueeze(0).to(device)  # (1,257,T)
    clean_np    = s['clean_wav'].numpy()
    noisy_np    = s['noisy_wav'].numpy()

    with torch.no_grad():
        inp          = torch.log1p(noisy_mag).unsqueeze(1)  # (1,1,257,T)
        mask         = model(inp)                            # (1,257,T)
        enh_mag      = (mask * noisy_mag).squeeze(0)        # (257,T)

    enh_stft = enh_mag * torch.exp(1j * noisy_phase.squeeze(0))
    enh_wav  = torch.istft(enh_stft, N_FFT, HOP_LENGTH, window=window_gpu, length=MAX_LEN)
    enh_np   = enh_wav.cpu().numpy()

    try:
        pesq_noisy_list.append(pesq_metric(SR, clean_np, noisy_np, 'wb'))
        pesq_enh_list.append(  pesq_metric(SR, clean_np, enh_np,   'wb'))
    except Exception as e:
        print(f'  PESQ error sample {i}: {e}')

    stoi_noisy_list.append(stoi_metric(clean_np, noisy_np, SR, extended=False))
    stoi_enh_list.append(  stoi_metric(clean_np, enh_np,   SR, extended=False))

    c_t = torch.from_numpy(clean_np).float()
    n_t = torch.from_numpy(noisy_np).float()
    e_t = torch.from_numpy(enh_np).float()
    sisdr_noisy_list.append(si_sdr(n_t, c_t).item())
    sisdr_enh_list.append(  si_sdr(e_t, c_t).item())

avg_pesq_n  = float(np.mean(pesq_noisy_list))  if pesq_noisy_list  else 0.0
avg_pesq_e  = float(np.mean(pesq_enh_list))    if pesq_enh_list    else 0.0
avg_stoi_n  = float(np.mean(stoi_noisy_list))
avg_stoi_e  = float(np.mean(stoi_enh_list))
avg_sisdr_n = float(np.mean(sisdr_noisy_list))
avg_sisdr_e = float(np.mean(sisdr_enh_list))

print(f'\\nResults on {len(test_ds)} test files:')
print(f'  PESQ  : noisy={avg_pesq_n:.3f}  enhanced={avg_pesq_e:.3f}  Δ={avg_pesq_e-avg_pesq_n:+.3f}')
print(f'  STOI  : noisy={avg_stoi_n:.3f}  enhanced={avg_stoi_e:.3f}  Δ={avg_stoi_e-avg_stoi_n:+.4f}')
print(f'  SI-SDR: noisy={avg_sisdr_n:.2f}dB  enhanced={avg_sisdr_e:.2f}dB  Δ={avg_sisdr_e-avg_sisdr_n:+.2f}dB')
print(f'\\n--- Review comparison ---')
print(f'  R1 CRN       (estimated) : PESQ ~ 3.10')
print(f'  R2 Trans+Mel (GriffinLim): PESQ = 1.141  SI-SDR = -25.58 dB')
print(f'  R3 Trans+STFT (ISTFT)    : PESQ = {avg_pesq_e:.3f}  SI-SDR = {avg_sisdr_e:.2f} dB')
""")

# ── 7: attention viz ──────────────────────────────────────────────────────────
md("## Attention Visualization")
code("""
# ============================================================================
# Cell 7: Attention heatmaps (2 layers × 4 heads)
# ============================================================================
sample = test_ds[0]
inp_t  = torch.log1p(sample['noisy_mag'].unsqueeze(0).unsqueeze(0)).to(device)
attn   = get_attention_weights(model, inp_t)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for li, aw in enumerate(attn):
    for hi in range(4):
        ax = axes[li, hi]
        w  = aw[0, hi].numpy()
        ax.imshow(w[:64, :64], aspect='auto', cmap='viridis')
        ax.set_title(f'Layer {li+1}  Head {hi+1}', fontsize=11)
        ax.set_xlabel('Key frame')
        ax.set_ylabel('Query frame')
plt.suptitle('Self-Attention Weights — first 64 frames (test sample 0)', fontsize=14)
plt.tight_layout()
plt.savefig('attention_weights.png', dpi=150)
plt.show()
print('Saved attention_weights.png')
""")

# ── 8: summary ────────────────────────────────────────────────────────────────
code("""
# ============================================================================
# Cell 8: Save review3_summary.json + pretty table
# ============================================================================
summary = {
    'review': 3,
    'run': 'eval_only',
    'model': 'STFTTransformerEnhancer',
    'pipeline': 'STFT (n_fft=512, hop=256) → mask → ISTFT',
    'params': total_params,
    'checkpoint': {'epoch': int(ckpt['epoch']), 'val_loss': float(ckpt['val_loss'])},
    'test_samples': len(test_ds),
    'metrics': {
        'pesq_noisy':    round(avg_pesq_n, 3),
        'pesq_enhanced': round(avg_pesq_e, 3),
        'stoi_noisy':    round(avg_stoi_n, 4),
        'stoi_enhanced': round(avg_stoi_e, 4),
        'sisdr_noisy_dB':    round(avg_sisdr_n, 2),
        'sisdr_enhanced_dB': round(avg_sisdr_e, 2),
    },
    'comparison': {
        'R1_CRN_PESQ_estimated': 3.10,
        'R2_TransMel_PESQ': 1.141,
        'R2_TransMel_SISDR_dB': -25.58,
        'R3_TransSTFT_PESQ': round(avg_pesq_e, 3),
        'R3_TransSTFT_SISDR_dB': round(avg_sisdr_e, 2),
    }
}
with open('review3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved review3_summary.json')

W = 72
print('\\n' + '='*W)
print(f'{"Review":>10} {"Pipeline":>22} {"PESQ":>8} {"STOI":>7} {"SI-SDR":>10} {"Params":>8}')
print('='*W)
print(f'{"R1 CRN":>10} {"Mel (estimated)":>22} {"~3.10":>8} {"—":>7} {"—":>10} {"~2.5M":>8}')
print(f'{"R2 Trans":>10} {"Mel+GriffinLim":>22} {1.141:>8.3f} {0.695:>7.3f} {-25.58:>9.2f}dB {"2.45M":>8}')
print(f'{"R3 Trans":>10} {"STFT+ISTFT":>22} {avg_pesq_e:>8.3f} {avg_stoi_e:>7.3f} {avg_sisdr_e:>9.2f}dB {total_params/1e6:>7.2f}M')
print('='*W)
print(f'Noisy baseline: PESQ={avg_pesq_n:.3f}  STOI={avg_stoi_n:.3f}  SI-SDR={avg_sisdr_n:.2f}dB')
print(f'R3 improvement over noisy: dPESQ={avg_pesq_e-avg_pesq_n:+.3f}  dSTOI={avg_stoi_e-avg_stoi_n:+.4f}  dSI-SDR={avg_sisdr_e-avg_sisdr_n:+.2f}dB')
""")

# ── Build notebook JSON ───────────────────────────────────────────────────────
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

# ── Kaggle payload ─────────────────────────────────────────────────────────────
# Update existing eval notebook by ID (110458458) so Kaggle keeps the data sources
kaggle_payload = {
    "hasId": True,
    "id": 110458458,
    "hasText": True,
    "text": nb_text,
    "hasNewTitle": True,
    "newTitle": "Review 3 Eval STFT Transformer",
    "hasLanguage": True,
    "language": "python",
    "hasKernelType": True,
    "kernelType": "notebook",
    "hasKernelExecutionType": True,
    "kernelExecutionType": "SaveAndRunAll",
    "hasEnableGpu": True,
    "enableGpu": True,
    "hasEnableInternet": True,
    "enableInternet": True,
    # Mount v3 training output (checkpoint files)
    "kernelDataSources": ["kjadeja/review-3-stft-transformer-speech-enhancement"],
    # Mount librispeech-noise dataset
    "datasetDataSources": ["earth16/libri-speech-noise-dataset"],
}

with open("kaggle_eval_nb.json", "w", encoding="utf-8") as f:
    json.dump(kaggle_payload, f, indent=2)

print(f"Cells: {len(cells)}")
print(f"Notebook text: {len(nb_text):,} chars")
print("Written: kaggle_eval_nb.json")

# Print cell summary
for i, c in enumerate(cells):
    ct = c['cell_type']
    preview = c['source'][:60].replace('\n',' ') if isinstance(c['source'], str) else '...'
    print(f"  [{i}] {ct:8s}  {preview}")
