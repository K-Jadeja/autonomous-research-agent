"""Download and extract test data, then run full eval locally."""
import os, sys, urllib.request, subprocess, glob, json, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset
import numpy as np
from pesq import pesq as pesq_metric
from pystoi import stoi as stoi_metric

# ─── Config ───────────────────────────────────────────────────────────────
N_FFT      = 512
HOP_LENGTH = 256
N_FREQ     = N_FFT // 2 + 1   # 257
SR         = 16000
MAX_LEN    = 48000             # 3 s
CKPT_PATH  = r'D:\Workspace\kaggle agent\ckpt_dl\stft_transformer_best.pth'
DATA_BASE  = r'D:\Workspace\kaggle agent\data'

device = 'cpu'
print(f'Device: {device}')
print(f'STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, freq_bins={N_FREQ}')

# ─── Download & extract test data ─────────────────────────────────────────
os.makedirs(DATA_BASE, exist_ok=True)

test_7z  = os.path.join(DATA_BASE, 'test.7z')
ytest_7z = os.path.join(DATA_BASE, 'y_test.7z')

test_url  = "https://storage.googleapis.com/kagglesdsdata/datasets/796374/1433476/test.7z?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260224T110512Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3bf54409f4f86072de5740b36668ac9b64c7314235d4be9f681d2e45345c35f2ababb63f3593ff34fa7b5b8a440ccc8c6a5e58dc024da4fe93a89e6cbe270192868634c05bc122787183835054748105ce0ea1740d72edf88c8b2db50b907051f1f9e74b024d68b14a7acb98b9361df0c8a7eee4e30901d689858e9afe8b91d08340cc9e9fa49b2b53afc376e466496dea89290f8a2d9ec91d5eb050ff3bf5135ccba06b840e8c1763075e4a048d91c63b1e831e5f1f809ff37b505e45839c35a2152265452cefb9a855da8895d4a659ac391e2018d5260637797f6c0d5e99a740a5fc0c29c33e06066c5fed81000fe7e6bb224df8aff05fd52d16bb0cb07d6d"
ytest_url = "https://storage.googleapis.com/kagglesdsdata/datasets/796374/1433476/y_test.7z?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260224%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260224T110815Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8a5cd29223e6cfad507c39b87194047e1f018a1191ce51f70563bbe2b814f893df98b762d3757a4ee5b864bf36f75e6d88cec667751849ec0ae2ccdb1986eba28cc270a0abe4c7eae4ccfd948ed7cf610692b72f7201a5a482247969ac286af7720dbea2707c621dc821b0c6978f2f9653b24566b32c3ee67d42b3293d3e550e0eaaa27fed6226cc8171619f80cae7c85f5523b8f9a9fd4050cd0aed5516d42c94ec4ae01b032f10013c8a48790c1df25e0586be72845d6b1567f8b84ad12c20d92d1fe60421ed7afa654afc22193efc45db71de695b189aaeb2075b5dfd8520c108581103a0e1699f7da3c1c0a3356a936713ba2dda0baa9c2b264ac866844e"

for url, fpath, label in [(test_url, test_7z, 'test.7z'), (ytest_url, ytest_7z, 'y_test.7z')]:
    if not os.path.exists(fpath):
        print(f'Downloading {label} ...')
        urllib.request.urlretrieve(url, fpath)
        print(f'  Downloaded: {os.path.getsize(fpath)/1e6:.1f} MB')
    else:
        print(f'{label} already exists')

# Extract with 7z (need 7-zip installed on Windows)
seven_z = r'C:\Program Files\7-Zip\7z.exe'
if not os.path.exists(seven_z):
    seven_z = '7z'  # try PATH

for arch in [test_7z, ytest_7z]:
    name = os.path.basename(arch)
    target_dir = os.path.join(DATA_BASE, name.replace('.7z', ''))
    if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f'{name} already extracted')
        continue
    print(f'Extracting {name} ...')
    r = subprocess.run([seven_z, 'x', arch, f'-o{DATA_BASE}', '-y'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  ERROR: {r.stderr[:500]}')
        sys.exit(1)
    print(f'  Extracted OK')

# Find dirs
noisy_test_dir = os.path.join(DATA_BASE, 'test')
clean_test_dir = os.path.join(DATA_BASE, 'y_test')
for tag, d in [('noisy_test', noisy_test_dir), ('clean_test', clean_test_dir)]:
    n = len(glob.glob(os.path.join(d, '*.wav'))) if os.path.isdir(d) else 0
    print(f'  {tag}: {d} ({n} files)')

# ─── Dataset ──────────────────────────────────────────────────────────────
class STFTSpeechDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 sr=SR, max_len=MAX_LEN):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        assert len(self.noisy_files) == len(self.clean_files), \
            f'Mismatch: {len(self.noisy_files)} noisy vs {len(self.clean_files)} clean'
        self.n_fft = n_fft; self.hop_length = hop_length
        self.sr = sr; self.max_len = max_len
        self.window = torch.hann_window(n_fft)

    def __len__(self): return len(self.noisy_files)

    def _load_fix(self, path):
        wav, sr_file = sf.read(path, dtype='float32')
        wav = torch.from_numpy(wav)
        if wav.ndim > 1:
            wav = wav[:, 0]  # mono
        if sr_file != self.sr:
            # Simple resampling via interpolation
            ratio = self.sr / sr_file
            new_len = int(len(wav) * ratio)
            wav = F.interpolate(wav.unsqueeze(0).unsqueeze(0), size=new_len,
                                mode='linear', align_corners=False).squeeze()
        if wav.shape[0] > self.max_len:
            wav = wav[:self.max_len]  # deterministic crop for eval
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

# ─── Model ────────────────────────────────────────────────────────────────
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

# ─── SI-SDR ───────────────────────────────────────────────────────────────
def si_sdr(estimate, reference):
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    dot = torch.sum(ref * est)
    s_target = dot * ref / (torch.sum(ref**2) + 1e-8)
    e_noise  = est - s_target
    return 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + 1e-8) + 1e-8)

# ─── Load model ───────────────────────────────────────────────────────────
print(f'\nLoading checkpoint: {CKPT_PATH}')
model = STFTTransformerEnhancer(n_freq=N_FREQ)
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model: {total_params:,} params ({total_params/1e6:.2f}M)')
print(f'Loaded: epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f}')

# ─── Build dataset ────────────────────────────────────────────────────────
test_ds = STFTSpeechDataset(noisy_test_dir, clean_test_dir)
print(f'Test samples: {len(test_ds)}')

# ─── Full evaluation ─────────────────────────────────────────────────────
window_eval = torch.hann_window(N_FFT)

pesq_noisy_list, pesq_enh_list   = [], []
stoi_noisy_list, stoi_enh_list   = [], []
sisdr_noisy_list, sisdr_enh_list = [], []

for i in range(len(test_ds)):
    if i % 10 == 0:
        print(f'  Eval {i}/{len(test_ds)} ...')
    s           = test_ds[i]
    noisy_mag   = s['noisy_mag'].unsqueeze(0)     # (1,257,T)
    noisy_phase = s['noisy_phase'].unsqueeze(0)    # (1,257,T)
    clean_np    = s['clean_wav'].numpy()
    noisy_np    = s['noisy_wav'].numpy()

    with torch.no_grad():
        inp  = torch.log1p(noisy_mag).unsqueeze(1)   # (1,1,257,T)
        mask = model(inp)                              # (1,257,T)
        enh_mag = (mask * noisy_mag).squeeze(0)       # (257,T)

    enh_stft = enh_mag * torch.exp(1j * noisy_phase.squeeze(0))
    enh_wav  = torch.istft(enh_stft, N_FFT, HOP_LENGTH, window=window_eval, length=MAX_LEN)
    enh_np   = enh_wav.numpy()

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

# ─── Results ──────────────────────────────────────────────────────────────
avg_pesq_n  = float(np.mean(pesq_noisy_list))  if pesq_noisy_list  else 0.0
avg_pesq_e  = float(np.mean(pesq_enh_list))    if pesq_enh_list    else 0.0
avg_stoi_n  = float(np.mean(stoi_noisy_list))
avg_stoi_e  = float(np.mean(stoi_enh_list))
avg_sisdr_n = float(np.mean(sisdr_noisy_list))
avg_sisdr_e = float(np.mean(sisdr_enh_list))

print(f'\n{"="*72}')
print(f'Results on {len(test_ds)} test files:')
print(f'  PESQ  : noisy={avg_pesq_n:.3f}  enhanced={avg_pesq_e:.3f}  Δ={avg_pesq_e-avg_pesq_n:+.3f}')
print(f'  STOI  : noisy={avg_stoi_n:.3f}  enhanced={avg_stoi_e:.3f}  Δ={avg_stoi_e-avg_stoi_n:+.4f}')
print(f'  SI-SDR: noisy={avg_sisdr_n:.2f}dB  enhanced={avg_sisdr_e:.2f}dB  Δ={avg_sisdr_e-avg_sisdr_n:+.2f}dB')

print(f'\n--- Review comparison ---')
print(f'  R1 CRN       (estimated) : PESQ ~ 3.10')
print(f'  R2 Trans+Mel (GriffinLim): PESQ = 1.141  SI-SDR = -25.58 dB')
print(f'  R3 Trans+STFT (ISTFT)    : PESQ = {avg_pesq_e:.3f}  SI-SDR = {avg_sisdr_e:.2f} dB')

W = 72
print(f'\n{"="*W}')
print(f'{"Review":>10} {"Pipeline":>22} {"PESQ":>8} {"STOI":>7} {"SI-SDR":>10} {"Params":>8}')
print('='*W)
print(f'{"R1 CRN":>10} {"Mel (estimated)":>22} {"~3.10":>8} {"—":>7} {"—":>10} {"~2.5M":>8}')
print(f'{"R2 Trans":>10} {"Mel+GriffinLim":>22} {1.141:>8.3f} {0.695:>7.3f} {-25.58:>9.2f}dB {"2.45M":>8}')
print(f'{"R3 Trans":>10} {"STFT+ISTFT":>22} {avg_pesq_e:>8.3f} {avg_stoi_e:>7.3f} {avg_sisdr_e:>9.2f}dB {total_params/1e6:>7.2f}M')
print('='*W)
print(f'Noisy baseline: PESQ={avg_pesq_n:.3f}  STOI={avg_stoi_n:.3f}  SI-SDR={avg_sisdr_n:.2f}dB')
print(f'R3 improvement over noisy: dPESQ={avg_pesq_e-avg_pesq_n:+.3f}  dSTOI={avg_stoi_e-avg_stoi_n:+.4f}  dSI-SDR={avg_sisdr_e-avg_sisdr_n:+.2f}dB')

# ─── Save summary JSON ───────────────────────────────────────────────────
summary = {
    'review': 3,
    'run': 'local_eval',
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
out_json = r'D:\Workspace\kaggle agent\review3_summary.json'
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSaved: {out_json}')
