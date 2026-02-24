"""Quick local validation of Review 3 STFT-Transformer architecture."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

N_FFT = 512
HOP_LENGTH = 256
N_FREQ = N_FFT // 2 + 1  # 257
SR = 16000
MAX_LEN = 48000

device = 'cpu'

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class STFTTransformerEnhancer(nn.Module):
    def __init__(self, n_freq=257, d_model=256, nhead=4, num_layers=2,
                 dim_ff=1024, dropout=0.1):
        super().__init__()
        self.n_freq = n_freq
        self.encoder = nn.Sequential(
            ConvBlock(1, 64), ConvBlock(64, 128), ConvBlock(128, 256))
        self.pre_proj  = nn.Linear(256, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.post_proj = nn.Linear(d_model, 256)
        self.decoder = nn.Sequential(
            ConvBlock(256, 128), ConvBlock(128, 64),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        enc = self.encoder(x)
        feat = enc.mean(dim=2).permute(0, 2, 1)
        feat = self.pos_enc(self.pre_proj(feat))
        feat = self.post_proj(self.transformer(feat))
        feat = feat.permute(0, 2, 1)
        feat = feat.unsqueeze(2).expand(-1, -1, self.n_freq, -1)
        mask = self.decoder(feat).squeeze(1)
        return mask

# ===== TEST 1: Model =====
model = STFTTransformerEnhancer(n_freq=N_FREQ)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[OK] Params: {total_params:,} ({total_params/1e6:.2f}M)')

dummy = torch.randn(2, 1, N_FREQ, 188)
with torch.no_grad():
    out = model(dummy)
print(f'[OK] Forward: {dummy.shape} -> {out.shape}')
assert out.shape == (2, N_FREQ, 188)
assert 0 <= out.min() and out.max() <= 1
print('[OK] Mask in [0,1]')

# ===== TEST 2: Attention extraction =====
def get_attention_weights(mdl, x):
    mdl.eval()
    weights = []
    with torch.no_grad():
        enc = mdl.encoder(x)
        feat = enc.mean(dim=2).permute(0, 2, 1)
        feat = mdl.pos_enc(mdl.pre_proj(feat))
        for layer in mdl.transformer.layers:
            normed = layer.norm1(feat)
            attn_out, w = layer.self_attn(normed, normed, normed,
                need_weights=True, average_attn_weights=False)
            weights.append(w.cpu())
            feat = feat + layer.dropout1(attn_out)
            normed2 = layer.norm2(feat)
            ff_out = layer.linear2(
                F.dropout(layer.activation(layer.linear1(normed2)), p=0.0, training=False))
            feat = feat + ff_out
    return weights

attn = get_attention_weights(model, torch.randn(1, 1, N_FREQ, 188))
print(f'[OK] Attention: {len(attn)} layers, shape {attn[0].shape}')

# ===== TEST 3: STFT → ISTFT roundtrip =====
wav = torch.randn(1, MAX_LEN)
window = torch.hann_window(N_FFT)
stft = torch.stft(wav[0], N_FFT, HOP_LENGTH, window=window, return_complex=True)
mag = stft.abs()
phase = torch.angle(stft)
print(f'[OK] STFT shape: {stft.shape} (expect {N_FREQ} x T)')

# Reconstruct with original phase (perfect roundtrip)
recon_stft = mag * torch.exp(1j * phase)
recon = torch.istft(recon_stft, N_FFT, HOP_LENGTH, window=window, length=MAX_LEN)
err = (wav[0] - recon).abs().max().item()
print(f'[OK] ISTFT roundtrip error: {err:.2e} (should be ~1e-7)')
assert err < 1e-4, f'ISTFT roundtrip error too high: {err}'

# With mask applied
with torch.no_grad():
    inp = torch.log1p(mag).unsqueeze(0).unsqueeze(0)  # (1, 1, 257, T)
    mask = model(inp)                                  # (1, 257, T)
    enhanced_mag = (mask * mag.unsqueeze(0)).squeeze(0) # (257, T)
    enhanced_stft = enhanced_mag * torch.exp(1j * phase)
    enhanced_wav = torch.istft(enhanced_stft, N_FFT, HOP_LENGTH, window=window, length=MAX_LEN)
    print(f'[OK] Enhanced waveform shape: {enhanced_wav.shape}')

# ===== TEST 4: SI-SDR utility =====
def si_sdr(estimate, reference):
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    dot = torch.sum(ref * est)
    s_target = dot * ref / (torch.sum(ref ** 2) + 1e-8)
    e_noise = est - s_target
    return 10 * torch.log10(torch.sum(s_target**2) / (torch.sum(e_noise**2) + 1e-8) + 1e-8)

sdr = si_sdr(enhanced_wav, wav[0])
print(f'[OK] SI-SDR (random): {sdr.item():.2f} dB')

# ===== TEST 5: Smoke train (10 steps) =====
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for step in range(10):
    noisy_wav = torch.randn(4, MAX_LEN)
    clean_wav = torch.randn(4, MAX_LEN)
    # Compute STFTs
    noisy_mags, clean_mags = [], []
    for b in range(4):
        ns = torch.stft(noisy_wav[b], N_FFT, HOP_LENGTH, window=window, return_complex=True)
        cs = torch.stft(clean_wav[b], N_FFT, HOP_LENGTH, window=window, return_complex=True)
        noisy_mags.append(ns.abs())
        clean_mags.append(cs.abs())
    noisy_mag = torch.stack(noisy_mags)  # (4, 257, T)
    clean_mag = torch.stack(clean_mags)  # (4, 257, T)
    
    inp = torch.log1p(noisy_mag).unsqueeze(1)  # (4, 1, 257, T)
    mask = model(inp)
    enhanced = mask * noisy_mag
    loss = F.l1_loss(torch.log1p(enhanced), torch.log1p(clean_mag))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    losses.append(loss.item())
    
print(f'[OK] Smoke train: loss {losses[0]:.4f} -> {losses[-1]:.4f}')
print(f'\n===== ALL TESTS PASSED =====')
