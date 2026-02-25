"""Local forward-pass test for CRN model (extracted from notebook)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FFT = 512
HOP_LENGTH = 256
N_FREQ = N_FFT // 2 + 1  # 257

class CRNBaseline(nn.Module):
    """CRN for STFT-based speech enhancement."""
    def __init__(self, n_freq=257):
        super().__init__()
        self.n_freq = n_freq
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=256, num_layers=2,
            batch_first=True, dropout=0.1)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.mask_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.xavier_normal_(param)
                    elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x):
        B, _, F_orig, T_orig = x.shape
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        B2, C, Fenc, T = e3.shape
        lstm_in = e3.permute(0, 2, 3, 1).reshape(B2 * Fenc, T, C)
        lstm_out, _ = self.lstm(lstm_in)
        h = lstm_out.reshape(B2, Fenc, T, C).permute(0, 3, 1, 2)
        d3 = self.dec3(h)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        if d1.shape[2] != F_orig:
            d1 = F.interpolate(d1, size=(F_orig, T_orig), mode='bilinear', align_corners=False)
        mask = self.mask_conv(d1).squeeze(1)
        return mask


device = 'cpu'
model = CRNBaseline(n_freq=N_FREQ).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'CRNBaseline: {total_params:,} params ({total_params/1e6:.2f}M)')

# Test with typical input size: batch=2, freq=257, time=188 (3s / hop=256)
with torch.no_grad():
    dummy = torch.randn(2, 1, N_FREQ, 188).to(device)
    mask = model(dummy)
    print(f'Input:  {dummy.shape}')
    print(f'Output: {mask.shape}')
    assert mask.shape == (2, N_FREQ, 188), f'Shape mismatch: {mask.shape}'
    assert mask.min().item() >= 0 and mask.max().item() <= 1, f'Mask out of [0,1]'
    print(f'Mask range: [{mask.min().item():.4f}, {mask.max().item():.4f}]')

# Encoder shapes trace
print('\n--- Encoder/Decoder Shape Trace ---')
with torch.no_grad():
    x = torch.randn(1, 1, 257, 188)
    e1 = model.enc1(x)
    e2 = model.enc2(e1)
    e3 = model.enc3(e2)
    print(f'enc1: {x.shape} -> {e1.shape}')        # (1,64,129,188)
    print(f'enc2: {e1.shape} -> {e2.shape}')        # (1,128,65,188)
    print(f'enc3: {e2.shape} -> {e3.shape}')        # (1,256,33,188)

    B2, C, F_enc, T = e3.shape
    lstm_in = e3.permute(0, 2, 3, 1).reshape(B2 * F_enc, T, C)
    print(f'LSTM in: {lstm_in.shape}')              # (33,188,256)
    lstm_out, _ = model.lstm(lstm_in)
    print(f'LSTM out: {lstm_out.shape}')
    h = lstm_out.reshape(B2, F_enc, T, C).permute(0, 3, 1, 2)
    print(f'h: {h.shape}')

    d3 = model.dec3(h)
    d2 = model.dec2(d3)
    d1 = model.dec1(d2)
    print(f'dec3: {h.shape} -> {d3.shape}')
    print(f'dec2: {d3.shape} -> {d2.shape}')
    print(f'dec1: {d2.shape} -> {d1.shape}')

# Architecture breakdown
enc_p = sum(p.numel() for n, p in model.named_parameters() if 'enc' in n)
lstm_p = sum(p.numel() for n, p in model.named_parameters() if 'lstm' in n)
dec_p = sum(p.numel() for n, p in model.named_parameters() if 'dec' in n or 'mask_conv' in n)
print(f'\n  Encoder:  {enc_p:>10,} ({enc_p/total_params*100:.1f}%)')
print(f'  LSTM:     {lstm_p:>10,} ({lstm_p/total_params*100:.1f}%)')
print(f'  Decoder:  {dec_p:>10,} ({dec_p/total_params*100:.1f}%)')

print('\nAll tests PASSED')
