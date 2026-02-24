"""Apply all 4 bug fixes to review3-stft-transformer.ipynb on disk."""
import json

with open('review3-stft-transformer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes_applied = []

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    original = src

    # Fix 1: total_mem -> getattr (safe for Python 3.12)
    old_vram = "print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"
    new_vram = """props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram / 1e9:.1f}GB')"""
    if old_vram in src:
        src = src.replace(old_vram, new_vram)
        fixes_applied.append('total_mem -> getattr')

    # Fix 2: Remove verbose=False from ReduceLROnPlateau (removed in Python 3.12)
    if ', verbose=False' in src:
        src = src.replace(', verbose=False', '')
        fixes_applied.append('removed verbose=False')

    # Fix 3: Save val_loss as Python float (prevents numpy UnpicklingError)
    if "'val_loss': va_loss}" in src:
        src = src.replace("'val_loss': va_loss}", "'val_loss': float(va_loss)}")
        fixes_applied.append('val_loss -> float(va_loss)')

    # Fix 4: Add weights_only=False to torch.load (PyTorch 2.6+)
    old_load = 'torch.load(CKPT, map_location=device)'
    new_load = 'torch.load(CKPT, map_location=device, weights_only=False)'
    if old_load in src and 'weights_only' not in src:
        src = src.replace(old_load, new_load)
        fixes_applied.append('added weights_only=False')

    if src != original:
        cell['source'] = src

with open('review3-stft-transformer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Applied {len(fixes_applied)} fixes:')
for fix in fixes_applied:
    print(f'  - {fix}')
