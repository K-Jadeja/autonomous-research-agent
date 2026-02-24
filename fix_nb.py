"""Apply all 4 bug fixes to review3-stft-transformer.ipynb on disk.
Source is stored as a LIST of strings (one per line)."""
import json

with open('review3-stft-transformer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes_applied = []

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    lines = cell['source']  # list of strings

    # Fix 1: total_mem -> getattr (Cell 1)
    new_lines = []
    fixed1 = False
    for line in lines:
        if 'total_mem' in line and 'getattr' not in line and 'get_device_properties' in line:
            new_lines.append("    props = torch.cuda.get_device_properties(0)\n")
            new_lines.append("    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))\n")
            new_lines.append("    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram / 1e9:.1f}GB')\n")
            fixed1 = True
        else:
            new_lines.append(line)
    if fixed1:
        lines = new_lines
        fixes_applied.append('total_mem -> getattr')

    # Fix 2: Remove verbose=False (Cell 6)
    new_lines = []
    fixed2 = False
    for line in lines:
        if 'verbose=False' in line:
            line = line.replace(', verbose=False', '')
            fixed2 = True
        new_lines.append(line)
    if fixed2:
        lines = new_lines
        fixes_applied.append('removed verbose=False')

    # Fix 3: float(va_loss) (Cell 7)
    new_lines = []
    fixed3 = False
    for line in lines:
        if "'val_loss': va_loss}" in line:
            line = line.replace("'val_loss': va_loss}", "'val_loss': float(va_loss)}")
            fixed3 = True
        new_lines.append(line)
    if fixed3:
        lines = new_lines
        fixes_applied.append('val_loss -> float(va_loss)')

    # Fix 4: weights_only=False (Cell 9)
    new_lines = []
    fixed4 = False
    for line in lines:
        if 'torch.load(CKPT, map_location=device)' in line and 'weights_only' not in line:
            line = line.replace('torch.load(CKPT, map_location=device)',
                               'torch.load(CKPT, map_location=device, weights_only=False)')
            fixed4 = True
        new_lines.append(line)
    if fixed4:
        lines = new_lines
        fixes_applied.append('added weights_only=False')

    cell['source'] = lines

with open('review3-stft-transformer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Applied {len(fixes_applied)} fixes:')
for fix in fixes_applied:
    print(f'  - {fix}')
