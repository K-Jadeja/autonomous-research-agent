import json

with open(r'c:\Users\Krishna\AppData\Roaming\Code\User\workspaceStorage\fae853281beb66b416b2cbf4bd2825e5\GitHub.copilot-chat\chat-session-resources\d506fc8b-6c94-40ed-87b3-b8bac3542c8b\toolu_bdrk_016TR5y9Havr5PJGKMMDLAop__vscode-1771853923014\content.json', encoding='utf-8') as f:
    data = json.load(f)

print("Keys:", list(data.keys())[:20])
source = data.get('source', '')
print("Source len:", len(source))

try:
    nb = json.loads(source)
    cells = nb.get('cells', [])
    print(f"Cells: {len(cells)}")
    for i, c in enumerate(cells):
        src = ''.join(c.get('source', []))
        keywords = ['MelSpectrogram', 'n_fft', 'hop', 'stft', 'pesq', 'PESQ', 'stoi', 'STOI', 'loss', 'Loss', 'model(', 'class CRN', 'sigmoid', 'mask', 'enhanced']
        if any(k in src for k in keywords):
            ct = c.get('cell_type', '?')
            print(f"\n=== Cell {i} ({ct}) ===")
            print(src[:600])
except Exception as e:
    print("Not JSON notebook:", e)
    # Maybe it's raw python
    for line in source.split('\n'):
        low = line.lower()
        if any(k in low for k in ['n_fft', 'hop', 'mel', 'stft', 'pesq', 'loss', 'class', 'sigmoid', 'mask']):
            print(line[:120])
