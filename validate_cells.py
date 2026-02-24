import json, ast

with open(r'D:\Workspace\kaggle agent\kaggle_r3_dpt_nb.json') as f:
    payload = json.load(f)

nb = json.loads(payload['text'])
errors = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    lines = [l for l in src.split('\n') if not l.strip().startswith('!')]
    clean = '\n'.join(lines)
    try:
        ast.parse(clean)
        print(f"Cell {i} ({cell['id']}): OK")
    except SyntaxError as e:
        print(f"Cell {i} ({cell['id']}): SYNTAX ERROR: {e}")
        errors += 1

if errors == 0:
    print("\nAll cells pass syntax check!")
else:
    print(f"\n{errors} cells have errors!")
