import json, ast

with open('crn_nb_text.txt', encoding='utf-8') as f:
    nb = json.loads(f.read())

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Code cells: {len(code_cells)}')

for i, c in enumerate(code_cells):
    cid = c['id']
    src = c['source']
    # Strip lines starting with !
    lines = [l for l in src.split('\n') if not l.strip().startswith('!')]
    clean = '\n'.join(lines)
    try:
        ast.parse(clean)
        print(f'  Cell {cid}: OK')
    except SyntaxError as e:
        print(f'  Cell {cid}: SYNTAX ERROR line {e.lineno}: {e.msg}')
        slines = clean.split('\n')
        start = max(0, e.lineno - 3)
        end = min(len(slines), e.lineno + 2)
        for j in range(start, end):
            marker = '>>>' if j == e.lineno - 1 else '   '
            print(f'    {marker} {j+1}: {slines[j]}')

print('Done')
