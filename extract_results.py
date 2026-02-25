import json, sys

def extract_text_outputs(path, label):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    print("=" * 60)
    print(label)
    print("=" * 60)
    for i, c in enumerate(code_cells):
        texts = []
        for o in c.get('outputs', []):
            otype = o.get('output_type', '')
            if otype == 'stream':
                texts.append(''.join(o.get('text', [])))
            elif otype == 'execute_result':
                d = o.get('data', {})
                if 'text/plain' in d:
                    texts.append(''.join(d['text/plain']))
            elif otype == 'error':
                ename = o.get('ename', '?')
                evalue = o.get('evalue', '?')
                texts.append(f'ERROR: {ename}: {evalue}')
        if texts:
            txt = ''.join(texts)
            if len(txt) > 800:
                print(f'--- Cell {i+1} ({len(txt)} chars, last 600) ---')
                print(txt[-600:])
            else:
                print(f'--- Cell {i+1} ---')
                print(txt)
    print()

extract_text_outputs('review-3-dpt-stft-speech-enhancement.ipynb', 'DPT NOTEBOOK')
extract_text_outputs('crn-baseline-fixed-stft-speech-enhancement.ipynb', 'CRN NOTEBOOK')
