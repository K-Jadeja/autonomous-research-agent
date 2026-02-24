"""Build clean Kaggle-ready notebook from review3-stft-transformer.ipynb and save as kaggle_r3.json"""
import json

with open('review3-stft-transformer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Strip all outputs and reset execution counts
for cell in nb['cells']:
    cell['outputs'] = []
    cell['execution_count'] = None
    # Ensure metadata has trusted flag
    cell.setdefault('metadata', {})['trusted'] = True

# Set kernel spec
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

# Save clean version
text = json.dumps(nb, indent=1)
with open('kaggle_r3.json', 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Saved kaggle_r3.json ({len(text)} bytes, {len(nb["cells"])} cells)')
