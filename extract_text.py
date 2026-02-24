import json

with open(r'D:\Workspace\kaggle agent\kaggle_r3_dpt_nb.json') as f:
    d = json.load(f)

text = d["text"]
print(f"Text length: {len(text):,}")

with open(r'D:\Workspace\kaggle agent\dpt_nb_text.txt', 'w') as f:
    f.write(text)
print("Written dpt_nb_text.txt")
