"""Push CRN v2 and DPT v2 notebooks to Kaggle using the CLI."""
import json, os, tempfile, subprocess, sys

NOTEBOOKS = [
    {
        "nb_text_file": "crn_v2_nb_text.txt",
        "slug": "kjadeja/crn-v2-aligned-speech-enhancement",
        "title": "CRN v2 Aligned Speech Enhancement",
    },
    {
        "nb_text_file": "dpt_v2_nb_text.txt",
        "slug": "kjadeja/dpt-v2-aligned-speech-enhancement",
        "title": "DPT v2 Aligned Speech Enhancement",
    },
]

for nb_info in NOTEBOOKS:
    print(f"\n{'='*60}")
    print(f"Pushing: {nb_info['title']}")
    print(f"{'='*60}")

    # Read notebook text
    with open(nb_info["nb_text_file"], "r", encoding="utf-8") as f:
        nb_text = f.read()

    # Validate it's proper JSON
    nb_json = json.loads(nb_text)
    print(f"  Cells: {len(nb_json['cells'])}")
    print(f"  Size: {len(nb_text):,} chars")

    # Create temp directory for push
    tmpdir = tempfile.mkdtemp(prefix="kaggle_push_")
    print(f"  Temp dir: {tmpdir}")

    # Write kernel-metadata.json
    metadata = {
        "id": nb_info["slug"],
        "title": nb_info["title"],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["earth16/libri-speech-noise-dataset"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path = os.path.join(tmpdir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote: {meta_path}")

    # Write notebook.ipynb
    nb_path = os.path.join(tmpdir, "notebook.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        f.write(nb_text)
    print(f"  Wrote: {nb_path}")

    # Push via Kaggle CLI
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", tmpdir],
        capture_output=True, text=True
    )
    print(f"  stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  stderr: {result.stderr.strip()}")
    if result.returncode != 0:
        print(f"  FAILED (return code {result.returncode})")
    else:
        print(f"  SUCCESS")

print("\n" + "="*60)
print("Both notebooks pushed. Check Kaggle for execution status.")
print("="*60)
