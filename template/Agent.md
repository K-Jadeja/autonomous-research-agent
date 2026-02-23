# AGENT INSTRUCTIONS & WORKSPACE GUIDE
_Updated: Feb 23, 2026_

---

## WHO YOU ARE
You are a researcher/student AI agent completing a Capstone project at VIT Bhopal.
- Project: "Lightweight Speech Enhancement Using Shallow Transformers"
- Team: Krishnasinh Jadeja (22BLC1211), Kirtan Sondagar (22BLC1228), Prabhu Kalyan Panda (22BLC1213)
- Guide: Dr. Praveen Jaraut
- This workspace (`d:\Workspace\kaggle agent`) is YOUR personal area

---

## MEMORY SYSTEM — READ THESE ON EVERY SESSION START
1. **`memory.md`** — Long-term project memory. Architecture, dataset, notebook cell map, known bugs, Kaggle slugs
2. **`learningsandprogress.md`** — Short-term session log. What was done, what's next, session-to-session handoffs
3. **This file (`Agent.md`)** — Instructions for how to operate

**Always update these files at the end of every session or before crashing.**

---

## TOOLS YOU HAVE
- **Kaggle MCP tools**: `mcp_kaggle_save_notebook`, `mcp_kaggle_get_notebook_session_status`, `mcp_kaggle_get_notebook_info`, `mcp_kaggle_list_notebook_files`, `mcp_kaggle_download_notebook_output`
- **Local notebook**: VS Code Jupyter kernel — run cells with `run_notebook_cell`
- **File tools**: read_file, create_file, replace_string_in_file, grep_search
- **Terminal**: run_in_terminal (Windows PowerShell)

---

## STANDARD WORKFLOW FOR NOTEBOOK WORK
1. Read `memory.md` and `learningsandprogress.md` first
2. Open the local notebook in VS Code
3. Run cells locally ONE BY ONE to validate code is correct
4. Only run Kaggle-specific cells (data extraction, training) via Kaggle MCP session
5. When all architecture/logic cells pass locally → upload to Kaggle and run training
6. Monitor session status → download output → verify results
7. Update memory files with results before ending session

---

## KAGGLE NOTEBOOK WORKFLOW

### To save + run a notebook on Kaggle:
Read the notebook file as text, then call `mcp_kaggle_save_notebook` with:
- `slug` = 'review2-transformer-speechenhance'
- `language` = 'python'
- `kernelType` = 'notebook'
- `enableGpu` = True
- `enableInternet` = True
- `datasetDataSources` = ['earth16/libri-speech-noise-dataset']
- `kernelExecutionType` = 'SaveAndRunAll'
- `text` = the full notebook .ipynb JSON as string

### To check status:
`mcp_kaggle_get_notebook_session_status` with userName='kjadeja', kernelSlug='review2-transformer-speechenhance'

### To download output:
`mcp_kaggle_download_notebook_output` with ownerSlug='kjadeja', kernelSlug='review2-transformer-speechenhance'

---

## PROJECT TIMELINE
| Phase | Date | Target | Status |
|---|---|---|---|
| Review 1 | Jan 21, 2026 | CRN Baseline PESQ ~3.1 | DONE |
| Review 2 | Feb 18, 2026 | CNN-Transformer PESQ ≥3.2 | Upload pending |
| Review 3 | Mar 18, 2026 | SileroVAD integration | Upcoming |
| Final | Apr 8, 2026 | Quantized model + Gradio demo | Upcoming |

---

## CURRENT NOTEBOOK FILES
| File | Description |
|---|---|
| `review2-transformer-speechenhance.ipynb` | Main Review 2 notebook — CNN+Transformer model |
| `memory.md` | Long-term memory |
| `learningsandprogress.md` | Session progress log |
| `Agent.md` | This file — agent instructions |

---

## IMPORTANT: DO NOT LOSE THESE
- **Kaggle username**: `kjadeja`
- **Dataset slug**: `earth16/libri-speech-noise-dataset` (6.6GB, 7000+105 WAV pairs)
- **CRN notebook**: `kjadeja/baseline-crn-speechenhance` (v6, Feb 20 result: PESQ ~3.1)
- **Review 2 target slug**: `kjadeja/review2-transformer-speechenhance`
- **Checkpoint filename**: `transformer_best.pth`
- **Local notebook path**: `d:\Workspace\kaggle agent\review2-transformer-speechenhance.ipynb`

---

## KEY TECHNICAL REMINDERS
- Model: `ShallowTransformerEnhancer` (2.45M params, 2 layers, 4 heads, Pre-LN Transformer)
- Input/Output: (B, 128, T) log-mel spectrograms, mask-based enhancement
- Waveform reconstruction: `expm1` → `InverseMelScale(driver='gelsd')` → `GriffinLim(n_iter=32)`
- The `driver='gelsd'` is CRITICAL for `InverseMelScale` to not crash
- Eval uses REAL waveform reconstruction (unlike CRN Review 1 which was approximate)
- Cell 14 (comparison) requires `avg_pesq` from Cell 12 to run
- Training uses `extract_path = '/kaggle/working/extracted_data'` (NOT local path)
