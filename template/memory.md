_Use this proactively for long term memory — LAST UPDATED: [DATE]_

---

## Project: [YOUR PROJECT TITLE]

**Team:** [TEAM MEMBER 1 (ID)], [TEAM MEMBER 2 (ID)], [TEAM MEMBER 3 (ID)]  
**Guide/Supervisor:** [ADVISOR NAME]  
**Institution:** [YOUR INSTITUTION]  
**Project Type:** [Capstone / Thesis / Research / Course Project]

---

## Overview

**Problem Statement:** [1-2 sentences describing what you're solving]

**Objective:** [Main goal of the project]

**Key Innovation:** [What makes your approach unique]

---

## Architecture/Methodology

### High-Level Approach
```
[Input] → [Processing Step 1] → [Processing Step 2] → [Output]
```

### Technical Specifications

#### Model Architecture (if ML/DL project)
- **Input:** [e.g., Log-Mel spectrogram (128 mels, n_fft=512, hop=256)]
- **Encoder:** [Architecture details]
- **Core Model:** [Main model architecture]
- **Decoder:** [Output generation details]
- **Output:** [Final output format]

#### Parameters
- Total parameters: [X M]
- Target metric: [e.g., PESQ ≥ 3.2]
- Target latency: [e.g., < 15ms]

#### Alternative: Algorithm/Method Details (if non-ML)
- **Method:** [Algorithm or approach name]
- **Key components:** [List main components]
- **Complexity:** [Time/space complexity if applicable]

---

## Dataset

### Primary Dataset
- **Name:** [Dataset name]
- **Source:** [Kaggle / Custom / Public dataset]
- **Kaggle slug:** `username/dataset-name`
- **Size:** [X GB, Y samples]
- **Format:** [e.g., .7z archives, WAV files, CSV]
- **Train/Test split:** [e.g., 7000 train / 105 test]

### Data Characteristics
- **Sampling rate:** [e.g., 16kHz]
- **Channels:** [e.g., single-channel mono]
- **Duration:** [e.g., 3-second segments]
- **Preprocessing:** [e.g., random crop during training, normalization]
- **Labels:** [Description of labels/annotations]

### Data Location
- **Local:** `[path/to/data]`
- **Kaggle:** `/kaggle/working/extracted_data`
- **Extraction method:** [e.g., `!7z x archive.7z -o/path`]

---

## Phase Timeline

| Phase | Date | Target | Status | Notes |
|---|---|---|---|---|
| Phase 1 | [DATE] | [Milestone] | ✅ Complete | [Key results] |
| Phase 2 | [DATE] | [Milestone] | 🟡 In Progress | [Current focus] |
| Phase 3 | [DATE] | [Milestone] | ⚪ Not Started | |
| Final | [DATE] | [Final deliverable] | ⚪ Not Started | |

---

## Notebooks & Code

### Current Notebooks

#### [Notebook 1 Name]
**Local file:** `[path/to/notebook1.ipynb]`  
**Kaggle slug:** `username/notebook-slug`  
**Status:** [Current status]

**Cell Map:**
| # | Type | Name | Local Status | Notes |
|---|---|---|---|---|
| 1 | MD | Title | — | |
| 2 | Code | Imports & setup | ✅ | [Notes] |
| 3 | Code | Data loading | ⚠️ | Kaggle only |
| 4 | Code | Model definition | ✅ | [Notes] |
| 5 | Code | Training loop | ⚠️ | Kaggle only |

**Architecture Summary:**
```
[Visual representation or pseudocode of the architecture]
```

**Training Config:**
- Loss: [Loss function]
- Optimizer: [Optimizer + lr]
- Scheduler: [Learning rate schedule]
- Batch size: [N]
- Epochs: [N]
- Checkpoint: `[filename.pth]`

**Evaluation Pipeline:**
- Metrics: [List metrics: e.g., PESQ, STOI, SI-SDR]
- Method: [How evaluation is done]
- Test set: [Description]

**Known Gotchas:**
- [Issue 1]: [Description] → [Solution]
- [Issue 2]: [Description] → [Solution]

**Next Steps:**
1. [Next action]
2. [Next action]

#### [Notebook 2 Name]
[Same structure as above]

---

## Results

### Phase 1 Results
- **Metric 1:** [Value] — [Comparison to baseline]
- **Metric 2:** [Value]
- **Key findings:** [Summary]

### Phase 2 Results
- **Metric 1:** [Value]
- **Metric 2:** [Value]
- **Key findings:** [Summary]

---

## Tools & Stack

### Core Libraries
- [Framework, e.g., PyTorch 2.x]
- [Audio processing, e.g., torchaudio]
- [Metrics, e.g., pesq, pystoi]
- [Visualization, e.g., matplotlib, seaborn]
- [Utilities, e.g., numpy, pandas, tqdm]

### Development Environment
- **OS:** [Windows/Mac/Linux]
- **Python:** [Version]
- **IDE:** [VS Code / PyCharm / Jupyter]
- **Virtual env:** [venv / conda]

### Compute Resources
- **Local:** [CPU/GPU specs]
- **Cloud:** [Kaggle T4 / Colab / AWS / etc.]
- **VRAM:** [X GB]

---

## Literature & References

### Key Papers
1. [Paper 1 title] — [Authors] ([Year]) — [Key contribution]
2. [Paper 2 title] — [Authors] ([Year]) — [Key contribution]

### Datasets
1. [Dataset name] — [Source]

### Code References
1. [Repository / Tutorial] — [URL] — [What you used]

---

## Technical Notes

### Important Implementation Details
- [Detail 1]: [Explanation]
- [Detail 2]: [Explanation]

### Hyperparameter Choices
- [Parameter]: [Value] — [Rationale]

### Preprocessing Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Post-processing
- [Description]

---

## Issues & Debugging Log

### Resolved Issues
| Date | Issue | Solution | Root Cause |
|---|---|---|---|
| [DATE] | [Description] | [Fix] | [Why it happened] |

### Open Issues
| Date | Issue | Impact | Priority |
|---|---|---|---|
| [DATE] | [Description] | [High/Med/Low] | [P0/P1/P2] |

---

## Resources

### Project Documents
- Proposal: `[filename]`
- Literature review: `[filename]`
- Presentation: `[filename]`

### External Links
- [Kaggle notebook]
- [Dataset page]
- [Documentation]
- [Tutorial]

---

## Administrative

### Meeting Notes
- **[DATE]:** [Summary of discussion with advisor]
- **[DATE]:** [Next review meeting]

### Deliverables
| Item | Due Date | Status |
|---|---|---|
| Proposal | [DATE] | ✅ |
| Review 1 | [DATE] | ✅ |
| Review 2 | [DATE] | 🟡 |
| Final Report | [DATE] | ⚪ |

---

_Last updated: [DATE] by [NAME]_
