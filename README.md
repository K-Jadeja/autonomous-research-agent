# Autonomous Research Agent Setup

An intelligent workspace system for automating research workflows using AI agents with persistent memory, external tool integration (Kaggle MCP), and structured documentation for capstone/research projects.

---

## Table of Contents

- [Overview](#overview)
- [What is OpenClaw & Autonomous Agents?](#what-is-openclaw--autonomous-agents)
- [System Architecture](#system-architecture)
- [How This Setup Works](#how-this-setup-works)
  - [Memory System](#memory-system)
  - [Agent Instructions](#agent-instructions)
  - [Kaggle MCP Integration](#kaggle-mcp-integration)
- [File Structure](#file-structure)
- [How to Use This Setup](#how-to-use-this-setup)
  - [Step 1: Initial Setup](#step-1-initial-setup)
  - [Step 2: Create Memory Files](#step-2-create-memory-files)
  - [Step 3: Configure MCP](#step-3-configure-mcp)
  - [Step 4: Working with Your Agent](#step-4-working-with-your-agent)
- [Example: Capstone Project Workflow](#example-capstone-project-workflow)
- [Benefits of This Approach](#benefits-of-this-approach)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Overview

This repository demonstrates a **personal autonomous research agent** setup designed to help students and researchers automate their project workflows. Unlike traditional development where you manually execute every step, this system uses AI agents that:

- **Remember** your project context across sessions (long-term memory)
- **Track** progress and next steps (short-term memory)
- **Execute** tasks using external tools like Kaggle GPUs via MCP
- **Maintain** consistent workflows through structured instructions

**Real-world use case:** This specific setup was used for a VIT Bhopal capstone project on "Lightweight Speech Enhancement Using Shallow Transformers," managing a multi-phase research workflow from baseline models to transformer architectures.

---

## What is OpenClaw & Autonomous Agents?

### OpenClaw

**OpenClaw** is a personal AI assistant platform designed to run locally with multi-channel communication capabilities. Think of it as your own private AI that can:

- Run on your devices (local-first, privacy-focused)
- Communicate across WhatsApp, Telegram, Slack, Discord, and more
- Execute tasks with tools and memory
- Maintain session context and learn from interactions

**Key Concept:** OpenClaw represents the broader movement toward **autonomous agents** — AI systems that can independently plan, execute, and manage tasks with minimal human intervention.

### Autonomous Agents Landscape

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **OpenClaw** | Personal use, multi-channel | Local-first, privacy-focused |
| **CrewAI** | Enterprise automation | Multi-agent crews with unified memory |
| **AutoGen** | Multi-agent systems | Cross-language (.NET + Python) |
| **LangChain** | General-purpose | 1000+ integrations, large ecosystem |
| **OpenAI Agents SDK** | Production apps | Simple primitives, handoffs |

### What Makes This Setup an "Autonomous Agent"?

This repository implements **agentic principles**:

1. **Memory Systems** — Agents remember context across sessions
2. **Tool Use** — Agents can execute code, run notebooks, manage files
3. **Planning** — Structured workflows for complex multi-step tasks
4. **Reflection** — Self-documenting progress and learnings

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS RESEARCH AGENT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   AGENT.md   │  │  memory.md   │  │learningsand  │          │
│  │ Instructions │  │ Long-term    │  │  progress.md │          │
│  │   & Tools    │  │   Memory     │  │ Short-term   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            │                                     │
│                   ┌────────▼────────┐                           │
│                   │  AI Agent       │                           │
│                   │  (GitHub        │                           │
│                   │   Copilot/LLM)  │                           │
│                   └────────┬────────┘                           │
│                            │                                     │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                  │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐         │
│  │  Local      │  │  Kaggle MCP     │  │   Project   │         │
│  │  Jupyter    │  │  (Cloud GPU)    │  │   Files     │         │
│  │  Notebooks  │  │                 │  │ (PDF/PPTX)  │         │
│  └─────────────┘  └─────────────────┘  └─────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## How This Setup Works

### Memory System

The core innovation of this setup is a **dual-memory architecture** inspired by human cognition:

#### Long-Term Memory (`memory.md`)

**Purpose:** Persistent project knowledge that survives across all sessions

**Contains:**
- Project overview and team information
- Architecture specifications and technical details
- Dataset information and Kaggle notebook references
- Phase timelines and milestones
- Known issues and technical gotchas
- Cell maps for notebooks

**Example from this project:**
```markdown
## Architecture Target (Review 2)
- Input: Log-Mel spectrogram (128 mels, n_fft=512, hop=256)
- CNN Encoder: Conv2d 1→64→128→256 (3×3, BN, ReLU)
- Shallow Transformer: 2 layers, 4 heads, d_model=256
- Target PESQ: ≥3.2 | Latency: <15ms

## Kaggle Notebooks
- Baseline CRN: `kjadeja/baseline-crn-speechenhance`
- Review 2 Transformer: to be created
```

**When to update:** After major milestones, architecture changes, or discovering important technical details

#### Short-Term Memory (`learningsandprogress.md`)

**Purpose:** Session-to-session handoff log — what was done, what's next

**Contains:**
- Session logs with dates
- What was accomplished
- What failed or needs attention
- Immediate next steps for next session

**Example:**
```markdown
## Session Log: Feb 23, 2026 — Pre-Kaggle Upload

### What Was Done This Session:
1. ✅ Read `Project2.pdf` — understood Review 2 requirements
2. ✅ Created `review2-transformer-speechenhance.ipynb` with 25 cells
3. ✅ Ran all locally-runnable cells and validated architecture

### Cells NOT Run (Kaggle-only):
- Cell 4 (data extraction): needs `p7zip-full`
- Cell 8 (training loop): needs dataloaders

### IMMEDIATE NEXT STEPS:
1. Save notebook to Kaggle using MCP
2. Run training and verify PESQ ≥ 3.2
```

**When to update:** At the end of every work session or before stopping

### Agent Instructions (`Agent.md`)

**Purpose:** The "operating manual" for the AI agent

**Contains:**
- Agent identity and role context
- Tool descriptions and usage patterns
- Standard workflows (e.g., Kaggle notebook workflow)
- Critical project information (passwords, slugs, paths)
- Technical reminders and constraints

**Example sections:**
```markdown
## WHO YOU ARE
You are a researcher/student AI agent completing a Capstone project at VIT Bhopal.
- Project: "Lightweight Speech Enhancement Using Shallow Transformers"
- Team: Krishnasinh Jadeja, Kirtan Sondagar, Prabhu Kalyan Panda

## MEMORY SYSTEM — READ THESE ON EVERY SESSION START
1. `memory.md` — Long-term project memory
2. `learningsandprogress.md` — Short-term session log
3. This file (`Agent.md`) — Instructions for how to operate

## TOOLS YOU HAVE
- Kaggle MCP tools: `mcp_kaggle_save_notebook`, etc.
- Local notebook: VS Code Jupyter kernel
- File tools: read_file, create_file, grep_search
```

### Kaggle MCP Integration

**MCP (Model Context Protocol)** is an emerging standard for connecting AI applications to external tools — think of it as "USB-C for AI applications."

This setup uses MCP to connect the agent to Kaggle's cloud GPU infrastructure:

```json
{
  "servers": {
    "kaggle": {
      "type": "http",
      "url": "https://www.kaggle.com/mcp"
    }
  }
}
```

**Available MCP Tools:**
- `mcp_kaggle_save_notebook` — Upload and run notebooks
- `mcp_kaggle_get_notebook_session_status` — Monitor training progress
- `mcp_kaggle_download_notebook_output` — Retrieve results and checkpoints

**Workflow:**
1. Develop notebook locally (cells that don't need GPU)
2. Save to Kaggle via MCP
3. Run training on cloud GPU (T4 with 15GB VRAM)
4. Monitor and download results

---

## File Structure

```
kaggle-agent/
│
├── Agent.md                    # Agent instructions and workflows
├── memory.md                   # Long-term project memory
├── learningsandprogress.md     # Short-term session logs
│
├── .vscode/
│   └── mcp.json               # MCP server configuration
│
├── Project2.pptx              # Capstone project proposal
├── Project2.pdf               # PDF version of proposal
│
├── review2-transformer-speechenhance.ipynb    # Main research notebook
├── kaggle_upload.ipynb        # Kaggle-specific notebook version
├── attention_weights.png      # Visualization output
│
├── .venv/                     # Python virtual environment
│   └── ...
│
└── README.md                  # This file
```

---

## How to Use This Setup

### Step 1: Initial Setup

1. **Clone or create your workspace:**
   ```bash
   mkdir my-research-agent
   cd my-research-agent
   ```

2. **Create Python environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchaudio numpy matplotlib tqdm pandas
   pip install pesq==0.0.4 pystoi
   ```

### Step 2: Create Memory Files

#### Create `Agent.md`

Use this template and customize for your project:

```markdown
# AGENT INSTRUCTIONS & WORKSPACE GUIDE
_Updated: [Date]_

---

## WHO YOU ARE
You are a researcher/student AI agent completing [PROJECT TYPE] at [INSTITUTION].
- Project: "[Your Project Title]"
- Team: [Names and IDs]
- Guide: [Advisor Name]
- This workspace is YOUR personal area

---

## MEMORY SYSTEM — READ THESE ON EVERY SESSION START
1. **`memory.md`** — Long-term project memory
2. **`learningsandprogress.md`** — Short-term session log  
3. **This file (`Agent.md`)** — Instructions for how to operate

**Always update these files at the end of every session.**

---

## TOOLS YOU HAVE
- **Kaggle MCP tools**: List available tools here
- **Local notebook**: VS Code Jupyter kernel
- **File tools**: read_file, create_file, replace_string_in_file
- **Terminal**: run_in_terminal

---

## STANDARD WORKFLOW
1. Read `memory.md` and `learningsandprogress.md` first
2. [Your specific workflow steps]
3. Update memory files with results before ending session

---

## PROJECT TIMELINE
| Phase | Date | Target | Status |
|---|---|---|---|
| Phase 1 | [Date] | [Target] | [Status] |
| Phase 2 | [Date] | [Target] | [Status] |

---

## IMPORTANT: DO NOT LOSE THESE
- **Kaggle username**: `your_username`
- **Dataset slug**: `username/dataset-name`
- **Target notebook slug**: `username/notebook-name`

---

## KEY TECHNICAL REMINDERS
- [Important technical notes]
- [Known bugs and workarounds]
- [Critical parameters]
```

#### Create `memory.md`

```markdown
_Use this proactively for long term memory — LAST UPDATED: [Date]_

---

## Project: [Your Project Title]

**Team:** [Names]  
**Guide:** [Advisor]  
**Institution:** [Your Institution]

## Architecture/Methodology
- [Technical specifications]
- [Models, algorithms, approaches]

## Dataset
- Source, size, format
- Preprocessing steps

## Phase Timeline
| Phase | Date | Target |
|---|---|---|
| Phase 1 | [Date] | [Target] |

## Tools & Stack
- [List your tools]

## [Your Notebook Name]
**Local file:** `path/to/notebook.ipynb`  
**Kaggle target slug:** `username/notebook-slug`  
**Status:** [Current status]

### Cell Map:
| # | Type | Name | Status |
|---|---|---|---|
| 1 | MD | Title | — |
| 2 | Code | Imports | ✅ |

### Key Technical Notes:
- [Important implementation details]
- [Known issues]

### Next Steps:
1. [Next action items]
```

#### Create `learningsandprogress.md`

```markdown
_Use this proactively for short term memory — LAST UPDATED: [Date]_

---

## Session Log: [Date] — [Session Title]

### What Was Done This Session:
1. ✅ [Accomplishment 1]
2. ✅ [Accomplishment 2]
3. ⚠️ [Partial completion]

### What's Blocked/Issues:
- [Problems encountered]

### IMMEDIATE NEXT STEPS:
1. [Next action]
2. [Next action]
```

### Step 3: Configure MCP

Create `.vscode/mcp.json`:

```json
{
  "servers": {
    "kaggle": {
      "type": "http",
      "url": "https://www.kaggle.com/mcp"
    }
  }
}
```

**Note:** You'll need Kaggle API credentials (`kaggle.json`) in your home directory for MCP to work.

### Step 4: Working with Your Agent

#### Starting a Session

1. **Read the memory files first** — The agent should always start by reading:
   - `memory.md` — Understand the project state
   - `learningsandprogress.md` — Know what to do next
   - `Agent.md` — Understand available tools and workflows

2. **Execute the workflow** — Follow the standard workflow in Agent.md

3. **Update memories** — Before ending:
   - Update `learningsandprogress.md` with what was done
   - Update `memory.md` if anything fundamental changed

#### Example Interaction

**You:** "Continue working on the Review 2 notebook"

**Agent thinks:**
1. Read `memory.md` — I see we're working on speech enhancement, Review 2 phase, targeting PESQ ≥ 3.2
2. Read `learningsandprogress.md` — Last session created notebook, validated locally, need to upload to Kaggle
3. Read `Agent.md` — I have Kaggle MCP tools available
4. **Action:** Save notebook to Kaggle using MCP

**Agent does:**
- Reads local notebook file
- Calls `mcp_kaggle_save_notebook` with correct parameters
- Monitors session status
- Downloads results when complete
- Updates `learningsandprogress.md` with results

---

## Example: Capstone Project Workflow

This repository contains a real capstone project. Here's how the workflow progressed:

### Phase 1: Baseline (Review 1)
- Created CRN (Convolutional Recurrent Network) baseline
- Trained on Kaggle T4 GPU
- Achieved PESQ ~3.1
- **Memory updated:** Baseline results, dataset info, CRN architecture

### Phase 2: Transformer (Review 2)
- Read memory — knew we needed to beat PESQ 3.1
- Designed ShallowTransformerEnhancer (2.45M params)
- Created notebook with 25 cells
- Validated locally (cells that don't need GPU)
- Uploaded to Kaggle via MCP
- Ran training on cloud GPU
- **Memory updated:** Transformer architecture, training config, next steps

### Phase 3: VAD Integration (Review 3 - Upcoming)
- Read memory — will know to integrate SileroVAD
- Plan based on previous phases

### Phase 4: Deployment (Final)
- Quantization and Gradio demo
- Complete documentation

---

## Benefits of This Approach

### 1. **Context Persistence**
Unlike typical AI chats where context is lost, your agent remembers:
- Project architecture
- Dataset locations
- Previous results
- Technical constraints

### 2. **Session Continuity**
Work for 2 hours, stop, resume next week — the agent knows exactly where to pick up

### 3. **Reduced Cognitive Load**
Don't remember Kaggle slugs, dataset paths, or hyperparameters — they're in memory

### 4. **Reproducibility**
Memory files serve as living documentation of your research process

### 5. **Scalability**
Easy to add new tools (MCP servers), workflows, or project phases

### 6. **Collaboration**
Team members can read memory files to understand project state

---

## Troubleshooting

### Issue: Agent doesn't read memory files
**Solution:** Explicitly tell the agent: "Read memory.md, learningsandprogress.md, and Agent.md first"

### Issue: Kaggle MCP not working
**Solutions:**
- Verify `kaggle.json` is in `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows)
- Check MCP server URL in `.vscode/mcp.json`
- Ensure you have Kaggle API access

### Issue: Memory files get too long
**Solutions:**
- Archive old sessions in `learningsandprogress.md` to a separate file
- Use collapsible sections in markdown
- Summarize old phases in `memory.md`

### Issue: Agent forgets critical information
**Solution:** Add it to `Agent.md` under "IMPORTANT: DO NOT LOSE THESE"

---

## Resources

### Autonomous Agent Frameworks
- [OpenClaw](https://github.com/openclaw) — Personal multi-channel AI assistant
- [CrewAI](https://crewai.com) — Multi-agent automation framework
- [AutoGen](https://microsoft.github.io/autogen/) — Microsoft's multi-agent framework
- [LangChain](https://langchain.com) — General-purpose LLM framework
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents) — Production agent framework

### Model Context Protocol (MCP)
- [MCP Specification](https://modelcontextprotocol.io) — Official documentation
- [MCP Servers](https://github.com/modelcontextprotocol/servers) — Community servers

### This Project's Stack
- [PyTorch](https://pytorch.org) — Deep learning framework
- [Kaggle](https://kaggle.com) — Cloud GPU notebooks
- [PESQ](https://github.com/ludlows/python-pesq) — Speech quality metric
- [STOI](https://github.com/mpariente/pystoi) — Speech intelligibility metric

---

## License

This setup is provided as a template for educational and research purposes. Customize it for your own projects!

---

## Acknowledgments

This setup was created for the capstone project "Lightweight Speech Enhancement Using Shallow Transformers" at VIT Bhopal.

**Team:**
- Krishnasinh Jadeja (22BLC1211)
- Kirtan Sondagar (22BLC1228)
- Prabhu Kalyan Panda (22BLC1213)

**Guide:** Dr. Praveen Jaraut

---

*Happy researching with your autonomous agent! 🚀*
