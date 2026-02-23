# Template Quick Start Guide

This folder contains starter templates for setting up your own autonomous research agent workspace.

## Files Included

1. **Agent.md** — Instructions for your AI agent (who it is, what tools it has, workflows)
2. **memory.md** — Long-term project memory (architecture, datasets, results)
3. **learningsandprogress.md** — Short-term session tracking (what was done, what's next)
4. **mcp.json** — MCP (Model Context Protocol) configuration for external tools

## How to Use These Templates

### Step 1: Copy Files to Your Project
```bash
# Create your project folder
mkdir my-research-project
cd my-research-project

# Copy template files
cp template/Agent.md .
cp template/memory.md .
cp template/learningsandprogress.md .
cp template/mcp.json .vscode/mcp.json  # If using VS Code
```

### Step 2: Customize Agent.md
Fill in the bracketed placeholders [LIKE THIS] with your project details:
- Project title and team info
- Institution and advisor
- Available tools (customize based on what you have)
- Project timeline
- Important credentials and paths
- Technical reminders

### Step 3: Initialize memory.md
Set up your long-term memory with:
- Project overview and objectives
- Architecture or methodology details
- Dataset information
- Phase timeline
- Empty results section (to be filled as you progress)

### Step 4: Start learningsandprogress.md
Begin tracking sessions:
- Date and focus of first session
- What you plan to accomplish
- Leave next steps blank (to fill at end of session)

### Step 5: Configure MCP
Set up `.vscode/mcp.json` (if using Kaggle or other MCP servers):
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

## Tips for Success

### For Agent.md
- Be specific about workflows — the better the instructions, the better the agent performs
- Update "IMPORTANT: DO NOT LOSE THESE" section as you discover critical info
- Include concrete examples (like Kaggle workflow example in the template)

### For memory.md
- Update after major milestones, not every small change
- Include cell maps for notebooks — helps track progress
- Document "Known Gotchas" — save future you from rediscovering issues
- Keep architecture summaries visual (ASCII diagrams help!)

### For learningsandprogress.md
- Update at the end of EVERY session, even if short
- Be honest about failures — helps diagnose patterns
- Prioritize next steps (P0/P1/P2) — guides the agent's focus
- Archive old sessions periodically to keep the file manageable

### General
- **Consistency is key:** Use the same format across sessions
- **Be explicit:** Don't assume the agent remembers — that's what memory files are for
- **Version control:** Consider git-tracking these files to see evolution
- **Backups:** Don't rely solely on one location — sync to cloud

## Example Workflow

### Starting a New Project

**You:** "Start a new project on sentiment analysis"

**Agent:**
1. Reads templates
2. Creates customized files:
   - Agent.md → "You are a researcher working on sentiment analysis..."
   - memory.md → Project overview, IMDB dataset info, LSTM architecture
   - learningsandprogress.md → "Session 1: Setup project structure"
3. Sets up folder structure
4. Creates initial notebook

### Continuing Work

**You:** "Continue working on sentiment analysis"

**Agent:**
1. Reads memory.md → "We're using LSTM, targeting 85% accuracy"
2. Reads learningsandprogress.md → "Last session: Data preprocessing done. Next: Build model"
3. Reads Agent.md → "I have Kaggle MCP tools available"
4. **Action:** Creates LSTM model notebook
5. **Updates:** learningsandprogress.md with results

## Common Patterns

### ML/DL Projects
- memory.md → Include model architecture, hyperparameters, dataset details
- Track experiments in learningsandprogress.md
- Use Kaggle MCP for GPU training

### Research Papers
- memory.md → Literature review, methodology, key papers
- Track writing progress in learningsandprogress.md
- Use file tools to manage LaTeX/document files

### Data Analysis
- memory.md → Data sources, cleaning steps, analysis plan
- Track insights and visualizations in learningsandprogress.md
- Use notebooks for iterative exploration

## Customization Ideas

### Add More Memory Files
- **literature_review.md** — Track papers and notes
- **experiments.md** — Detailed experiment logs
- **meeting_notes.md** — Advisor meeting summaries

### Add MCP Servers
```json
{
  "servers": {
    "kaggle": { "type": "http", "url": "https://www.kaggle.com/mcp" },
    "arxiv": { "type": "http", "url": "https://arxiv.org/mcp" },
    "github": { "type": "http", "url": "https://api.github.com/mcp" }
  }
}
```

### Custom Workflows
Add specific workflows to Agent.md:
- "Paper Writing Workflow"
- "Experiment Tracking Workflow"
- "Code Review Workflow"

## Need Help?

- Check the main README.md for detailed explanations
- Look at the example files in the repository root
- Refer to the "Troubleshooting" section in README.md

---

**Ready to start?** Copy these templates and customize for your project!
