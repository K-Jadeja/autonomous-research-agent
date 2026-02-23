# Template Quick Start Guide

This folder contains the **actual working files** from a real capstone project. These are not abstract templates — they show exactly how the autonomous agent system works.

## ⚡ Quick Setup (3 Steps)

### Step 1: Copy These Files to Your Project
```bash
# Create your project folder
mkdir my-research-project
cd my-research-project

# Copy the core memory system files
cp template/Agent.md .
cp template/memory.md .
cp template/learningsandprogress.md .

# Create MCP config folder
mkdir .vscode
cp template/mcp.json .vscode/mcp.json
```

### Step 2: Add Your Research Proposal
Place your research proposal document in the root folder:
- `Project_Proposal.pdf` or
- `Project_Proposal.pptx` or
- `Research_Idea.docx`

**The agent will read this to understand your project!**

### Step 3: Configure Kaggle MCP
1. Get your Kaggle API credentials from https://www.kaggle.com/settings/account
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)
3. Update `.vscode/mcp.json` with your Kaggle username

---

## 🚀 How to Start Your Agent

### First Time Setup Prompt

After completing the 3 steps above, start a chat with your AI coding assistant (GitHub Copilot, ChatGPT, Claude, etc.) and say:

```
I have set up an autonomous research agent workspace. Please:

1. First, read my research proposal file [Project_Proposal.pdf]
2. Then read Agent.md, memory.md, and learningsandprogress.md
3. Use /mcp auth kaggle to authenticate with Kaggle
4. Check if I have any existing Kaggle notebooks

This is your personal workspace. You are a researcher/student AI agent 
helping me with my [capstone/thesis/research] project.

The memory files are for you to continuously update as you work. Always 
read them at the start of each session and update them before ending.
```

### Example Starting Prompt (From Actual Project)

```
can you see my https://www.kaggle.com/code/kjadeja/baseline-crn-speechenhance 
notebook? try to understand everything about it then now go through the whole 
folder system here. this is your personal area. you are a researcher student 
working on his college project for capstone. notice the md files that are here 
for you to continuously use

you have access to kaggle mcp to download notebooks, and run notebooks.

i want to create a new notebook and always run a session and run the code 
cell by cell to know your code works. then when you know training works, 
then save and run.

you are to read the project 2 pdf or pptx and know about what the project 
is about. 

begin.
```

---

## 📋 What the Agent Will Do

### Phase 1: Understanding
The agent will:
1. **Read your proposal** (PDF/PPTX) to understand the research goal
2. **Read memory files** to see current project state
3. **Analyze existing notebooks** on Kaggle (if any)
4. **Check MCP connection** to Kaggle

### Phase 2: Local Development
The agent will:
1. **Create a new notebook** based on your proposal
2. **Write code cell by cell** (imports, data loading, model architecture)
3. **Run cells locally** to validate they work (on CPU)
4. **Test architecture** with dummy data
5. **Update memory files** with progress

### Phase 3: Cloud Execution
The agent will:
1. **Save notebook to Kaggle** using MCP
2. **Run training** on cloud GPU (T4/V100/A100)
3. **Monitor progress** and check for errors
4. **Download results** (models, metrics, plots)
5. **Update memories** with final results

### Phase 4: Iteration
The agent will:
1. **Analyze results** from memory files
2. **Plan next experiments** based on learnings
3. **Repeat the cycle** until project complete

---

## 📁 File Structure After Setup

```
my-research-project/
│
├── Agent.md                    ← Instructions for the AI agent (CUSTOMIZE THIS)
├── memory.md                   ← Long-term project memory (AGENT UPDATES THIS)
├── learningsandprogress.md     ← Session logs (AGENT UPDATES THIS)
│
├── .vscode/
│   └── mcp.json               ← Kaggle MCP configuration
│
├── Project_Proposal.pdf        ← Your research proposal (ADD THIS)
│
└── [notebook-name].ipynb      ← Created by agent (WILL BE CREATED)
```

---

## 🎓 Example Workflow

### Starting a New Project

**You:** "I want to work on sentiment analysis for movie reviews"

**Agent:**
1. Checks for existing project files → None found
2. Reads Agent.md → "I'm a researcher agent, I should create memory files"
3. Creates initial memory.md with sentiment analysis project structure
4. Creates learningsandprogress.md: "Session 1: Initialize project"
5. Says: "I've set up your workspace. Please add your research proposal PDF or PPTX, then I'll begin development."

**You:** [Add `Sentiment_Analysis_Proposal.pdf`]

**Agent:**
1. Reads the PDF → Extracts: "Use LSTM with attention, IMDB dataset, target 85% accuracy"
2. Updates memory.md with architecture details
3. Creates notebook with:
   - Data loading from Kaggle
   - LSTM model with attention
   - Training loop
4. Runs locally → validates architecture works
5. Uploads to Kaggle
6. Monitors training
7. Downloads model with 87% accuracy
8. Updates memories: "Achieved 87% accuracy, exceeding target!"

### Continuing Work

**You:** (2 weeks later) "Continue working on sentiment analysis"

**Agent:**
1. Reads memory.md → "We're at 87% accuracy, need to try Transformer architecture"
2. Reads learningsandprogress.md → "Last session: Need to implement BERT variant"
3. Implements BERT-based model
4. Uploads to Kaggle
5. Achieves 92% accuracy
6. Updates memories

**No context lost!** The agent remembers everything from 2 weeks ago.

---

## 💡 Tips for Success

### For You (The User)
- **Add your proposal first** — The agent needs context!
- **Be specific** — "Improve the model" is vague. "Try ResNet-50 instead of VGG" is actionable.
- **Let the agent work** — Don't micromanage. Give high-level goals.
- **Review memory files** — Check what the agent documented.

### For The Agent (What to Tell It)
- "Always read memory.md, learningsandprogress.md, and Agent.md first"
- "Run cells locally before uploading to Kaggle"
- "Update memory files at the end of every session"
- "If you encounter an error, document it in learningsandprogress.md"

---

## 🔧 Customization

### Change Project Details
Edit `Agent.md`:
- Update "WHO YOU ARE" section with your project details
- Change team names and IDs
- Update project timeline
- Add your Kaggle username and dataset slugs

### Add More Tools
Edit `.vscode/mcp.json`:
```json
{
  "servers": {
    "kaggle": {
      "type": "http",
      "url": "https://www.kaggle.com/mcp"
    },
    "arxiv": {
      "type": "http", 
      "url": "https://arxiv.org/mcp"
    }
  }
}
```

### Modify Workflows
Edit `Agent.md` → "STANDARD WORKFLOW" section to match your development style.

---

## ⚠️ Common Issues

### "Agent doesn't read my proposal"
**Fix:** Make sure the PDF/PPTX filename is in the prompt. Example: "Read my Project_Proposal.pdf"

### "Kaggle MCP not working"
**Fix:** 
1. Check `kaggle.json` is in the right location
2. Verify Kaggle API is enabled on your account
3. Test with: `/mcp auth kaggle`

### "Agent forgets what we were doing"
**Fix:** Tell the agent: "Read memory.md and learningsandprogress.md first" at the start of each session.

---

## 📚 Resources

### Files Explained

**Agent.md** — The agent's "brain" — contains:
- Who the agent is (student/researcher identity)
- What tools it has access to
- Standard workflows
- Project timeline
- Critical information (API keys, paths, etc.)

**memory.md** — Long-term memory:
- Project overview
- Architecture details
- Dataset information
- Results and metrics
- Known issues and solutions

**learningsandprogress.md** — Session log:
- What was done in each session
- What worked and what didn't
- Next steps for the next session
- Handoff notes for continuity

---

## 🎯 Next Steps

1. ✅ Copy template files to your project
2. ✅ Add your research proposal (PDF/PPTX)
3. ✅ Configure Kaggle MCP
4. ✅ Start chat with your AI assistant
5. ✅ Give the starting prompt (see above)
6. ✅ Watch your agent work!

---

**Remember:** These templates show a **real working system** that completed an actual capstone project. The agent successfully:
- Read PowerPoint proposals
- Wrote thousands of lines of code
- Ran experiments on Kaggle GPUs
- Achieved research objectives
- Maintained perfect context over 3 months

**You can do this too!** 🚀
