# AutoStream Agent - Setup Guide

Complete step-by-step instructions to get the AutoStream Agent running on your machine.

## Prerequisites

- **Python 3.9+** installed ([download here](https://www.python.org/downloads/))
- **pip** (comes with Python)
- An API key from one of:
  - Anthropic (Claude) - **Recommended**
  - OpenAI (GPT-4o-mini)
  - Google (Gemini 1.5 Flash)

## Step 1: Get an API Key

### Option A: Using Claude (Recommended)

1. Go to [Anthropic Console](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to **API Keys**
4. Click **Create Key**
5. Copy your API key

### Option B: Using OpenAI

1. Go to [OpenAI Platform](https://platform.openai.com)
2. Sign up or log in
3. Navigate to **API Keys**
4. Click **Create new secret key**
5. Copy your API key

### Option C: Using Google Gemini

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy your API key

## Step 2: Clone/Setup Project

```bash
# Navigate to the project directory
cd autostream-agent

# Verify the project structure
# You should see: agent.py, requirements.txt, knowledge_base/, tests/, demo/, etc.
```

## Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

## Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep langchain
# or
pip show langchain
```

## Step 5: Configure API Key

### Windows (Command Prompt)
```cmd
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

### Windows (PowerShell)
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxx"
```

### macOS/Linux (Bash)
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxx"
```

### Using .env File (Recommended for Development)

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
# or
copy .env.example .env  # Windows
```

2. Edit `.env` and add your API key:
```
ANTHROPIC_API_KEY=your-actual-api-key-here
```

3. Install python-dotenv (if not already installed):
```bash
pip install python-dotenv
```

## Step 6: Verify Setup

Run the verification script:

```bash
python setup_verify.py
```

This will check:
- ✅ Python version
- ✅ Project structure
- ✅ Dependencies installation
- ✅ API key configuration
- ✅ Knowledge base validity
- ✅ Quick agent test

Expected output:
```
✅ PASS: Python Version
✅ PASS: Project Structure
✅ PASS: Dependencies
✅ PASS: API Configuration
✅ PASS: Knowledge Base
✅ PASS: Quick Test

Result: 6/6 checks passed
🎉 All checks passed! You're ready to use AutoStream Agent!
```

## Step 7: Run the Agent

### Option A: Run Demo

```bash
# Interactive demo with multiple scenarios
python demo/example_conversations.py

# Or run the main agent
python agent.py
```

### Option B: Use as Python Module

```python
from agent import AutoStreamAgent

# Initialize agent
agent = AutoStreamAgent()

# Single turn conversation
response = agent.chat("Tell me about your pricing")
print(response)

# Multi-turn conversation
messages = [
    "Hi!",
    "What's your pricing?",
    "I want to sign up",
    "My name is John",
    "john@example.com",
    "YouTube"
]

conversation = agent.multi_turn_chat(messages)
for exchange in conversation:
    print(f"{exchange['role']}: {exchange['content']}")
```

### Option C: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_intent_detection.py -v
pytest tests/test_rag_retrieval.py -v
pytest tests/test_lead_capture.py -v

# Run with coverage
pytest tests/ --cov=agent
```

## Troubleshooting

### Error: "ANTHROPIC_API_KEY not set"

**Solution:**
1. Verify you set the environment variable correctly
2. Restart your terminal/IDE after setting the variable
3. Use the `.env` file method

Test if variable is set:
```bash
# Windows
echo %ANTHROPIC_API_KEY%

# macOS/Linux
echo $ANTHROPIC_API_KEY
```

### Error: "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Then install dependencies
pip install -r requirements.txt
```

### Error: "FileNotFoundError: knowledge_base/autostream_kb.json"

**Solution:**
Ensure you're running the script from the project root directory:
```bash
# Wrong
cd autostream-agent/demo
python example_conversations.py

# Correct
cd autostream-agent
python demo/example_conversations.py
```

### Error: "LangGraph not found"

**Solution:**
```bash
# Update langchain and langgraph
pip install --upgrade langchain langgraph langchain-core
```

### API Rate Limiting

If you get rate limit errors:
- Add delays between requests
- Use a different API provider
- Check your API key's usage limits in the console

## Next Steps

1. **Explore the Code:**
   - Read [README.md](README.md) for architecture details
   - Check [agent.py](agent.py) to understand the implementation
   - Review [knowledge_base/autostream_kb.json](knowledge_base/autostream_kb.json)

2. **Try Examples:**
   - Run the demo: `python demo/example_conversations.py`
   - Modify conversations in `demo/example_conversations.py`

3. **Run Tests:**
   - `pytest tests/ -v` to run all tests
   - Tests cover intent detection, RAG, and lead capture

4. **Integrate with WhatsApp:**
   - See README.md section: "WhatsApp Integration"
   - Use the provided Flask webhook example

5. **Customize:**
   - Modify knowledge base in `knowledge_base/autostream_kb.json`
   - Add new intents in `IntentDetector` class
   - Extend RAG with semantic search

## Project Structure Reference

```
autostream-agent/
├── agent.py                    # Main agent implementation
├── setup_verify.py            # Setup verification script
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── SETUP_GUIDE.md            # This file
├── .env.example              # Environment template
│
├── knowledge_base/
│   └── autostream_kb.json    # Product knowledge base
│
├── tests/
│   ├── test_intent_detection.py
│   ├── test_rag_retrieval.py
│   └── test_lead_capture.py
│
└── demo/
    └── example_conversations.py  # Demo scenarios
```

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key"  # macOS/Linux
set ANTHROPIC_API_KEY=your-key       # Windows

# Verify setup
python setup_verify.py

# Run agent
python agent.py

# Run demo
python demo/example_conversations.py

# Run tests
pytest tests/ -v

# Deactivate virtual environment
deactivate
```

## Support & Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangChain Docs:** https://python.langchain.com/
- **Claude API:** https://docs.anthropic.com/
- **OpenAI API:** https://platform.openai.com/docs/
- **Anthropic Console:** https://console.anthropic.com/

## Next: Create Demo Video

Once everything is set up, you can create your demo video:

```bash
# Run the agent and record your screen
python demo/example_conversations.py

# Narrate:
# 1. Show agent answering pricing question
# 2. Show intent detection triggering lead flow
# 3. Show information collection
# 4. Show successful lead capture
```

---

**Ready to go?** Run: `python setup_verify.py` 🚀
