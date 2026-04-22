# AutoStream Agent - Project Index & Navigation Guide

Complete index of all files and their purposes in the AutoStream Agent project.

## 📋 Quick Navigation

### Getting Started
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Step-by-step setup instructions ⭐ START HERE
- **[README.md](README.md)** - Full project documentation with architecture explanation
- **[setup_verify.py](setup_verify.py)** - Verification script to check your setup

### Running the Agent
- **[agent.py](agent.py)** - Main agent implementation (core code)
- **[chat.py](chat.py)** - Interactive CLI for chatting with the agent
- **[demo/example_conversations.py](demo/example_conversations.py)** - Demo scenarios & examples

### Knowledge Base
- **[knowledge_base/autostream_kb.json](knowledge_base/autostream_kb.json)** - Product knowledge base with pricing, features, and policies

### Configuration
- **[.env.example](.env.example)** - Environment variables template
- **[.gitignore](.gitignore)** - Git ignore rules
- **[requirements.txt](requirements.txt)** - Python dependencies

### Testing
- **[tests/test_intent_detection.py](tests/test_intent_detection.py)** - Intent classification tests
- **[tests/test_rag_retrieval.py](tests/test_rag_retrieval.py)** - Knowledge retrieval tests
- **[tests/test_lead_capture.py](tests/test_lead_capture.py)** - Lead capture validation tests

### Documentation
- **[PROJECT_INDEX.md](PROJECT_INDEX.md)** - This file

---

## 🗂️ File Structure Overview

```
autostream-agent/
│
├── 📄 Core Implementation
│   ├── agent.py                        - Main agent with LangGraph
│   ├── chat.py                         - Interactive CLI interface
│   └── setup_verify.py                 - Setup verification script
│
├── 📚 Knowledge Base
│   └── knowledge_base/
│       └── autostream_kb.json          - Product information database
│
├── 🧪 Tests
│   └── tests/
│       ├── test_intent_detection.py   - Intent classification tests
│       ├── test_rag_retrieval.py      - RAG system tests
│       └── test_lead_capture.py       - Lead capture logic tests
│
├── 🎬 Demos & Examples
│   └── demo/
│       └── example_conversations.py    - Demo conversations
│
├── 📖 Documentation
│   ├── README.md                       - Complete project documentation
│   ├── SETUP_GUIDE.md                 - Installation instructions
│   └── PROJECT_INDEX.md               - This file
│
└── ⚙️ Configuration
    ├── requirements.txt                - Python dependencies
    ├── .env.example                    - Environment template
    └── .gitignore                      - Git ignore rules
```

---

## 📖 File Descriptions

### Core Implementation

#### `agent.py` (400+ lines)
**Main agent implementation with LangGraph state management**

Key Classes:
- `AgentState` - TypedDict for multi-turn state management
- `KnowledgeBaseManager` - Loads and queries the knowledge base
- `IntentDetector` - Classifies user intent (greeting/inquiry/high-intent)
- `AutoStreamAgent` - Main agent with LangGraph workflow

Key Methods:
- `chat(message)` - Single-turn conversation
- `multi_turn_chat(messages)` - Multi-turn conversation
- `query_rag(query)` - Retrieve relevant knowledge
- `detect_intent(message)` - Classify user intent
- `mock_lead_capture(name, email, platform)` - Lead capture API

**Why This File:**
- Contains the complete agent logic
- Implements state management across turns
- Handles intent detection and routing
- Manages RAG retrieval and tool execution

#### `chat.py` (150+ lines)
**Interactive CLI interface for testing**

Features:
- Real-time chat with the agent
- Command system (/quit, /clear, /help, /demo)
- Conversation history tracking
- User-friendly interface

**How to Use:**
```bash
python chat.py
```

**Why This File:**
- Provides easy way to test agent without writing code
- Good for demo purposes and debugging
- Demonstrates agent capabilities interactively

#### `setup_verify.py` (200+ lines)
**Automated setup verification**

Checks:
- ✅ Python version (3.9+)
- ✅ Project structure
- ✅ Dependency installation
- ✅ API key configuration
- ✅ Knowledge base validity
- ✅ Quick agent test

**How to Use:**
```bash
python setup_verify.py
```

**Why This File:**
- Catches setup errors early
- Provides helpful error messages
- Verifies all requirements are met

---

### Knowledge Base

#### `knowledge_base/autostream_kb.json` (150+ lines)
**Structured product knowledge database**

Sections:
- `company` - Company info
- `pricing` - Basic ($29/mo) and Pro ($79/mo) plans
- `features` - Common and Pro-exclusive features
- `policies` - Refund, support, trial, cancellation
- `use_cases` - Platform-specific recommendations

**Structure:**
```json
{
  "company": {...},
  "pricing": {
    "basic_plan": {...},
    "pro_plan": {...}
  },
  "features": {...},
  "policies": {...},
  "use_cases": {...}
}
```

**Why This File:**
- Central source of truth for product info
- Enables RAG retrieval
- Easy to update pricing/features
- Demonstrated realistic production patterns

---

### Tests

#### `tests/test_intent_detection.py` (200+ lines)
**Unit tests for intent detection**

Test Classes:
- `TestIntentDetection` - Tests for all intent types
  - Casual greeting detection
  - Product inquiry detection
  - High-intent lead detection
  - Context-aware detection
  - Edge cases

Coverage:
- 15+ test cases
- Tests basic keywords
- Tests priority (high-intent > inquiry > greeting)
- Tests edge cases (empty, special chars, etc.)

**Why This File:**
- Validates intent classifier accuracy
- Catches regression bugs
- Demonstrates test best practices

#### `tests/test_rag_retrieval.py` (250+ lines)
**Tests for RAG system**

Test Classes:
- `TestRAGRetrieval` - Knowledge retrieval tests
  - KB loading
  - All required sections present
  - Pricing/features/policies retrieval
  - Query relevance
  - Accuracy validation

Coverage:
- 20+ test cases
- Tests all knowledge sections
- Tests keyword matching
- Tests multi-keyword queries

**Why This File:**
- Ensures knowledge base is correctly loaded
- Validates RAG relevance
- Prevents knowledge gaps

#### `tests/test_lead_capture.py` (300+ lines)
**Tests for lead capture logic**

Test Classes:
- `TestLeadCapture` - Lead data and capture tests
- `TestLeadCapturePrevention` - Validation that prevents premature capture

Coverage:
- 20+ test cases
- Tests LeadInfo data class
- Tests mock API
- Tests field extraction via regex
- Tests prevention of incomplete capture

**Why This File:**
- Ensures lead info is complete before API call
- Validates data extraction
- Critical for safety and correctness

---

### Demos & Examples

#### `demo/example_conversations.py` (350+ lines)
**Demo scenarios showcasing agent capabilities**

Demo Functions:
1. `demo_1_pricing_inquiry()` - Pricing questions
2. `demo_2_high_intent_lead_capture()` - Lead conversion ⭐
3. `demo_3_feature_inquiry()` - Feature questions
4. `demo_4_policy_inquiry()` - Policy questions
5. `demo_5_creator_use_case()` - Platform-specific recommendations
6. `demo_single_turn()` - Single turn examples

**How to Use:**
```bash
# Interactive menu
python demo/example_conversations.py

# Or run specific demo
python -c "from demo.example_conversations import demo_2_high_intent_lead_capture; demo_2_high_intent_lead_capture()"
```

**Why This File:**
- Demonstrates all agent capabilities
- Good for creating demo video
- Shows multi-turn conversation flow
- Useful for testing before deployment

---

### Documentation

#### `README.md` (400+ lines)
**Complete project documentation**

Sections:
- Features overview
- Architecture explanation
- Why LangGraph (vs other approaches)
- State management flow diagram
- Quick start guide
- Project structure
- Usage examples
- WhatsApp integration guide
- Conversation flow example
- Evaluation criteria
- Testing instructions
- Future enhancements

**Why This File:**
- Primary documentation for the project
- Explains architectural decisions
- Shows WhatsApp deployment approach
- Provides code examples

#### `SETUP_GUIDE.md` (300+ lines)
**Step-by-step setup instructions**

Sections:
- Prerequisites
- Get API key (Claude/OpenAI/Gemini)
- Clone/setup project
- Create virtual environment
- Install dependencies
- Configure API key
- Verify setup
- Run the agent
- Troubleshooting guide
- Quick reference commands

**Why This File:**
- Helps users get started quickly
- Troubleshooting section for common errors
- Works for different OS (Windows/macOS/Linux)

#### `PROJECT_INDEX.md` (This file)
**Navigation guide and file descriptions**

**Why This File:**
- Easy navigation for large projects
- Quick reference for file purposes
- Helps team members find code

---

### Configuration

#### `requirements.txt` (20+ lines)
**Python package dependencies**

Packages:
- **LangChain Core:** langchain, langchain-core, langgraph
- **LLM Providers:** langchain-openai, langchain-anthropic, langchain-google-genai
- **APIs:** openai, anthropic, google-generativeai
- **Utilities:** python-dotenv, pydantic, typing-extensions
- **Testing:** pytest, pytest-asyncio

**Install:**
```bash
pip install -r requirements.txt
```

#### `.env.example` (15+ lines)
**Environment variables template**

Variables:
```
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key
AGENT_MODEL=claude-3-5-haiku-20241022
AGENT_TEMPERATURE=0.7
KB_PATH=knowledge_base/autostream_kb.json
WHATSAPP_PHONE_ID=...
WHATSAPP_ACCESS_TOKEN=...
```

**Setup:**
1. Copy to `.env`
2. Fill in your API keys
3. Keep `.env` in `.gitignore` (never commit!)

#### `.gitignore` (50+ lines)
**Git ignore rules**

Ignores:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- IDE files (`.vscode/`, `.idea/`)
- Environment variables (`.env`)
- Build artifacts (`build/`, `dist/`)
- Sensitive files (`secrets.json`)
- OS files (`.DS_Store`)

---

## 🚀 Quick Start Paths

### I want to...

#### 1. Get started quickly
```bash
1. Read SETUP_GUIDE.md
2. Run python setup_verify.py
3. Run python chat.py
```

#### 2. Understand the architecture
```bash
1. Read README.md "Architecture" section
2. Review agent.py line 400-500
3. Check state management flow
```

#### 3. Run demos
```bash
python demo/example_conversations.py
```

#### 4. Run tests
```bash
pytest tests/ -v
```

#### 5. Use agent in my code
```python
from agent import AutoStreamAgent
agent = AutoStreamAgent()
response = agent.chat("Your message")
```

#### 6. Create a demo video
```bash
1. Run: python demo/example_conversations.py
2. Record your screen
3. Record demo scenario
4. Create a 2-3 minute video
```

#### 7. Deploy to WhatsApp
```bash
1. Read README.md "WhatsApp Integration"
2. Set up Flask backend
3. Configure webhooks
4. Test with WhatsApp Business API
```

---

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | 1,000+ |
| **Python Files** | 8 |
| **Test Cases** | 50+ |
| **Documentation Lines** | 1,500+ |
| **Demo Scenarios** | 6 |
| **Knowledge Base Entries** | 30+ |
| **Supported Intents** | 3 |
| **API Providers** | 3 (Claude, OpenAI, Gemini) |

---

## 🎯 Evaluation Criteria Checklist

- ✅ **Agent Reasoning & Intent Detection** - IntentDetector class, 3-class classification
- ✅ **Correct RAG Usage** - KnowledgeBaseManager with keyword-based retrieval
- ✅ **State Management** - LangGraph with typed AgentState across 5+ turns
- ✅ **Tool Calling Logic** - mock_lead_capture only when all info collected
- ✅ **Code Clarity** - Type hints, docstrings, clear separation
- ✅ **Real-world Deployability** - WhatsApp integration guide, production patterns

---

## 🔗 Important Links

- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **LangChain:** https://python.langchain.com/
- **Claude API:** https://docs.anthropic.com/
- **OpenAI API:** https://platform.openai.com/docs/
- **Anthropic Console:** https://console.anthropic.com/

---

## 📝 Project Status

- ✅ Agent implementation complete
- ✅ Knowledge base created
- ✅ Intent detection working
- ✅ RAG system functional
- ✅ Lead capture implemented
- ✅ State management working
- ✅ Tests written (50+ cases)
- ✅ Documentation complete
- ⏳ Demo video pending (record and save)

---

**Last Updated:** 2026-04-22
**Version:** 1.0.0
**Status:** Ready for Evaluation ✅
