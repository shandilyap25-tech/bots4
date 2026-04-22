# AutoStream Agent - Quick Reference Card

## 🚀 Quick Start (Choose One)

### Option 1: Interactive Chat
```bash
python chat.py
```
Type messages naturally. Commands: `/help`, `/quit`, `/demo`

### Option 2: Run Demo
```bash
python demo/example_conversations.py
```
6 pre-built scenarios showing different agent capabilities

### Option 3: Use in Code
```python
from agent import AutoStreamAgent
agent = AutoStreamAgent()
response = agent.chat("Tell me about pricing")
```

---

## 📋 Setup Checklist

```bash
1. Read SETUP_GUIDE.md
2. python setup_verify.py          # Verify setup
3. python chat.py                  # Start chatting!
```

**Env Variable Needed:**
- `ANTHROPIC_API_KEY` (Claude) OR
- `OPENAI_API_KEY` (OpenAI) OR
- `GOOGLE_API_KEY` (Gemini)

---

## 🤖 Agent Components

| Component | File | Purpose |
|-----------|------|---------|
| **Agent** | agent.py:AutoStreamAgent | Main orchestrator |
| **Intent Detector** | agent.py:IntentDetector | Classifies user intent |
| **RAG System** | agent.py:KnowledgeBaseManager | Retrieves knowledge |
| **Lead Tool** | agent.py:mock_lead_capture() | Captures leads |
| **State Manager** | agent.py:AgentState | Tracks conversation |

---

## 🎯 Intent Classes

| Intent | Keywords | Action |
|--------|----------|--------|
| **Greeting** | hi, hello, thanks | Respond naturally |
| **Inquiry** | price, feature, plan | Retrieve from KB |
| **High-Intent** | sign up, want to try | Start lead capture |

---

## 📝 Lead Capture Flow

```
High-Intent Detected
    ↓
Ask for Name
    ↓
Ask for Email
    ↓
Ask for Platform
    ↓
All Complete? → mock_lead_capture() → Success! 🎉
```

**Lead requires:** Name + Email + Platform (all 3)

---

## 🗄️ Knowledge Base

Located: `knowledge_base/autostream_kb.json`

Sections:
- **Pricing:** Basic $29/mo (720p), Pro $79/mo (4K)
- **Features:** Editing, AI captions, background removal
- **Policies:** 7-day refund, 24/7 support (Pro), 7-day trial
- **Use Cases:** YouTube, Instagram, TikTok, Business, Education

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Individual test files
pytest tests/test_intent_detection.py -v
pytest tests/test_rag_retrieval.py -v
pytest tests/test_lead_capture.py -v

# With coverage
pytest tests/ --cov=agent
```

**Coverage:** 50+ tests covering:
- Intent detection accuracy
- RAG relevance
- Lead validation
- Edge cases

---

## 📱 WhatsApp Integration (Overview)

See README.md for full details. Quick summary:

```
WhatsApp Message
    ↓
[Your Backend]
    ├─ Webhook listener
    ├─ Parse message
    ├─ Call agent.chat()
    ├─ Send response back
    ↓
WhatsApp User
```

**Setup Required:**
1. WhatsApp Business Account
2. Backend with Flask/FastAPI
3. Webhook URL configuration
4. Message handling logic

---

## 💡 Code Examples

### Single Turn
```python
from agent import AutoStreamAgent

agent = AutoStreamAgent()
response = agent.chat("What's your pricing?")
print(response)
```

### Multi-Turn
```python
messages = [
    "Hi there",
    "Tell me about Pro plan",
    "I want to sign up",
    "My name is John",
    "john@email.com",
    "YouTube"
]

conversation = agent.multi_turn_chat(messages)
for msg in conversation:
    print(f"{msg['role']}: {msg['content']}")
```

### Direct Components
```python
# Just intent detection
from agent import IntentDetector
detector = IntentDetector()
intent = detector.detect_intent("I want to sign up", [])
# Returns: "high_intent_lead"

# Just RAG retrieval
from agent import KnowledgeBaseManager
kb = KnowledgeBaseManager()
pricing = kb.get_pricing_info()
print(pricing)

# Just lead capture
from agent import mock_lead_capture
result = mock_lead_capture("John", "john@email.com", "YouTube")
print(result["lead_id"])
```

---

## 📁 File Locations

| What | Where |
|------|-------|
| Main agent | agent.py |
| Interactive chat | chat.py |
| Tests | tests/ |
| Demos | demo/example_conversations.py |
| Knowledge | knowledge_base/autostream_kb.json |
| Setup help | SETUP_GUIDE.md |
| Full docs | README.md |

---

## ⚙️ Configuration

**API Keys (pick one):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Claude (recommended)
export OPENAI_API_KEY="sk-..."            # OpenAI
export GOOGLE_API_KEY="..."               # Gemini
```

**Optional Settings in `.env`:**
```
AGENT_MODEL=claude-3-5-haiku-20241022
AGENT_TEMPERATURE=0.7
KB_PATH=knowledge_base/autostream_kb.json
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not set" | Run: `export ANTHROPIC_API_KEY="your-key"` |
| "Module not found" | Run: `pip install -r requirements.txt` |
| "KB not found" | Run from project root: `cd autostream-agent` |
| "Connection error" | Check internet, verify API key in console |

Run `python setup_verify.py` to diagnose issues.

---

## 📊 Architecture Decision: Why LangGraph?

| Feature | LangChain | **LangGraph** ✅ |
|---------|-----------|-----------------|
| State Management | Limited | Full typed state |
| Conditional Routing | Manual | Built-in edges |
| Tool Execution Control | Automatic | Explicit conditions |
| Multi-turn State | Via memory | Native support |
| Reliability | Good | Better for agents |

→ LangGraph enables preventing premature lead capture!

---

## 🎬 Demo Video Checklist

Record this 2-3 minute video:

1. **Pricing Question (30 sec)**
   - User asks about pricing
   - Show RAG retrieval
   - Agent responds with accurate prices

2. **Intent Detection (30 sec)**
   - User shows interest in Pro plan
   - Show agent detecting high-intent
   - Agent asks for information

3. **Lead Collection (30 sec)**
   - User provides name
   - User provides email
   - User provides platform
   - Show agent collecting all info

4. **Lead Capture (30 sec)**
   - Show mock_lead_capture() executing
   - Display lead ID and confirmation
   - Show success message

**How to Record:**
```bash
python demo/example_conversations.py
# Select demo 2 (High-Intent Lead Capture)
# Record your screen with audio narration
```

---

## ✅ Before Submitting

- [ ] Run `python setup_verify.py` - all pass ✅
- [ ] Run `pytest tests/ -v` - all pass ✅
- [ ] Run `python demo/example_conversations.py` - works ✅
- [ ] README.md complete with architecture ✅
- [ ] SETUP_GUIDE.md covers all OS ✅
- [ ] requirements.txt has all packages ✅
- [ ] Code has type hints and docstrings ✅
- [ ] Demo video recorded and saved ✅

---

## 🎓 Evaluation Score Checklist

Rate yourself 1-10 on each:

- **Intent Detection Accuracy** - ___ / 10
- **RAG System Quality** - ___ / 10
- **State Management** - ___ / 10
- **Tool Safety** - ___ / 10
- **Code Quality** - ___ / 10
- **Documentation** - ___ / 10
- **Deployability** - ___ / 10
- **Overall** - ___ / 10

---

## 🚀 Next Steps

1. **Get Started:** `python chat.py`
2. **Understand:** Read `README.md`
3. **Test:** `pytest tests/ -v`
4. **Demo:** Run scenarios and record video
5. **Submit:** GitHub repo + demo video

---

## 📞 Support Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangChain Docs:** https://python.langchain.com/
- **Anthropic Docs:** https://docs.anthropic.com/
- **This Project:** See README.md

---

**Quick Start:** `python setup_verify.py` then `python chat.py` 🚀
