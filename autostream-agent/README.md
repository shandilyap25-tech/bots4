# AutoStream Social-to-Lead Agentic Workflow

A production-ready conversational AI agent that converts social media conversations into qualified business leads for AutoStream, a fictional SaaS platform providing automated video editing tools.

## 🎯 Project Overview

This project demonstrates a real-world GenAI agent built with **LangGraph** that:
- ✅ Understands user intent with multi-class classification (greeting, inquiry, high-intent lead)
- ✅ Retrieves accurate product information using RAG (Retrieval Augmented Generation)
- ✅ Identifies high-intent users ready to sign up
- ✅ Collects lead information through natural conversation
- ✅ Triggers backend lead capture with a mock API
- ✅ Maintains conversation state across 5+ turns

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [WhatsApp Integration](#whatsapp-integration)
- [Conversation Flow](#conversation-flow)
- [Evaluation Criteria](#evaluation-criteria)

## ✨ Features

### 1. **Multi-Class Intent Detection**
The agent classifies user intent into three categories:
- **Casual Greeting**: Simple greetings and pleasantries
- **Product Inquiry**: Questions about features, pricing, or capabilities
- **High-Intent Lead**: User expressing readiness to purchase or sign up

### 2. **RAG-Powered Knowledge Retrieval**
The agent accesses a local knowledge base containing:
- **Pricing Plans**: Basic ($29/mo, 720p) and Pro ($79/mo, 4K)
- **Features**: Video editing capabilities, AI-powered features
- **Company Policies**: Refund policy, support availability
- **Use Cases**: Optimized recommendations for different creator types

### 3. **Stateful Lead Capture**
Intelligent collection of lead information:
- Requests name, email, and content platform
- Validates information before API call
- Prevents premature tool execution
- Maintains user context across conversation turns

### 4. **State Management**
LangGraph-based state machine tracks:
- Conversation history and turn count
- User intent classification
- Partially collected lead information
- Lead capture status

## 🏗️ Architecture

### Why LangGraph?

**LangGraph** was chosen over simple chains for several critical reasons:

1. **Stateful Agentic Workflows**: LangGraph's node-based architecture allows precise control over the conversation flow. Unlike simple chains (LLMChain), we can:
   - Maintain rich state across turns (not just message history)
   - Implement conditional routing based on intent
   - Execute tools only when specific conditions are met

2. **Fine-Grained Intent-Driven Routing**: 
   - After each LLM response, we conditionally route to different nodes
   - Lead qualification only happens for high-intent users
   - Information collection continues until all fields are complete

3. **Reliability & Determinism**: 
   - Explicit state transitions prevent accidental tool execution
   - Clear separation between intent detection, RAG retrieval, and response generation
   - Easier to debug and test individual nodes

### State Management Flow

```
Input Processing
    ↓
Intent Detection
    ↓
RAG Retrieval
    ↓
Generate Response
    ├─→ Is High-Intent? 
    │   ├─→ Lead Qualification
    │   │   ├─→ Missing Info?
    │   │   │   └─→ Collect Lead Info
    │   │   │       ├─→ All Info Complete?
    │   │   │       │   └─→ Capture Lead (API Call)
    │   │   │       └─→ End Turn
    │   │   └─→ End Turn (info complete)
    │   └─→ End Turn
    └─→ Continue Conversation
```

### Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Framework** | LangGraph | Agentic control flow, explicit state management |
| **LLM** | Claude 3 Haiku (configurable) | Fast, cost-effective, good reasoning |
| **State Management** | TypedDict (Python) | Type-safe, serializable state |
| **Knowledge Base** | JSON + RAG | Simple, keyword-based retrieval |
| **Tool Execution** | Mock API | Demonstrates production-ready patterns |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- API key for an LLM provider (Claude, OpenAI, or Google Gemini)

### Installation

1. **Clone/Setup the project**
```bash
cd autostream-agent
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up your API key**
```bash
# For Claude (recommended for this project)
export ANTHROPIC_API_KEY="your-api-key-here"

# Or for OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Or for Google Gemini
export GOOGLE_API_KEY="your-api-key-here"
```

5. **Run the agent**
```bash
python agent.py
```

### Quick Test

```python
from agent import AutoStreamAgent

# Initialize agent
agent = AutoStreamAgent()

# Single turn conversation
response = agent.chat("Tell me about your pricing")
print(response)

# Multi-turn conversation
messages = [
    "Hi, I'm interested in video editing",
    "What's your Pro plan pricing?",
    "I want to try it for my YouTube channel",
    "My name is John Smith and email is john@example.com"
]
conversation = agent.multi_turn_chat(messages)
```

## 📁 Project Structure

```
autostream-agent/
├── agent.py                          # Main agent implementation
├── knowledge_base/
│   └── autostream_kb.json           # Product knowledge base
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── tests/
│   ├── test_intent_detection.py     # Intent detection tests
│   ├── test_rag_retrieval.py        # RAG pipeline tests
│   └── test_lead_capture.py         # Lead capture tests
└── demo/
    └── example_conversations.py      # Example usage patterns
```

## 💬 Usage

### Basic Conversation

```python
from agent import AutoStreamAgent

agent = AutoStreamAgent()

# Start a conversation
response = agent.chat("Tell me about AutoStream")
print(response)

# Agent will respond with relevant information from the knowledge base
```

### Multi-Turn Conversation with State Preservation

```python
conversation_history = agent.multi_turn_chat([
    "Hi there!",
    "Can you tell me about your video editing tools?",
    "What's the difference between your plans?",
    "I'm interested in the Pro plan for my YouTube channel"
])

# Returns structured conversation with intent detection at each turn
for exchange in conversation_history:
    print(f"{exchange['role']}: {exchange['content']}")
```

### Accessing Agent State

```python
# The agent maintains state across turns
state = {
    "messages": [...],           # Full conversation history
    "intent": "product_inquiry",  # Current detected intent
    "user_name": "John",         # Collected lead info
    "user_email": "john@email.com",
    "user_platform": "YouTube",
    "lead_captured": False,       # Lead status
    "conversation_turn": 3        # Turn count
}
```

## 🔌 WhatsApp Integration

### Integration Architecture

Integrating this agent with WhatsApp requires a webhook-based approach:

```
WhatsApp Business API
    ↓
[Your Backend Server]
    ├─→ Webhook: /webhook/messages (POST)
    └─→ Uses: agent.chat() & agent.multi_turn_chat()
    ↓
[AutoStream Agent]
    ├─→ Intent Detection
    ├─→ RAG Retrieval
    ├─→ Lead Capture
    ↓
[CRM/Database]
    └─→ Store leads
```

### Implementation Steps

1. **Set up WhatsApp Business Account**
   - Register for WhatsApp Business Platform
   - Create a Business App
   - Generate phone number and webhook token

2. **Implement Webhook Handler**
```python
from flask import Flask, request
from agent import AutoStreamAgent

app = Flask(__name__)
agent = AutoStreamAgent()

# Store conversation sessions
user_sessions = {}

@app.route('/webhook/messages', methods=['POST'])
def handle_message():
    data = request.json
    user_id = data['messages'][0]['from']
    message_text = data['messages'][0]['text']['body']
    
    # Maintain multi-turn conversation
    if user_id not in user_sessions:
        user_sessions[user_id] = {"history": []}
    
    # Get agent response
    response = agent.chat(message_text)
    user_sessions[user_id]["history"].append({
        "role": "user",
        "content": message_text
    })
    user_sessions[user_id]["history"].append({
        "role": "assistant", 
        "content": response
    })
    
    # Send response back to WhatsApp
    send_whatsapp_message(user_id, response)
    
    return {"status": "ok"}

def send_whatsapp_message(phone_id, message):
    # Call WhatsApp Business API
    # POST to: https://graph.instagram.com/v18.0/{PHONE_NUMBER_ID}/messages
    pass

if __name__ == '__main__':
    app.run(port=5000)
```

3. **Configure Webhook in WhatsApp Dashboard**
   - Set Webhook URL: `https://your-domain.com/webhook/messages`
   - Set Verify Token: Your custom token
   - Subscribe to: `messages`, `messaging_postbacks`

4. **Deploy to Production**
   - Use ngrok for testing: `ngrok http 5000`
   - Deploy backend to cloud (AWS Lambda, Heroku, etc.)
   - Ensure HTTPS is enabled
   - Store sessions in Redis or database instead of memory

5. **Handle Lead Capture**
   - When lead is captured, integrate with CRM via API
   - Send confirmation message to WhatsApp user
   - Sync data to your sales system

### Key Considerations for WhatsApp

- **Message Rate Limiting**: WhatsApp has throughput limits
- **Session Management**: Store conversations in database (Redis, DynamoDB)
- **Rich Messages**: Can send images, buttons, templates
- **Lead Verification**: Validate email via OTP confirmation
- **Privacy**: Comply with GDPR and WhatsApp policies
- **Multi-Language**: Add language detection for global deployment

## 📝 Conversation Flow Example

```
Turn 1:
  User: "Hi, tell me about your pricing"
  Intent: product_inquiry
  Agent: [Retrieves pricing from KB and responds]

Turn 2:
  User: "That sounds good, I want to try the Pro plan"
  Intent: high_intent_lead
  Agent: "Great! To get you started with the Pro plan, 
          could I get your name?"
  Action: Lead Qualification begins

Turn 3:
  User: "My name is Sarah Johnson"
  Intent: high_intent_lead
  Agent: "Thanks Sarah! What's the best email to reach you?"
  Action: Collecting information

Turn 4:
  User: "sarah.johnson@email.com"
  Intent: high_intent_lead
  Agent: "Perfect! Which platform will you be using - 
          YouTube, Instagram, TikTok, etc.?"
  Action: Still collecting information

Turn 5:
  User: "YouTube"
  Intent: high_intent_lead
  Agent: "Excellent! I've registered you for AutoStream Pro. 
         Your lead has been captured. Confirmation sent to 
         your email!"
  Action: mock_lead_capture() called → Lead captured ✅
```

## ✅ Evaluation Criteria

This project is evaluated on:

| Criterion | Implementation |
|-----------|-----------------|
| **Intent Detection** | Multi-class classification with conversation context awareness |
| **RAG Usage** | Keyword-based retrieval from JSON KB; extensible to vector embeddings |
| **State Management** | Full TypedDict state preserving context across 5+ turns |
| **Tool Execution** | Conditional execution only when all lead info is collected |
| **Code Quality** | Type hints, docstrings, clear separation of concerns |
| **Deployability** | Mock API demonstrates production patterns; ready for real CRM integration |

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Individual test files:
```bash
pytest tests/test_intent_detection.py
pytest tests/test_rag_retrieval.py
pytest tests/test_lead_capture.py
```

## 🎬 Creating a Demo Video

To record a 2-3 minute demo:

1. **Run the agent** with sample conversation
2. **Screen record** with audio narration
3. **Show**:
   - Agent answering pricing question
   - Intent detection triggering lead flow
   - Collection of user information
   - Successful lead capture with mock API output

Example script:
```bash
python agent.py  # Runs demo conversation automatically
# Record the output
```

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Claude API Docs](https://docs.anthropic.com/)
- [WhatsApp Business API](https://developers.facebook.com/docs/whatsapp/business-platform)

## 🔮 Future Enhancements

- [ ] Semantic similarity search using embeddings (vector DB)
- [ ] Multi-language support with translation
- [ ] Advanced lead scoring with conversation analytics
- [ ] Integration with real CRM systems (Salesforce, HubSpot)
- [ ] A/B testing different conversation flows
- [ ] Real-time analytics dashboard
- [ ] Voice interface support
- [ ] Mobile app deployment

## 📄 License

This project is part of a ServiceHive Machine Learning Intern assignment.

---

**Built with ❤️ using LangGraph and Claude AI**
