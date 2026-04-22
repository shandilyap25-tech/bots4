"""
AutoStream Agent - Social-to-Lead Agentic Workflow
Built with LangGraph for state management and agentic control flow
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_model import BaseLangModel  # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.types import StreamWriter
from typing_extensions import TypedDict
import re


# ============================================================================
# Data Models
# ============================================================================

class AgentState(TypedDict):
    """State management for multi-turn conversation"""
    messages: List[BaseMessage]
    intent: Optional[str]
    user_name: Optional[str]
    user_email: Optional[str]
    user_platform: Optional[str]
    lead_captured: bool
    conversation_turn: int
    knowledge_context: Optional[str]


@dataclass
class LeadInfo:
    """Data class for lead information"""
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None
    
    def is_complete(self) -> bool:
        return all([self.name, self.email, self.platform])


class IntentClassification(BaseModel):
    intent: Literal["casual_greeting", "product_inquiry", "high_intent_lead"] = Field(
        description="The detected intent of the user. 'high_intent_lead' implies they are ready to purchase, try the product, or sign up."
    )


class LeadExtraction(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the user, if provided.")
    email: Optional[str] = Field(default=None, description="The email address of the user, if provided.")
    platform: Optional[str] = Field(default=None, description="The content creator platform the user uses (e.g., YouTube, Instagram), if provided.")


# ============================================================================
# Knowledge Base Loader
# ============================================================================

class KnowledgeBaseManager:
    """Manages loading and querying the AutoStream knowledge base"""
    
    def __init__(self, kb_path: str = "knowledge_base/autostream_kb.json"):
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()
    
    def _load_kb(self) -> Dict[str, Any]:
        """Load knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")
    
    def get_pricing_info(self) -> str:
        """Retrieve pricing information"""
        pricing = self.knowledge_base.get("pricing", {})
        return json.dumps(pricing, indent=2)
    
    def get_features(self) -> str:
        """Retrieve features information"""
        features = self.knowledge_base.get("features", {})
        return json.dumps(features, indent=2)
    
    def get_policies(self) -> str:
        """Retrieve company policies"""
        policies = self.knowledge_base.get("policies", {})
        return json.dumps(policies, indent=2)
    
    def get_use_cases(self) -> str:
        """Retrieve use case information"""
        use_cases = self.knowledge_base.get("use_cases", {})
        return json.dumps(use_cases, indent=2)
    
    def query_rag(self, query: str) -> str:
        """
        Simple RAG - retrieve relevant knowledge based on keywords
        In production, this would use vector embeddings for semantic search
        """
        query_lower = query.lower()
        context_parts = []
        
        # Check for pricing queries
        if any(word in query_lower for word in ["price", "cost", "plan", "subscription", "payment"]):
            context_parts.append(f"Pricing Information:\n{self.get_pricing_info()}")
        
        # Check for feature queries
        if any(word in query_lower for word in ["feature", "capability", "can", "support", "does"]):
            context_parts.append(f"Features:\n{self.get_features()}")
        
        # Check for policy queries
        if any(word in query_lower for word in ["refund", "cancel", "support", "policy", "guarantee"]):
            context_parts.append(f"Policies:\n{self.get_policies()}")
        
        # Check for use case queries
        if any(word in query_lower for word in ["youtube", "instagram", "tiktok", "use", "best for", "creator"]):
            context_parts.append(f"Use Cases:\n{self.get_use_cases()}")
        
        # If no specific context found, return general info
        if not context_parts:
            context_parts.append(f"General Information:\n{json.dumps(self.knowledge_base.get('company', {}), indent=2)}")
        
        return "\n\n".join(context_parts)


# ============================================================================
# Tool Execution
# ============================================================================

def mock_lead_capture(name: str, email: str, platform: str) -> Dict[str, Any]:
    """
    Mock API function to capture leads
    In production, this would send data to CRM/database
    """
    timestamp = datetime.now().isoformat()
    lead_id = f"LEAD_{timestamp.replace('-', '').replace(':', '').replace('.', '')}"
    
    result = {
        "success": True,
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "message": f"✅ Lead captured successfully! Lead ID: {lead_id}"
    }
    
    print(f"\n{'='*60}")
    print("🎉 LEAD CAPTURE SUCCESSFUL")
    print(f"{'='*60}")
    print(f"Lead ID: {lead_id}")
    print(f"Name: {name}")
    print(f"Email: {email}")
    print(f"Platform: {platform}")
    print(f"Captured at: {timestamp}")
    print(f"{'='*60}\n")
    
    return result


# ============================================================================
# LLM Setup
# ============================================================================

def initialize_llm() -> BaseLangModel:
    """Initialize the LLM - uses Gemini 1.5 Flash"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Please set your API key.")
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )


# ============================================================================
# Agent Nodes
# ============================================================================

class AutoStreamAgent:
    """Main agent class with LangGraph state management"""
    
    def __init__(self, kb_path: str = "knowledge_base/autostream_kb.json"):
        self.kb_manager = KnowledgeBaseManager(kb_path)
        self.llm = initialize_llm()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", self.process_input_node)
        workflow.add_node("intent_detection", self.intent_detection_node)
        workflow.add_node("rag_retrieval", self.rag_retrieval_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("lead_qualification", self.lead_qualification_node)
        workflow.add_node("collect_lead_info", self.collect_lead_info_node)
        workflow.add_node("capture_lead", self.capture_lead_node)
        
        # Set entry point
        workflow.set_entry_point("process_input")
        
        # Add edges
        workflow.add_edge("process_input", "intent_detection")
        workflow.add_edge("intent_detection", "rag_retrieval")
        workflow.add_edge("rag_retrieval", "generate_response")
        workflow.add_conditional_edges(
            "generate_response",
            self.should_qualify_lead,
            {
                "lead_qualification": "lead_qualification",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "lead_qualification",
            self.should_collect_info,
            {
                "collect_lead_info": "collect_lead_info",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "collect_lead_info",
            self.is_lead_complete,
            {
                "capture_lead": "capture_lead",
                "end": END
            }
        )
        workflow.add_edge("capture_lead", END)
        
        return workflow.compile()
    
    def process_input_node(self, state: AgentState) -> AgentState:
        """Process and validate user input"""
        state["conversation_turn"] += 1
        return state
    
    def intent_detection_node(self, state: AgentState) -> AgentState:
        """Detect user intent"""
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                classifier = self.llm.with_structured_output(IntentClassification)
                
                # Provide a bit of context for better classification
                recent_history = [str(msg.content) for msg in state["messages"][-4:-1] if isinstance(msg, HumanMessage)]
                context = "\n".join(recent_history)
                prompt = f"Previous messages:\n{context}\n\nCurrent message: {last_message.content}\n\nClassify the intent of the current message."
                
                result = classifier.invoke([SystemMessage(content="You classify user intents."), HumanMessage(content=prompt)])
                state["intent"] = result.intent
        return state
    
    def rag_retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge using RAG"""
        if state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                context = self.kb_manager.query_rag(last_message.content)
                state["knowledge_context"] = context
        return state
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate agent response using LLM"""
        # Build system prompt
        system_prompt = self._build_system_prompt(state)
        
        # Build context for LLM
        messages_for_llm = [SystemMessage(content=system_prompt)]
        messages_for_llm.extend(state["messages"])
        
        # Generate response
        response = self.llm.invoke(messages_for_llm)
        
        # Add response to messages
        state["messages"].append(response)
        
        return state
    
    def lead_qualification_node(self, state: AgentState) -> AgentState:
        """Qualify if user is a high-intent lead"""
        if state["intent"] == "high_intent_lead":
            # Extract lead info if present in message
            last_message = state["messages"][-2]  # Get user message before agent response
            if isinstance(last_message, HumanMessage):
                    state = self._extract_lead_info(state, str(last_message.content))
        return state
    
    def collect_lead_info_node(self, state: AgentState) -> AgentState:
        """Collect missing lead information"""
        # The LLM already generated a response asking for info
        # This node just marks that we're collecting info
        return state
    
    def capture_lead_node(self, state: AgentState) -> AgentState:
        """Capture the lead using mock API"""
        if state["user_name"] and state["user_email"] and state["user_platform"]:
            result = mock_lead_capture(
                state["user_name"],
                state["user_email"],
                state["user_platform"]
            )
            state["lead_captured"] = True
            
            # Add success message to conversation
            success_msg = AIMessage(
                content=f"🎉 Perfect! I've registered you for AutoStream Pro. "
                f"Your lead has been captured (ID: {result['lead_id']}). "
                f"You should receive a confirmation email at {state['user_email']} shortly. "
                f"Welcome to AutoStream, {state['user_name']}!"
            )
            state["messages"].append(success_msg)
        
        return state
    
    def should_qualify_lead(self, state: AgentState) -> Literal["lead_qualification", "end"]:
        """Determine if we should qualify the user as a lead"""
        if state["intent"] == "high_intent_lead" and not state["lead_captured"]:
            return "lead_qualification"
        return "end"
    
    def should_collect_info(self, state: AgentState) -> Literal["collect_lead_info", "end"]:
        """Determine if we need to collect more info"""
        lead = LeadInfo(
            name=state["user_name"],
            email=state["user_email"],
            platform=state["user_platform"]
        )
        if not lead.is_complete() and not state["lead_captured"]:
            return "collect_lead_info"
        return "end"
    
    def is_lead_complete(self, state: AgentState) -> Literal["capture_lead", "end"]:
        """Check if all lead info has been collected"""
        last_message = state["messages"][-1]  # Last user message
        if isinstance(last_message, HumanMessage):
            state = self._extract_lead_info(state, str(last_message.content))
        
        lead = LeadInfo(
            name=state["user_name"],
            email=state["user_email"],
            platform=state["user_platform"]
        )
        
        if lead.is_complete():
            return "capture_lead"
        return "end"
    
    def _extract_lead_info(self, state: AgentState, message: str) -> AgentState:
        """Extract lead information using LLM structured output"""
        extractor = self.llm.with_structured_output(LeadExtraction)
        
        result = extractor.invoke([
            SystemMessage(content="Extract the user's name, email, and content creator platform from the message. If any field is missing, leave it as null."),
            HumanMessage(content=message)
        ])
        
        if result.name:
            state["user_name"] = result.name
        if result.email:
            state["user_email"] = result.email
        if result.platform:
            state["user_platform"] = result.platform
        
        return state
    
    def _build_system_prompt(self, state: AgentState) -> str:
        """Build dynamic system prompt based on state"""
        base_prompt = """You are a friendly and knowledgeable AutoStream sales assistant. 
AutoStream is an AI-powered video editing platform for content creators.

Your responsibilities:
1. Answer questions about AutoStream pricing, features, and policies accurately
2. Help users understand which plan (Basic or Pro) fits their needs
3. Identify when a user is showing high intent to purchase
4. When detecting high-intent, politely collect their name, email, and content platform

Current Context:
"""
        if state["knowledge_context"]:
            base_prompt += f"\nRelevant Knowledge:\n{state['knowledge_context']}\n"
        
        base_prompt += f"\nUser Intent: {state['intent']}\n"
        
        if state["user_name"]:
            base_prompt += f"User Name: {state['user_name']}\n"
        if state["user_email"]:
            base_prompt += f"User Email: {state['user_email']}\n"
        if state["user_platform"]:
            base_prompt += f"User Platform: {state['user_platform']}\n"
        
        base_prompt += """
Guidelines:
- Be conversational and enthusiastic
- Reference specific pricing and features from the knowledge base
- For high-intent users, gently ask for missing information (name, email, platform)
- Only mention that you're capturing a lead after ALL information is collected
- Be helpful and address concerns
- If unsure about something, ask the user or admit the limitation
"""
        return base_prompt
    
    def chat(self, user_message: str) -> str:
        """Process a user message and return agent response"""
        # Initialize state for first message
        state = {
            "messages": [],
            "intent": None,
            "user_name": None,
            "user_email": None,
            "user_platform": None,
            "lead_captured": False,
            "conversation_turn": 0,
            "knowledge_context": None
        }
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_message))
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Extract final response
        if result["messages"]:
            last_message = result["messages"][-1]
            return last_message.content if isinstance(last_message, AIMessage) else str(last_message)
        
        return "I couldn't generate a response. Please try again."
    
    def multi_turn_chat(self, messages: List[str]) -> List[Dict[str, str]]:
        """
        Run a multi-turn conversation
        Returns list of {role, content} for the entire conversation
        """
        state = {
            "messages": [],
            "intent": None,
            "user_name": None,
            "user_email": None,
            "user_platform": None,
            "lead_captured": False,
            "conversation_turn": 0,
            "knowledge_context": None
        }
        
        conversation_history = []
        
        for user_message in messages:
            # Add user message
            state["messages"].append(HumanMessage(content=user_message))
            conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Run the graph
            result = self.graph.invoke(state)
            
            # Update state for next iteration
            state = result
            
            # Extract and store agent response
            if result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    conversation_history.append({
                        "role": "assistant",
                        "content": last_message.content
                    })
        
        return conversation_history


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Initialize agent
    agent = AutoStreamAgent()
    
    # Example multi-turn conversation
    print("🤖 AutoStream Sales Agent Started\n")
    print("="*60)
    
    sample_conversation = [
        "Hi, I'm interested in video editing software",
        "Can you tell me about your pricing?",
        "That sounds great! I'd like to try the Pro plan for my YouTube channel",
        "My name is Sarah Johnson and my email is sarah.johnson@email.com"
    ]
    
    conversation = agent.multi_turn_chat(sample_conversation)
    
    print("\n📝 Conversation Summary:")
    print("="*60)
    for turn in conversation:
        role = "👤 User" if turn["role"] == "user" else "🤖 Agent"
        print(f"\n{role}:")
        print(f"{turn['content']}")
    
    print("\n" + "="*60)
    print("✅ Demo conversation completed!")
