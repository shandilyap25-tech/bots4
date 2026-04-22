"""
Unit tests for Intent Detection
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import MagicMock
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import AutoStreamAgent, IntentClassification


class TestIntentDetection:
    """Test suite for LLM-based intent detection"""
    
    @pytest.fixture
    def agent_with_mock_llm(self):
        # Provide a dummy key to bypass initialization checks
        os.environ["GOOGLE_API_KEY"] = "mock_key_for_testing"
        agent = AutoStreamAgent()
        
        # Mock the with_structured_output chain
        mock_chain = MagicMock()
        agent.llm = MagicMock()
        agent.llm.with_structured_output.return_value = mock_chain
        return agent, mock_chain
    
    def test_intent_detection_high_intent(self, agent_with_mock_llm):
        """Test detection of high intent via LLM"""
        agent, mock_chain = agent_with_mock_llm
        
        # Setup mock to return a high-intent classification
        mock_chain.invoke.return_value = IntentClassification(intent="high_intent_lead")
        
        state = {"messages": [HumanMessage(content="I want to sign up!")], "intent": None}
        new_state = agent.intent_detection_node(state)
        
        assert new_state["intent"] == "high_intent_lead"
        mock_chain.invoke.assert_called_once()

    def test_intent_detection_casual_greeting(self, agent_with_mock_llm):
        """Test detection of casual greeting via LLM"""
        agent, mock_chain = agent_with_mock_llm
        
        # Setup mock to return a casual_greeting classification
        mock_chain.invoke.return_value = IntentClassification(intent="casual_greeting")
        
        state = {"messages": [HumanMessage(content="Hello there!")], "intent": None}
        new_state = agent.intent_detection_node(state)
        
        assert new_state["intent"] == "casual_greeting"
        mock_chain.invoke.assert_called_once()

    def test_intent_detection_product_inquiry(self, agent_with_mock_llm):
        """Test detection of product inquiry via LLM"""
        agent, mock_chain = agent_with_mock_llm
        
        # Setup mock to return a product_inquiry classification
        mock_chain.invoke.return_value = IntentClassification(intent="product_inquiry")
        
        state = {"messages": [HumanMessage(content="How much does it cost?")], "intent": None}
        new_state = agent.intent_detection_node(state)
        
        assert new_state["intent"] == "product_inquiry"
        mock_chain.invoke.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
