"""
Unit tests for Intent Detection
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

import sys
sys.path.insert(0, '/Users/acer/Downloads/bot/autostream-agent')

from agent import IntentDetector


class TestIntentDetection:
    """Test suite for intent detection"""
    
    @pytest.fixture
    def detector(self):
        return IntentDetector()
    
    # Test Casual Greeting Detection
    def test_casual_greeting_simple(self, detector):
        """Test detection of simple greetings"""
        messages = [HumanMessage(content="Hi there")]
        intent = detector.detect_intent("Hi", messages)
        assert intent == "casual_greeting"
    
    def test_casual_greeting_hello(self, detector):
        """Test 'hello' greeting"""
        messages = []
        intent = detector.detect_intent("hello", messages)
        assert intent == "casual_greeting"
    
    def test_casual_greeting_thank_you(self, detector):
        """Test 'thank you' greeting"""
        messages = []
        intent = detector.detect_intent("Thanks for the info", messages)
        assert intent == "casual_greeting"
    
    # Test Product Inquiry Detection
    def test_product_inquiry_pricing(self, detector):
        """Test detection of pricing inquiry"""
        messages = []
        intent = detector.detect_intent("What is your pricing?", messages)
        assert intent == "product_inquiry"
    
    def test_product_inquiry_features(self, detector):
        """Test detection of feature inquiry"""
        messages = []
        intent = detector.detect_intent("Can you tell me about your features?", messages)
        assert intent == "product_inquiry"
    
    def test_product_inquiry_how_does(self, detector):
        """Test detection of 'how does' inquiry"""
        messages = []
        intent = detector.detect_intent("How does your video editor work?", messages)
        assert intent == "product_inquiry"
    
    def test_product_inquiry_plans(self, detector):
        """Test detection of plan inquiry"""
        messages = []
        intent = detector.detect_intent("What plans do you offer?", messages)
        assert intent == "product_inquiry"
    
    # Test High-Intent Lead Detection
    def test_high_intent_ready_to_try(self, detector):
        """Test detection of 'ready to try' high-intent"""
        messages = []
        intent = detector.detect_intent("I'm ready to try your service", messages)
        assert intent == "high_intent_lead"
    
    def test_high_intent_want_to_sign_up(self, detector):
        """Test detection of sign-up intent"""
        messages = []
        intent = detector.detect_intent("I want to sign up for the Pro plan", messages)
        assert intent == "high_intent_lead"
    
    def test_high_intent_get_started(self, detector):
        """Test detection of 'get started' intent"""
        messages = []
        intent = detector.detect_intent("Let's get started!", messages)
        assert intent == "high_intent_lead"
    
    def test_high_intent_where_to_subscribe(self, detector):
        """Test detection of subscription ready intent"""
        messages = []
        intent = detector.detect_intent("Where can I subscribe?", messages)
        assert intent == "high_intent_lead"
    
    def test_high_intent_excited_to_try(self, detector):
        """Test detection of excited/positive signals"""
        messages = []
        intent = detector.detect_intent("I'm excited to try your Pro plan", messages)
        assert intent == "high_intent_lead"
    
    # Test Context-Aware Detection
    def test_context_aware_detection(self, detector):
        """Test that context from conversation history affects intent"""
        # Simulate previous product discussion
        messages = [
            AIMessage(content="Our Pro plan costs $79/month with 4K resolution"),
            HumanMessage(content="That sounds great")
        ]
        intent = detector.detect_intent("That sounds great", messages)
        # Should recognize product context
        assert intent in ["product_inquiry", "casual_greeting"]
    
    # Test Edge Cases
    def test_empty_message(self, detector):
        """Test handling of empty messages"""
        messages = []
        intent = detector.detect_intent("", messages)
        assert intent in ["casual_greeting", "product_inquiry"]
    
    def test_special_characters(self, detector):
        """Test handling of special characters"""
        messages = []
        intent = detector.detect_intent("Hi!!!!! 😀", messages)
        assert intent == "casual_greeting"
    
    def test_lowercase_conversion(self, detector):
        """Test that detection works with uppercase"""
        messages = []
        intent = detector.detect_intent("TELL ME ABOUT YOUR PRICING", messages)
        assert intent == "product_inquiry"
    
    def test_mixed_intent_signals(self, detector):
        """Test message with multiple intent signals (high-intent takes priority)"""
        messages = []
        intent = detector.detect_intent("Hi, I want to sign up for your service", messages)
        # High-intent should take priority
        assert intent == "high_intent_lead"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
