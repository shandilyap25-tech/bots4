"""
Unit tests for RAG Retrieval System
"""

import pytest
import json
import os
import tempfile

import sys
sys.path.insert(0, '/Users/acer/Downloads/bot/autostream-agent')

from agent import KnowledgeBaseManager


class TestRAGRetrieval:
    """Test suite for RAG retrieval system"""
    
    @pytest.fixture
    def kb_manager(self):
        """Create a knowledge base manager instance"""
        return KnowledgeBaseManager("knowledge_base/autostream_kb.json")
    
    # Test Knowledge Base Loading
    def test_kb_loads_successfully(self, kb_manager):
        """Test that knowledge base loads without errors"""
        assert kb_manager.knowledge_base is not None
        assert isinstance(kb_manager.knowledge_base, dict)
    
    def test_kb_contains_required_sections(self, kb_manager):
        """Test that KB contains all required sections"""
        required_sections = ["company", "pricing", "features", "policies", "use_cases"]
        for section in required_sections:
            assert section in kb_manager.knowledge_base
    
    # Test Pricing Retrieval
    def test_pricing_retrieval(self, kb_manager):
        """Test retrieval of pricing information"""
        pricing_info = kb_manager.get_pricing_info()
        assert pricing_info is not None
        assert "basic_plan" in pricing_info or "Basic" in pricing_info
        assert "pro_plan" in pricing_info or "Pro" in pricing_info
        assert "29" in pricing_info  # Basic plan price
        assert "79" in pricing_info  # Pro plan price
    
    def test_pricing_contains_required_fields(self, kb_manager):
        """Test that pricing has all required fields"""
        pricing = json.loads(kb_manager.get_pricing_info())
        basic = pricing.get("basic_plan", {})
        pro = pricing.get("pro_plan", {})
        
        assert basic.get("price_monthly") == 29
        assert pro.get("price_monthly") == 79
    
    # Test Features Retrieval
    def test_features_retrieval(self, kb_manager):
        """Test retrieval of features"""
        features_info = kb_manager.get_features()
        assert features_info is not None
        assert len(features_info) > 0
    
    def test_features_contains_pro_exclusive(self, kb_manager):
        """Test that Pro exclusive features are available"""
        features_info = kb_manager.get_features()
        assert "4K" in features_info or "4k" in features_info.lower()
        assert "24/7" in features_info
    
    # Test Policies Retrieval
    def test_policies_retrieval(self, kb_manager):
        """Test retrieval of policies"""
        policies_info = kb_manager.get_policies()
        assert policies_info is not None
        assert "refund" in policies_info.lower() or "policy" in policies_info.lower()
    
    def test_refund_policy_accuracy(self, kb_manager):
        """Test that refund policy is correct"""
        policies_info = kb_manager.get_policies()
        assert "7" in policies_info  # 7 day refund window
    
    def test_support_policy_accuracy(self, kb_manager):
        """Test that support policy is correct"""
        policies_info = kb_manager.get_policies()
        assert "24/7" in policies_info  # Pro has 24/7 support
    
    # Test Use Cases Retrieval
    def test_use_cases_retrieval(self, kb_manager):
        """Test retrieval of use cases"""
        use_cases_info = kb_manager.get_use_cases()
        assert use_cases_info is not None
        assert len(use_cases_info) > 0
    
    def test_use_cases_contains_platforms(self, kb_manager):
        """Test that use cases cover major platforms"""
        use_cases_info = kb_manager.get_use_cases()
        platforms = ["youtube", "instagram", "tiktok"]
        for platform in platforms:
            assert platform.lower() in use_cases_info.lower()
    
    # Test RAG Query System
    def test_rag_query_pricing(self, kb_manager):
        """Test RAG query for pricing information"""
        result = kb_manager.query_rag("What is your pricing?")
        assert result is not None
        assert "pricing" in result.lower() or "plan" in result.lower()
    
    def test_rag_query_features(self, kb_manager):
        """Test RAG query for features"""
        result = kb_manager.query_rag("What features do you offer?")
        assert result is not None
        assert len(result) > 0
    
    def test_rag_query_youtube(self, kb_manager):
        """Test RAG query for YouTube-specific use case"""
        result = kb_manager.query_rag("Can I use this for YouTube?")
        assert result is not None
        assert "youtube" in result.lower() or "use" in result.lower()
    
    def test_rag_query_refund(self, kb_manager):
        """Test RAG query for refund policy"""
        result = kb_manager.query_rag("What's your refund policy?")
        assert result is not None
        assert "policy" in result.lower() or "refund" in result.lower()
    
    def test_rag_query_support(self, kb_manager):
        """Test RAG query for support information"""
        result = kb_manager.query_rag("Do you offer 24/7 support?")
        assert result is not None
        assert "support" in result.lower()
    
    def test_rag_query_generic_fallback(self, kb_manager):
        """Test that generic queries return company info"""
        result = kb_manager.query_rag("Tell me about AutoStream")
        assert result is not None
        # Should return general company info if no specific match
        assert len(result) > 0
    
    # Test RAG Accuracy
    def test_rag_returns_accurate_prices(self, kb_manager):
        """Test that RAG returns accurate pricing"""
        result = kb_manager.query_rag("pricing")
        assert "29" in result  # Basic plan
        assert "79" in result  # Pro plan
    
    def test_rag_returns_accurate_resolutions(self, kb_manager):
        """Test that RAG returns accurate resolution info"""
        result = kb_manager.query_rag("resolution")
        assert "720p" in result  # Basic
        assert "4K" in result or "4k" in result.lower()  # Pro
    
    def test_rag_keyword_combination(self, kb_manager):
        """Test RAG with multiple keyword queries"""
        result = kb_manager.query_rag("youtube 4K unlimited")
        assert result is not None
        # Should match use case + feature queries
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
