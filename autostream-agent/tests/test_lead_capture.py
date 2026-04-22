"""
Unit tests for Lead Capture System
"""

import pytest
import re
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, '/Users/acer/Downloads/bot/autostream-agent')

from agent import mock_lead_capture, LeadInfo


class TestLeadCapture:
    """Test suite for lead capture system"""
    
    @pytest.fixture
    def lead_info(self):
        """Create a LeadInfo instance"""
        return LeadInfo()
    
    # Test LeadInfo Data Class
    def test_lead_info_initialization(self, lead_info):
        """Test LeadInfo initialization"""
        assert lead_info.name is None
        assert lead_info.email is None
        assert lead_info.platform is None
    
    def test_lead_info_with_values(self):
        """Test LeadInfo with values"""
        lead = LeadInfo(
            name="John Doe",
            email="john@example.com",
            platform="YouTube"
        )
        assert lead.name == "John Doe"
        assert lead.email == "john@example.com"
        assert lead.platform == "YouTube"
    
    def test_lead_info_is_complete_false(self, lead_info):
        """Test is_complete returns False with missing fields"""
        lead_info.name = "John"
        assert not lead_info.is_complete()
    
    def test_lead_info_is_complete_true(self, lead_info):
        """Test is_complete returns True with all fields"""
        lead_info.name = "John Doe"
        lead_info.email = "john@example.com"
        lead_info.platform = "YouTube"
        assert lead_info.is_complete()
    
    def test_lead_info_partial_completion(self, lead_info):
        """Test is_complete with partial data"""
        lead_info.name = "John"
        lead_info.email = "john@example.com"
        assert not lead_info.is_complete()
    
    # Test Mock Lead Capture API
    def test_mock_lead_capture_basic(self):
        """Test basic lead capture"""
        result = mock_lead_capture("John Doe", "john@example.com", "YouTube")
        
        assert result["success"] is True
        assert result["name"] == "John Doe"
        assert result["email"] == "john@example.com"
        assert result["platform"] == "YouTube"
    
    def test_mock_lead_capture_returns_lead_id(self):
        """Test that lead capture returns a lead ID"""
        result = mock_lead_capture("Jane Smith", "jane@example.com", "Instagram")
        
        assert "lead_id" in result
        assert result["lead_id"].startswith("LEAD_")
    
    def test_mock_lead_capture_returns_timestamp(self):
        """Test that lead capture returns a timestamp"""
        result = mock_lead_capture("Bob Wilson", "bob@example.com", "TikTok")
        
        assert "captured_at" in result
        assert "T" in result["captured_at"]  # ISO format check
    
    def test_mock_lead_capture_multiple_calls_unique_ids(self):
        """Test that multiple calls generate unique lead IDs"""
        result1 = mock_lead_capture("User1", "user1@example.com", "YouTube")
        result2 = mock_lead_capture("User2", "user2@example.com", "Instagram")
        
        assert result1["lead_id"] != result2["lead_id"]
    
    def test_mock_lead_capture_different_platforms(self):
        """Test lead capture with different platforms"""
        platforms = ["YouTube", "Instagram", "TikTok", "Twitter", "LinkedIn"]
        
        for platform in platforms:
            result = mock_lead_capture("Test User", "test@example.com", platform)
            assert result["platform"] == platform
    
    def test_mock_lead_capture_email_validation(self):
        """Test that different email formats are captured"""
        emails = [
            "simple@example.com",
            "user.name@example.co.uk",
            "user+tag@example.com",
            "user_name@example-domain.com"
        ]
        
        for email in emails:
            result = mock_lead_capture("Test User", email, "YouTube")
            assert result["email"] == email
    
    def test_mock_lead_capture_name_variations(self):
        """Test lead capture with name variations"""
        names = [
            "John",
            "John Doe",
            "John Michael Doe",
            "José García",
            "李明"
        ]
        
        for name in names:
            result = mock_lead_capture(name, "test@example.com", "YouTube")
            assert result["name"] == name
    
    @patch('builtins.print')
    def test_mock_lead_capture_prints_confirmation(self, mock_print):
        """Test that lead capture prints confirmation"""
        mock_lead_capture("John", "john@example.com", "YouTube")
        
        # Verify print was called
        assert mock_print.called
        print_output = str(mock_print.call_args_list)
        assert "LEAD CAPTURE SUCCESSFUL" in print_output
    
    # Test Lead Info Extraction
    class TestLeadInfoExtraction:
        """Test extraction of lead info from messages"""
        
        def test_extract_name_my_name_is(self):
            """Test extracting name from 'my name is' pattern"""
            message = "My name is Sarah Johnson"
            match = re.search(r"my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", message)
            assert match is not None
            assert match.group(1) == "Sarah Johnson"
        
        def test_extract_email_pattern(self):
            """Test extracting email"""
            message = "You can reach me at sarah@example.com"
            match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", message)
            assert match is not None
            assert match.group(1) == "sarah@example.com"
        
        def test_extract_platform_youtube(self):
            """Test extracting platform - YouTube"""
            message = "I use YouTube for my content"
            platforms = ["youtube", "instagram", "tiktok", "twitter", "linkedin", "facebook"]
            for platform in platforms:
                if platform in message.lower():
                    assert platform == "youtube"
        
        def test_extract_platform_instagram(self):
            """Test extracting platform - Instagram"""
            message = "I'm an Instagram influencer"
            platforms = ["youtube", "instagram", "tiktok", "twitter", "linkedin", "facebook"]
            for platform in platforms:
                if platform in message.lower():
                    assert platform == "instagram"
        
        def test_extract_multiple_platforms_first_match(self):
            """Test that first platform mentioned is extracted"""
            message = "I post on YouTube and Instagram"
            platforms = ["youtube", "instagram", "tiktok", "twitter", "linkedin", "facebook"]
            found_platform = None
            for platform in platforms:
                if platform in message.lower():
                    found_platform = platform
                    break
            assert found_platform == "youtube"


class TestLeadCapturePrevention:
    """Test that premature lead capture is prevented"""
    
    def test_lead_not_captured_without_name(self):
        """Test that lead is not captured without name"""
        lead = LeadInfo(email="test@example.com", platform="YouTube")
        assert not lead.is_complete()
    
    def test_lead_not_captured_without_email(self):
        """Test that lead is not captured without email"""
        lead = LeadInfo(name="John", platform="YouTube")
        assert not lead.is_complete()
    
    def test_lead_not_captured_without_platform(self):
        """Test that lead is not captured without platform"""
        lead = LeadInfo(name="John", email="john@example.com")
        assert not lead.is_complete()
    
    def test_lead_validation_all_fields_required(self):
        """Test that all fields are required"""
        # Missing one field
        incomplete_leads = [
            LeadInfo(name="John", email="john@example.com"),  # missing platform
            LeadInfo(name="John", platform="YouTube"),  # missing email
            LeadInfo(email="john@example.com", platform="YouTube"),  # missing name
        ]
        
        for lead in incomplete_leads:
            assert not lead.is_complete()
    
    def test_lead_captured_only_when_complete(self):
        """Test that lead is only marked complete when all fields present"""
        lead = LeadInfo()
        assert not lead.is_complete()
        
        lead.name = "John"
        assert not lead.is_complete()
        
        lead.email = "john@example.com"
        assert not lead.is_complete()
        
        lead.platform = "YouTube"
        assert lead.is_complete()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
