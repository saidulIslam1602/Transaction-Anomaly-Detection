"""
Tests for LLM Service module.
"""

import pytest
from src.services.llm_service import LLMRiskAssessmentService, RiskExplanation


class TestRiskExplanation:
    """Test RiskExplanation dataclass."""
    
    def test_risk_explanation_creation(self):
        """Test creating a risk explanation."""
        explanation = RiskExplanation(
            risk_level="HIGH",
            explanation="Suspicious transaction pattern detected",
            recommendations=["Review transaction", "Contact customer"],
            confidence_score=0.9,
            language="en"
        )
        
        assert explanation.risk_level == "HIGH"
        assert len(explanation.recommendations) == 2
        assert explanation.confidence_score == 0.9
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        explanation = RiskExplanation(
            risk_level="MEDIUM",
            explanation="Some risk indicators",
            recommendations=["Monitor"],
            confidence_score=0.7
        )
        
        result = explanation.to_dict()
        assert isinstance(result, dict)
        assert result['risk_level'] == "MEDIUM"


class TestLLMRiskAssessmentService:
    """Test LLMRiskAssessmentService class."""
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        service = LLMRiskAssessmentService(api_key=None)
        assert service.enabled == False
    
    def test_determine_risk_level(self):
        """Test risk level determination."""
        service = LLMRiskAssessmentService()
        
        assert service._determine_risk_level(9.0) == "CRITICAL"
        assert service._determine_risk_level(6.0) == "HIGH"
        assert service._determine_risk_level(3.0) == "MEDIUM"
        assert service._determine_risk_level(1.0) == "LOW"
    
    def test_fallback_explanation(self):
        """Test fallback explanation generation."""
        service = LLMRiskAssessmentService()
        
        explanation = service._get_fallback_explanation(7.5, "en")
        
        assert isinstance(explanation, RiskExplanation)
        assert explanation.risk_level == "CRITICAL"
        assert len(explanation.recommendations) > 0
        assert explanation.language == "en"
    
    def test_fallback_explanation_norwegian(self):
        """Test fallback explanation in Norwegian."""
        service = LLMRiskAssessmentService()
        
        explanation = service._get_fallback_explanation(5.5, "no")
        
        assert explanation.language == "no"
        assert "transaksjonen" in explanation.explanation.lower()
    
    def test_analyze_transaction_risk_disabled(self, sample_transaction):
        """Test risk analysis when service is disabled."""
        service = LLMRiskAssessmentService(api_key=None)
        
        explanation = service.analyze_transaction_risk(
            sample_transaction,
            risk_score=6.0,
            detection_flags={'rule_based': True}
        )
        
        assert isinstance(explanation, RiskExplanation)
        assert explanation.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_fallback_alert(self, sample_transaction, fraud_transaction):
        """Test fallback alert generation."""
        service = LLMRiskAssessmentService()
        
        explanation = RiskExplanation(
            risk_level="HIGH",
            explanation="Suspicious activity",
            recommendations=["Investigate"],
            confidence_score=0.8
        )
        
        alert = service._get_fallback_alert("M123", fraud_transaction, explanation)
        
        assert isinstance(alert, str)
        assert "HIGH" in alert
        assert len(alert) > 0

