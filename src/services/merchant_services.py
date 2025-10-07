"""
Merchant-focused services for risk intelligence and communication.

This module provides services for merchant risk assessment, alerts,
and personalized fraud prevention insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MerchantRiskProfile:
    """
    Merchant risk profile with transaction patterns and risk history.
    """
    
    def __init__(self, merchant_id: str):
        """
        Initialize merchant risk profile.
        
        Args:
            merchant_id: Unique merchant identifier
        """
        self.merchant_id = merchant_id
        self.transaction_history = []
        self.risk_scores = []
        self.fraud_incidents = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Risk metrics
        self.average_risk_score = 0.0
        self.fraud_rate = 0.0
        self.total_transactions = 0
        self.total_amount = 0.0
        
        # Behavioral patterns
        self.typical_transaction_amount = 0.0
        self.typical_daily_volume = 0
        self.peak_hours = []
        self.common_transaction_types = []
    
    def update_from_transactions(self, transactions: pd.DataFrame) -> None:
        """
        Update profile from transaction data.
        
        Args:
            transactions: DataFrame with merchant transactions
        """
        self.total_transactions = len(transactions)
        self.total_amount = float(transactions['amount'].sum())
        
        # Calculate risk metrics
        if 'final_risk_score' in transactions.columns:
            self.average_risk_score = float(transactions['final_risk_score'].mean())
            self.risk_scores = transactions['final_risk_score'].tolist()
        
        if 'isFraud' in transactions.columns:
            self.fraud_rate = float(transactions['isFraud'].mean())
            self.fraud_incidents = transactions[transactions['isFraud'] == 1].to_dict('records')
        
        # Behavioral patterns
        self.typical_transaction_amount = float(transactions['amount'].median())
        
        if 'type' in transactions.columns:
            type_counts = transactions['type'].value_counts()
            self.common_transaction_types = type_counts.head(3).index.tolist()
        
        if 'step' in transactions.columns:
            # Convert steps to hours (assuming 1 step = 1 hour)
            hourly_volume = transactions.groupby(transactions['step'] % 24).size()
            self.peak_hours = hourly_volume.nlargest(3).index.tolist()
        
        self.updated_at = datetime.now()


class MerchantRiskIntelligenceService:
    """
    Provide risk intelligence and insights for merchants.
    
    This service analyzes merchant transaction patterns and provides
    personalized recommendations for fraud prevention.
    """
    
    def __init__(self):
        """Initialize merchant risk intelligence service."""
        self.merchant_profiles = {}
        logger.info("Merchant risk intelligence service initialized")
    
    def create_merchant_profile(self,
                               merchant_id: str,
                               transactions: pd.DataFrame) -> MerchantRiskProfile:
        """
        Create or update merchant risk profile.
        
        Args:
            merchant_id: Merchant identifier
            transactions: Merchant's transaction data
            
        Returns:
            MerchantRiskProfile object
        """
        if merchant_id in self.merchant_profiles:
            profile = self.merchant_profiles[merchant_id]
        else:
            profile = MerchantRiskProfile(merchant_id)
        
        profile.update_from_transactions(transactions)
        self.merchant_profiles[merchant_id] = profile
        
        logger.info(f"Profile created/updated for merchant {merchant_id}")
        
        return profile
    
    def generate_risk_report(self, merchant_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive risk report for a merchant.
        
        Args:
            merchant_id: Merchant identifier
            
        Returns:
            Dictionary with risk report details
        """
        if merchant_id not in self.merchant_profiles:
            return {'error': 'Merchant profile not found'}
        
        profile = self.merchant_profiles[merchant_id]
        
        # Calculate risk level
        if profile.average_risk_score >= 7.0:
            risk_level = 'HIGH'
        elif profile.average_risk_score >= 4.0:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(profile)
        
        report = {
            'merchant_id': merchant_id,
            'report_date': datetime.now().isoformat(),
            'risk_level': risk_level,
            'risk_score': profile.average_risk_score,
            'fraud_rate': profile.fraud_rate * 100,
            
            'transaction_summary': {
                'total_transactions': profile.total_transactions,
                'total_amount': profile.total_amount,
                'average_amount': profile.typical_transaction_amount,
                'typical_daily_volume': profile.typical_daily_volume
            },
            
            'behavioral_patterns': {
                'peak_hours': profile.peak_hours,
                'common_types': profile.common_transaction_types
            },
            
            'fraud_incidents': len(profile.fraud_incidents),
            
            'recommendations': recommendations,
            
            'health_score': self._calculate_health_score(profile)
        }
        
        return report
    
    def _generate_recommendations(self, profile: MerchantRiskProfile) -> List[str]:
        """Generate personalized recommendations for merchant."""
        recommendations = []
        
        if profile.fraud_rate > 0.05:
            recommendations.append(
                "High fraud rate detected. Consider implementing additional verification steps."
            )
        
        if profile.average_risk_score > 6.0:
            recommendations.append(
                "Average risk score is elevated. Review high-risk transactions manually."
            )
        
        if profile.total_transactions < 10:
            recommendations.append(
                "Limited transaction history. Monitor closely during initial period."
            )
        
        if not recommendations:
            recommendations.append(
                "Transaction patterns appear normal. Continue monitoring."
            )
        
        return recommendations
    
    def _calculate_health_score(self, profile: MerchantRiskProfile) -> float:
        """Calculate overall health score (0-100)."""
        # Base score
        health_score = 100.0
        
        # Deduct for fraud rate
        health_score -= profile.fraud_rate * 500  # -50 for 10% fraud rate
        
        # Deduct for high risk scores
        if profile.average_risk_score > 5.0:
            health_score -= (profile.average_risk_score - 5.0) * 10
        
        # Ensure score is in valid range
        health_score = max(0.0, min(100.0, health_score))
        
        return health_score
    
    def compare_to_industry_benchmark(self,
                                     merchant_id: str,
                                     industry_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Compare merchant metrics to industry benchmarks.
        
        Args:
            merchant_id: Merchant identifier
            industry_stats: Industry benchmark statistics
            
        Returns:
            Dictionary with comparison results
        """
        if merchant_id not in self.merchant_profiles:
            return {'error': 'Merchant profile not found'}
        
        profile = self.merchant_profiles[merchant_id]
        
        # Default industry benchmarks
        if industry_stats is None:
            industry_stats = {
                'avg_fraud_rate': 0.02,
                'avg_risk_score': 3.5,
                'avg_transaction_amount': 5000.0
            }
        
        comparison = {
            'merchant_id': merchant_id,
            'fraud_rate': {
                'merchant': profile.fraud_rate,
                'industry': industry_stats['avg_fraud_rate'],
                'difference_pct': ((profile.fraud_rate - industry_stats['avg_fraud_rate']) /
                                  (industry_stats['avg_fraud_rate'] + 1e-10) * 100)
            },
            'risk_score': {
                'merchant': profile.average_risk_score,
                'industry': industry_stats['avg_risk_score'],
                'difference_pct': ((profile.average_risk_score - industry_stats['avg_risk_score']) /
                                  (industry_stats['avg_risk_score'] + 1e-10) * 100)
            },
            'transaction_amount': {
                'merchant': profile.typical_transaction_amount,
                'industry': industry_stats['avg_transaction_amount'],
                'difference_pct': ((profile.typical_transaction_amount - industry_stats['avg_transaction_amount']) /
                                  (industry_stats['avg_transaction_amount'] + 1e-10) * 100)
            }
        }
        
        return comparison


class MerchantAlertPrioritization:
    """
    Smart alert prioritization to reduce merchant alert fatigue.
    
    Uses ML-based scoring to rank alerts by importance and actionability.
    """
    
    def __init__(self):
        """Initialize alert prioritization system."""
        self.alert_history = []
        logger.info("Merchant alert prioritization initialized")
    
    def prioritize_alerts(self,
                         alerts: List[Dict],
                         merchant_context: Optional[Dict] = None) -> List[Dict]:
        """
        Prioritize alerts based on multiple factors.
        
        Args:
            alerts: List of alert dictionaries
            merchant_context: Optional merchant context information
            
        Returns:
            Sorted list of alerts with priority scores
        """
        prioritized_alerts = []
        
        for alert in alerts:
            priority_score = self._calculate_priority_score(alert, merchant_context)
            
            alert_with_priority = alert.copy()
            alert_with_priority['priority_score'] = priority_score
            alert_with_priority['priority_level'] = self._get_priority_level(priority_score)
            
            prioritized_alerts.append(alert_with_priority)
        
        # Sort by priority score (descending)
        prioritized_alerts.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized_alerts
    
    def _calculate_priority_score(self,
                                 alert: Dict,
                                 merchant_context: Optional[Dict]) -> float:
        """Calculate priority score for an alert."""
        score = 0.0
        
        # Base score from risk level
        risk_score = alert.get('risk_score', 0)
        score += risk_score * 10
        
        # Amount factor
        amount = alert.get('amount', 0)
        if amount > 10000:
            score += 20
        elif amount > 5000:
            score += 10
        elif amount > 1000:
            score += 5
        
        # Historical fraud involvement
        if merchant_context:
            fraud_rate = merchant_context.get('fraud_rate', 0)
            score += fraud_rate * 50
        
        # Detection method factor
        detection_flags = alert.get('detection_flags', {})
        if detection_flags.get('rule_based'):
            score += 15  # Rule-based has high weight
        if detection_flags.get('ml'):
            score += 10
        if detection_flags.get('network'):
            score += 12
        
        # Urgency factor
        if alert.get('requires_immediate_action'):
            score += 25
        
        return score
    
    def _get_priority_level(self, priority_score: float) -> str:
        """Convert priority score to level."""
        if priority_score >= 80:
            return 'CRITICAL'
        elif priority_score >= 60:
            return 'HIGH'
        elif priority_score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'


class MerchantOnboardingRiskAssessment:
    """
    Risk assessment for merchant onboarding process.
    
    Evaluates new merchants to identify high-risk accounts before approval.
    """
    
    def __init__(self):
        """Initialize onboarding risk assessment."""
        logger.info("Merchant onboarding risk assessment initialized")
    
    def assess_new_merchant(self,
                           merchant_data: Dict,
                           business_info: Dict) -> Dict[str, Any]:
        """
        Assess risk for a new merchant during onboarding.
        
        Args:
            merchant_data: Merchant application data
            business_info: Business information and verification data
            
        Returns:
            Dictionary with risk assessment results
        """
        risk_factors = []
        risk_score = 0
        
        # Check business age
        business_age_years = business_info.get('years_in_operation', 0)
        if business_age_years < 1:
            risk_factors.append("New business (less than 1 year)")
            risk_score += 15
        elif business_age_years < 3:
            risk_factors.append("Young business (less than 3 years)")
            risk_score += 10
        
        # Check business type (based on FinCEN and international AML standards)
        high_risk_industries = [
            'gambling', 'casino', 'betting', 'lottery',
            'cryptocurrency', 'crypto', 'bitcoin', 'blockchain exchange',
            'adult entertainment', 'adult', 'escort',
            'forex', 'foreign exchange', 'money transfer', 'remittance',
            'precious metals', 'jewelry', 'gold dealer',
            'arms', 'weapons', 'ammunition',
            'tobacco', 'vaping', 'e-cigarette',
            'cannabis', 'marijuana', 'cbd',
            'pawn shop', 'pawnbroker',
            'cash intensive', 'atm operation',
            'money service business', 'msb', 'check cashing'
        ]
        # Medium-risk industries requiring enhanced monitoring
        medium_risk_industries = [
            'real estate', 'property development',
            'car dealer', 'auto sales', 'vehicle sales',
            'art dealer', 'antique',
            'nonprofit', 'charity', 'ngo',
            'travel agency', 'tourism',
            'import export', 'trading company',
            'consulting', 'advisory services'
        ]
        
        industry = business_info.get('industry', '').lower()
        if any(risky in industry for risky in high_risk_industries):
            risk_factors.append(f"High-risk industry per AML standards: {industry}")
            risk_score += 30
        elif any(medium in industry for medium in medium_risk_industries):
            risk_factors.append(f"Medium-risk industry requiring enhanced monitoring: {industry}")
            risk_score += 15
        
        # Check ownership structure
        if business_info.get('ownership_verified') == False:
            risk_factors.append("Ownership not verified")
            risk_score += 20
        
        # Check registration information
        if not business_info.get('business_registered'):
            risk_factors.append("Business not properly registered")
            risk_score += 30
        
        # Check requested transaction limits
        requested_limit = merchant_data.get('monthly_transaction_limit', 0)
        if requested_limit > 1000000:
            risk_factors.append("Very high transaction limit requested")
            risk_score += 15
        
        # Check geographical risk (based on FATF high-risk jurisdictions and AML standards)
        # Note: This list should be regularly updated based on official FATF publications
        high_risk_countries = [
            'iran', 'north korea', 'myanmar', 'afghanistan', 'syria',
            'yemen', 'zimbabwe', 'belarus', 'pakistan', 'uganda',
            'south sudan', 'mali', 'mozambique', 'burkina faso',
            'senegal', 'kenya', 'nicaragua', 'haiti', 'jamaica'
        ]
        # Additional monitoring jurisdictions (lower risk but require enhanced due diligence)
        enhanced_monitoring_countries = [
            'cayman islands', 'panama', 'uae', 'philippines',
            'albania', 'barbados', 'mauritius', 'morocco'
        ]
        
        country = business_info.get('country', '').lower()
        if country in high_risk_countries:
            risk_factors.append(f"High-risk jurisdiction per FATF: {country}")
            risk_score += 25
        elif country in enhanced_monitoring_countries:
            risk_factors.append(f"Enhanced monitoring jurisdiction: {country}")
            risk_score += 15
        
        # Determine risk level and recommendation
        if risk_score >= 60:
            risk_level = 'HIGH'
            recommendation = 'REJECT or require additional verification'
        elif risk_score >= 40:
            risk_level = 'MEDIUM'
            recommendation = 'APPROVE with monitoring and lower limits'
        elif risk_score >= 20:
            risk_level = 'LOW'
            recommendation = 'APPROVE with standard monitoring'
        else:
            risk_level = 'VERY_LOW'
            recommendation = 'APPROVE'
        
        assessment = {
            'merchant_id': merchant_data.get('merchant_id'),
            'assessment_date': datetime.now().isoformat(),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'suggested_transaction_limit': self._suggest_transaction_limit(risk_score),
            'required_verifications': self._suggest_verifications(risk_factors),
            'monitoring_intensity': risk_level
        }
        
        return assessment
    
    def _suggest_transaction_limit(self, risk_score: int) -> int:
        """Suggest appropriate transaction limit based on risk."""
        if risk_score >= 60:
            return 10000  # Low limit for high risk
        elif risk_score >= 40:
            return 50000  # Medium limit
        elif risk_score >= 20:
            return 100000  # Standard limit
        else:
            return 500000  # Higher limit for low risk
    
    def _suggest_verifications(self, risk_factors: List[str]) -> List[str]:
        """Suggest additional verifications based on risk factors."""
        verifications = []
        
        if any('ownership' in factor.lower() for factor in risk_factors):
            verifications.append("Enhanced ownership verification")
        
        if any('registered' in factor.lower() for factor in risk_factors):
            verifications.append("Business registration verification")
        
        if any('industry' in factor.lower() for factor in risk_factors):
            verifications.append("Industry-specific compliance checks")
        
        if any('country' in factor.lower() for factor in risk_factors):
            verifications.append("Enhanced due diligence for jurisdiction")
        
        return verifications
