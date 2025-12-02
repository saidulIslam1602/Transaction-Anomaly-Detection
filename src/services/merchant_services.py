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
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize merchant risk intelligence service.
        
        Args:
            config: Configuration dictionary with merchant services settings
        """
        self.merchant_profiles = {}
        self.config = config or {}
        
        # Load risk thresholds from config
        risk_thresholds = self.config.get('merchant_services', {}).get('risk_thresholds', {})
        self.high_risk_threshold = risk_thresholds.get('high_risk', 7.0)
        self.medium_risk_threshold = risk_thresholds.get('medium_risk', 4.0)
        
        # Load recommendation thresholds
        rec_config = self.config.get('merchant_services', {}).get('recommendations', {})
        self.fraud_rate_threshold = rec_config.get('fraud_rate_threshold', 0.05)
        self.risk_score_threshold = rec_config.get('risk_score_threshold', 6.0)
        self.min_transactions_threshold = rec_config.get('min_transactions_threshold', 10)
        
        # Load health score parameters
        health_config = self.config.get('merchant_services', {}).get('health_score', {})
        self.health_base_score = health_config.get('base_score', 100.0)
        self.health_fraud_multiplier = health_config.get('fraud_rate_multiplier', 500)
        self.health_risk_threshold = health_config.get('risk_score_threshold', 5.0)
        self.health_risk_penalty = health_config.get('risk_score_penalty', 10.0)
        
        # Load industry benchmarks
        self.industry_benchmarks = self.config.get('business_metrics', {}).get('industry_benchmarks', {
            'avg_fraud_rate': 0.02,
            'avg_risk_score': 3.5,
            'avg_transaction_amount': 5000.0
        })
        
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
        
        # Calculate risk level using config thresholds
        if profile.average_risk_score >= self.high_risk_threshold:
            risk_level = 'HIGH'
        elif profile.average_risk_score >= self.medium_risk_threshold:
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
        
        if profile.fraud_rate > self.fraud_rate_threshold:
            recommendations.append(
                "High fraud rate detected. Consider implementing additional verification steps."
            )
        
        if profile.average_risk_score > self.risk_score_threshold:
            recommendations.append(
                "Average risk score is elevated. Review high-risk transactions manually."
            )
        
        if profile.total_transactions < self.min_transactions_threshold:
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
        # Base score from config
        health_score = self.health_base_score
        
        # Deduct for fraud rate using config multiplier
        health_score -= profile.fraud_rate * self.health_fraud_multiplier
        
        # Deduct for high risk scores using config threshold and penalty
        if profile.average_risk_score > self.health_risk_threshold:
            health_score -= (profile.average_risk_score - self.health_risk_threshold) * self.health_risk_penalty
        
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
        
        # Use provided industry stats or default from config
        if industry_stats is None:
            industry_stats = self.industry_benchmarks
        
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
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize alert prioritization system.
        
        Args:
            config: Configuration dictionary with alert prioritization settings
        """
        self.alert_history = []
        self.config = config or {}
        
        # Load alert prioritization config
        alert_config = self.config.get('merchant_services', {}).get('alert_prioritization', {})
        
        # Amount thresholds
        amount_thresholds = alert_config.get('amount_thresholds', {})
        self.amount_critical = amount_thresholds.get('critical', 10000)
        self.amount_high = amount_thresholds.get('high', 5000)
        self.amount_medium = amount_thresholds.get('medium', 1000)
        
        # Detection method scores
        method_scores = alert_config.get('detection_method_scores', {})
        self.rule_based_score = method_scores.get('rule_based', 15)
        self.ml_score = method_scores.get('ml', 10)
        self.network_score = method_scores.get('network', 12)
        self.immediate_action_score = method_scores.get('immediate_action', 25)
        
        # Priority level thresholds
        priority_levels = alert_config.get('priority_levels', {})
        self.critical_threshold = priority_levels.get('critical', 80)
        self.high_threshold = priority_levels.get('high', 60)
        self.medium_threshold = priority_levels.get('medium', 40)
        
        # Fraud rate multiplier
        self.fraud_rate_multiplier = alert_config.get('fraud_rate_multiplier', 50)
        
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
        
        # Amount factor using config thresholds
        amount = alert.get('amount', 0)
        if amount > self.amount_critical:
            score += 20
        elif amount > self.amount_high:
            score += 10
        elif amount > self.amount_medium:
            score += 5
        
        # Historical fraud involvement
        if merchant_context:
            fraud_rate = merchant_context.get('fraud_rate', 0)
            score += fraud_rate * self.fraud_rate_multiplier
        
        # Detection method factor using config scores
        detection_flags = alert.get('detection_flags', {})
        if detection_flags.get('rule_based'):
            score += self.rule_based_score
        if detection_flags.get('ml'):
            score += self.ml_score
        if detection_flags.get('network'):
            score += self.network_score
        
        # Urgency factor
        if alert.get('requires_immediate_action'):
            score += self.immediate_action_score
        
        return score
    
    def _get_priority_level(self, priority_score: float) -> str:
        """Convert priority score to level using config thresholds."""
        if priority_score >= self.critical_threshold:
            return 'CRITICAL'
        elif priority_score >= self.high_threshold:
            return 'HIGH'
        elif priority_score >= self.medium_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'


class MerchantOnboardingRiskAssessment:
    """
    Risk assessment for merchant onboarding process.
    
    Evaluates new merchants to identify high-risk accounts before approval.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize onboarding risk assessment.
        
        Args:
            config: Configuration dictionary with onboarding settings
        """
        self.config = config or {}
        
        # Load onboarding config
        onboard_config = self.config.get('merchant_services', {}).get('onboarding', {})
        
        # Risk score thresholds
        risk_thresholds = onboard_config.get('risk_score_thresholds', {})
        self.reject_threshold = risk_thresholds.get('reject', 60)
        self.monitor_threshold = risk_thresholds.get('monitor', 40)
        self.review_threshold = risk_thresholds.get('review', 20)
        
        # Business age thresholds and scores
        age_thresholds = onboard_config.get('business_age_thresholds', {})
        self.new_business_years = age_thresholds.get('new_business_years', 1)
        self.young_business_years = age_thresholds.get('young_business_years', 3)
        
        age_scores = onboard_config.get('business_age_scores', {})
        self.new_business_score = age_scores.get('new_business', 15)
        self.young_business_score = age_scores.get('young_business', 10)
        
        # Industry risk scores
        self.high_risk_industry_score = onboard_config.get('high_risk_industry_score', 30)
        self.medium_risk_industry_score = onboard_config.get('medium_risk_industry_score', 15)
        
        # Other risk factor scores
        self.ownership_not_verified_score = onboard_config.get('ownership_not_verified_score', 20)
        self.not_registered_score = onboard_config.get('not_registered_score', 30)
        self.high_limit_threshold = onboard_config.get('high_limit_threshold', 1000000)
        self.high_limit_score = onboard_config.get('high_limit_score', 15)
        self.high_risk_country_score = onboard_config.get('high_risk_country_score', 25)
        self.enhanced_monitoring_country_score = onboard_config.get('enhanced_monitoring_country_score', 15)
        
        # Transaction limits
        limits = onboard_config.get('transaction_limits', {})
        self.limit_high_risk = limits.get('high_risk', 10000)
        self.limit_medium_risk = limits.get('medium_risk', 50000)
        self.limit_low_risk = limits.get('low_risk', 100000)
        self.limit_very_low_risk = limits.get('very_low_risk', 500000)
        
        # Load high-risk industries and countries from merchant_services config
        onboarding_assessment = self.config.get('merchant_services', {}).get('onboarding_assessment', {})
        self.high_risk_industries = onboarding_assessment.get('high_risk_industries', [])
        self.medium_risk_industries = onboarding_assessment.get('medium_risk_industries', [])
        self.high_risk_countries = onboarding_assessment.get('high_risk_countries', [])
        self.enhanced_monitoring_countries = onboarding_assessment.get('enhanced_monitoring_countries', [])
        
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
        
        # Check business age using config thresholds
        business_age_years = business_info.get('years_in_operation', 0)
        if business_age_years < self.new_business_years:
            risk_factors.append(f"New business (less than {self.new_business_years} year)")
            risk_score += self.new_business_score
        elif business_age_years < self.young_business_years:
            risk_factors.append(f"Young business (less than {self.young_business_years} years)")
            risk_score += self.young_business_score
        
        # Check business type (using config-loaded industries)
        industry = business_info.get('industry', '').lower()
        if any(risky in industry for risky in self.high_risk_industries):
            risk_factors.append(f"High-risk industry per AML standards: {industry}")
            risk_score += self.high_risk_industry_score
        elif any(medium in industry for medium in self.medium_risk_industries):
            risk_factors.append(f"Medium-risk industry requiring enhanced monitoring: {industry}")
            risk_score += self.medium_risk_industry_score
        
        # Check ownership structure
        if business_info.get('ownership_verified') == False:
            risk_factors.append("Ownership not verified")
            risk_score += self.ownership_not_verified_score
        
        # Check registration information
        if not business_info.get('business_registered'):
            risk_factors.append("Business not properly registered")
            risk_score += self.not_registered_score
        
        # Check requested transaction limits
        requested_limit = merchant_data.get('monthly_transaction_limit', 0)
        if requested_limit > self.high_limit_threshold:
            risk_factors.append("Very high transaction limit requested")
            risk_score += self.high_limit_score
        
        # Check geographical risk (using config-loaded countries)
        country = business_info.get('country', '').lower()
        if country in self.high_risk_countries:
            risk_factors.append(f"High-risk jurisdiction per FATF: {country}")
            risk_score += self.high_risk_country_score
        elif country in self.enhanced_monitoring_countries:
            risk_factors.append(f"Enhanced monitoring jurisdiction: {country}")
            risk_score += self.enhanced_monitoring_country_score
        
        # Determine risk level and recommendation using config thresholds
        if risk_score >= self.reject_threshold:
            risk_level = 'HIGH'
            recommendation = 'REJECT or require additional verification'
        elif risk_score >= self.monitor_threshold:
            risk_level = 'MEDIUM'
            recommendation = 'APPROVE with monitoring and lower limits'
        elif risk_score >= self.review_threshold:
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
        """Suggest appropriate transaction limit based on risk using config values."""
        if risk_score >= self.reject_threshold:
            return self.limit_high_risk
        elif risk_score >= self.monitor_threshold:
            return self.limit_medium_risk
        elif risk_score >= self.review_threshold:
            return self.limit_low_risk
        else:
            return self.limit_very_low_risk
    
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
