"""
LLM-powered risk assessment and merchant communication service.

This module provides LLM integration for automated transaction risk explanation,
merchant communication, and fraud investigation assistance.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RiskExplanation:
    """Data class for risk explanation results."""
    risk_level: str
    explanation: str
    recommendations: List[str]
    confidence_score: float
    language: str = "en"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class LLMRiskAssessmentService:
    """
    LLM-powered transaction risk assessment and explanation service.
    
    This service uses Large Language Models to generate natural language
    explanations of transaction risks and create personalized merchant alerts.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 max_tokens: int = 500):
        """
        Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum tokens to generate
        """
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. LLM features will be disabled.")
            self.enabled = False
            return
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. LLM features will be disabled.")
            self.enabled = False
            return
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enabled = True
        
        openai.api_key = self.api_key
        
        logger.info(f"LLM service initialized with model: {model}")
    
    def analyze_transaction_risk(self,
                                transaction_data: Dict,
                                risk_score: float,
                                detection_flags: Dict,
                                language: str = "en") -> RiskExplanation:
        """
        Generate natural language risk explanation for a transaction.
        
        Args:
            transaction_data: Transaction details
            risk_score: Calculated risk score (0-10)
            detection_flags: Boolean flags from different detection methods
            language: Target language for explanation
            
        Returns:
            RiskExplanation object with detailed analysis
        """
        if not self.enabled:
            return self._get_fallback_explanation(risk_score, language)
        
        prompt = self._build_risk_analysis_prompt(
            transaction_data, risk_score, detection_flags, language
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(language)},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            explanation_text = response.choices[0].message.content
            
            return self._parse_llm_response(explanation_text, risk_score, language)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return self._get_fallback_explanation(risk_score, language)
    
    def generate_merchant_alert(self,
                              merchant_id: str,
                              transaction_data: Dict,
                              risk_explanation: RiskExplanation) -> str:
        """
        Generate personalized merchant alert message.
        
        Args:
            merchant_id: Merchant identifier
            transaction_data: Transaction details
            risk_explanation: Risk explanation from analyze_transaction_risk
            
        Returns:
            Formatted alert message string
        """
        if not self.enabled:
            return self._get_fallback_alert(
                merchant_id, transaction_data, risk_explanation
            )
        
        prompt = self._build_merchant_alert_prompt(
            merchant_id, transaction_data, risk_explanation
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_merchant_communication_prompt(
                        risk_explanation.language
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Alert generation failed: {str(e)}")
            return self._get_fallback_alert(merchant_id, transaction_data, risk_explanation)
    
    def generate_investigation_summary(self,
                                      transaction_history: List[Dict],
                                      network_info: Dict,
                                      risk_patterns: List[str]) -> str:
        """
        Generate investigation summary for fraud analysts.
        
        Args:
            transaction_history: List of transaction dictionaries
            network_info: Network analysis results
            risk_patterns: List of identified risk patterns
            
        Returns:
            Investigation summary text
        """
        if not self.enabled:
            return "LLM service not available for investigation summary."
        
        prompt = self._build_investigation_prompt(
            transaction_history, network_info, risk_patterns
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_investigation_assistant_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Investigation summary generation failed: {str(e)}")
            return "Failed to generate investigation summary."
    
    def _get_system_prompt(self, language: str = "en") -> str:
        """Get system prompt for risk analysis."""
        prompts = {
            "en": """You are an expert financial fraud analyst specializing in transaction risk assessment.
Your role is to explain transaction risks in clear, professional language that helps merchants 
understand potential issues without causing unnecessary alarm. Focus on:
1. Specific risk indicators found
2. Why these patterns are concerning
3. Actionable recommendations
4. Regulatory compliance context""",
            
            "no": """Du er en ekspert på økonomisk svindel som spesialiserer seg på vurdering av transaksjonsrisiko.
Din rolle er å forklare transaksjonsrisiko på klart, profesjonelt språk som hjelper kjøpmenn 
forstå potensielle problemer uten å forårsake unødvendig alarm. Fokuser på:
1. Spesifikke risikoindikatorer funnet
2. Hvorfor disse mønstrene er bekymringsfulle
3. Handlingsrettede anbefalinger
4. Kontekst for regulatorisk overholdelse"""
        }
        
        return prompts.get(language, prompts["en"])
    
    def _get_merchant_communication_prompt(self, language: str = "en") -> str:
        """Get system prompt for merchant communication."""
        prompts = {
            "en": """You are a customer success specialist at a payment processing company.
Your role is to communicate with merchants about transaction risks in a friendly, helpful manner.
Always be professional, clear, and supportive. Provide specific guidance on next steps.""",
            
            "no": """Du er en kundesuksess-spesialist hos et betalingsbehandlingsfirma.
Din rolle er å kommunisere med kjøpmenn om transaksjonsrisiko på en vennlig, hjelpsom måte.
Vær alltid profesjonell, klar og støttende. Gi spesifikk veiledning om neste skritt."""
        }
        
        return prompts.get(language, prompts["en"])
    
    def _get_investigation_assistant_prompt(self) -> str:
        """Get system prompt for investigation assistant."""
        return """You are an AI assistant helping fraud investigators analyze complex cases.
Your role is to:
1. Summarize key findings from transaction data
2. Identify suspicious patterns and connections
3. Suggest investigation paths
4. Highlight evidence that needs further review
5. Provide clear, actionable insights

Be concise, factual, and organized in your summaries."""
    
    def _build_risk_analysis_prompt(self,
                                   transaction_data: Dict,
                                   risk_score: float,
                                   detection_flags: Dict,
                                   language: str) -> str:
        """Build prompt for risk analysis."""
        # Build flags summary
        flags_summary = []
        if detection_flags.get('rule_based'):
            flags_summary.append("- AML rule-based scenario triggered")
        if detection_flags.get('ml'):
            flags_summary.append("- Machine learning model flagged as anomalous")
        if detection_flags.get('network'):
            flags_summary.append("- Suspicious network patterns detected")
        
        lang_instruction = {
            "en": "Respond in English.",
            "no": "Svar på norsk (bokmål).",
            "sv": "Svara på svenska.",
            "dk": "Svar på dansk."
        }.get(language, "Respond in English.")
        
        prompt = f"""{lang_instruction}

Analyze this transaction and explain the risk:

Transaction Details:
- Amount: {transaction_data.get('amount', 'N/A')}
- Type: {transaction_data.get('type', 'N/A')}
- Origin Account: {transaction_data.get('nameOrig', 'N/A')}
- Destination Account: {transaction_data.get('nameDest', 'N/A')}

Risk Score: {risk_score:.2f} / 10.0

Detection Flags:
{chr(10).join(flags_summary) if flags_summary else "- No specific flags"}

Additional Context:
- Origin Balance Before: {transaction_data.get('oldbalanceOrg', 'N/A')}
- Origin Balance After: {transaction_data.get('newbalanceOrig', 'N/A')}
- Destination Balance Before: {transaction_data.get('oldbalanceDest', 'N/A')}
- Destination Balance After: {transaction_data.get('newbalanceDest', 'N/A')}

Provide:
1. Risk Level (LOW/MEDIUM/HIGH/CRITICAL)
2. Clear explanation of why this transaction is flagged (2-3 sentences)
3. 2-3 specific recommendations for the merchant
4. Confidence score (0.0-1.0)

Format your response as:
RISK_LEVEL|Explanation text here|Recommendation 1;Recommendation 2;Recommendation 3|Confidence"""
        
        return prompt
    
    def _build_merchant_alert_prompt(self,
                                    merchant_id: str,
                                    transaction_data: Dict,
                                    risk_explanation: RiskExplanation) -> str:
        """Build prompt for merchant alert generation."""
        language_map = {
            "en": "in English",
            "no": "in Norwegian (Bokmål)",
            "sv": "in Swedish",
            "dk": "in Danish"
        }
        
        lang_instruction = language_map.get(risk_explanation.language, "in English")
        
        prompt = f"""Create a merchant alert message {lang_instruction}:

Merchant ID: {merchant_id}
Risk Level: {risk_explanation.risk_level}
Transaction Amount: {transaction_data.get('amount', 'N/A')} {transaction_data.get('currency', '')}
Transaction Type: {transaction_data.get('type', 'N/A')}

Risk Explanation:
{risk_explanation.explanation}

Recommendations:
{chr(10).join(f"- {rec}" for rec in risk_explanation.recommendations)}

Create a professional, friendly message that:
1. Acknowledges the transaction
2. Explains the concern clearly and simply
3. Provides specific next steps
4. Maintains a supportive, helpful tone
5. Is concise (under 150 words)

Do not include subject line or formal headers, just the message body."""
        
        return prompt
    
    def _build_investigation_prompt(self,
                                   transaction_history: List[Dict],
                                   network_info: Dict,
                                   risk_patterns: List[str]) -> str:
        """Build prompt for investigation summary."""
        # Summarize transaction history
        total_transactions = len(transaction_history)
        total_amount = sum(t.get('amount', 0) for t in transaction_history)
        
        # Get unique accounts
        origins = set(t.get('nameOrig', '') for t in transaction_history)
        destinations = set(t.get('nameDest', '') for t in transaction_history)
        
        prompt = f"""Generate an investigation summary for this fraud case:

Transaction History:
- Total Transactions: {total_transactions}
- Total Amount: {total_amount:,.2f}
- Unique Origin Accounts: {len(origins)}
- Unique Destination Accounts: {len(destinations)}
- Time Span: {transaction_history[0].get('step', 0)} to {transaction_history[-1].get('step', 0)}

Network Analysis:
- Suspicious Accounts: {network_info.get('suspicious_accounts_count', 0)}
- Cycles Detected: {network_info.get('cycles_count', 0)}
- Fan Patterns: {network_info.get('fan_patterns_count', 0)}
- Communities: {network_info.get('communities_count', 0)}

Risk Patterns Identified:
{chr(10).join(f"- {pattern}" for pattern in risk_patterns)}

Sample Transactions:
{json.dumps(transaction_history[:5], indent=2, default=str)}

Provide:
1. Executive Summary (2-3 sentences)
2. Key Findings (3-5 bullet points)
3. Suspicious Patterns (2-3 specific patterns)
4. Recommended Actions (3-4 specific next steps)
5. Evidence to Collect (2-3 items)

Keep the summary organized, concise, and actionable."""
        
        return prompt
    
    def _parse_llm_response(self,
                           response: str,
                           risk_score: float,
                           language: str) -> RiskExplanation:
        """Parse LLM response into structured format."""
        try:
            # Try to parse structured format
            parts = response.split('|')
            
            if len(parts) >= 4:
                risk_level = parts[0].strip()
                explanation = parts[1].strip()
                recommendations = [r.strip() for r in parts[2].split(';') if r.strip()]
                confidence = float(parts[3].strip())
            else:
                # Fallback parsing if format is not followed
                risk_level = self._determine_risk_level(risk_score)
                explanation = response[:200]  # Take first 200 chars
                recommendations = [
                    "Review transaction details carefully",
                    "Contact customer if needed",
                    "Monitor for similar patterns"
                ]
                confidence = 0.7
            
            return RiskExplanation(
                risk_level=risk_level,
                explanation=explanation,
                recommendations=recommendations,
                confidence_score=min(max(confidence, 0.0), 1.0),
                language=language
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return self._get_fallback_explanation(risk_score, language)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from numeric score."""
        if risk_score >= 7.5:
            return "CRITICAL"
        elif risk_score >= 5.0:
            return "HIGH"
        elif risk_score >= 2.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_fallback_explanation(self,
                                  risk_score: float,
                                  language: str = "en") -> RiskExplanation:
        """Provide fallback explanation if LLM fails."""
        risk_level = self._determine_risk_level(risk_score)
        
        explanations = {
            "en": {
                "CRITICAL": "This transaction shows multiple high-risk indicators requiring immediate review and potential blocking.",
                "HIGH": "This transaction exhibits patterns consistent with potentially fraudulent activity and should be investigated.",
                "MEDIUM": "This transaction shows some unusual patterns that warrant monitoring and possible review.",
                "LOW": "This transaction shows minor irregularities but appears within acceptable risk parameters."
            },
            "no": {
                "CRITICAL": "Denne transaksjonen viser flere høyrisikoindikatorer som krever umiddelbar gjennomgang og potensiell blokkering.",
                "HIGH": "Denne transaksjonen viser mønstre som samsvarer med potensielt uredelig aktivitet og bør undersøkes.",
                "MEDIUM": "Denne transaksjonen viser noen uvanlige mønstre som krever overvåking og mulig gjennomgang.",
                "LOW": "Denne transaksjonen viser mindre uregelmessigheter, men ser ut til å være innenfor akseptable risikoparametere."
            }
        }
        
        lang_explanations = explanations.get(language, explanations["en"])
        
        recommendations = {
            "en": [
                "Review transaction history for patterns",
                "Verify customer identity if needed",
                "Monitor account for similar activity"
            ],
            "no": [
                "Gjennomgå transaksjonshistorikk for mønstre",
                "Bekreft kundeidentitet om nødvendig",
                "Overvåk konto for lignende aktivitet"
            ]
        }
        
        lang_recommendations = recommendations.get(language, recommendations["en"])
        
        return RiskExplanation(
            risk_level=risk_level,
            explanation=lang_explanations[risk_level],
            recommendations=lang_recommendations,
            confidence_score=0.6,
            language=language
        )
    
    def _get_fallback_alert(self,
                           merchant_id: str,
                           transaction_data: Dict,
                           risk_explanation: RiskExplanation) -> str:
        """Provide fallback alert message if LLM fails."""
        templates = {
            "en": """Dear Merchant,

We've detected a transaction requiring your attention:

Transaction Amount: {amount}
Risk Level: {risk_level}

{explanation}

Recommended Actions:
{recommendations}

If you have any questions or concerns, please contact our support team.

Best regards,
Fraud Prevention Team""",
            
            "no": """Kjære kjøpmann,

Vi har oppdaget en transaksjon som krever din oppmerksomhet:

Transaksjonsbeløp: {amount}
Risikonivå: {risk_level}

{explanation}

Anbefalte handlinger:
{recommendations}

Hvis du har spørsmål eller bekymringer, vennligst kontakt vårt supportteam.

Med vennlig hilsen,
Svindelforebyggingsteamet"""
        }
        
        template = templates.get(risk_explanation.language, templates["en"])
        
        recommendations_text = '\n'.join(
            f"• {rec}" for rec in risk_explanation.recommendations
        )
        
        return template.format(
            amount=transaction_data.get('amount', 'N/A'),
            risk_level=risk_explanation.risk_level,
            explanation=risk_explanation.explanation,
            recommendations=recommendations_text
        )
