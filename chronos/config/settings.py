"""
CHRONOS Configuration and Customization

Configurable risk thresholds, alert rules, and compliance settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class RiskCategory(Enum):
    """Risk categories for classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskThresholds:
    """
    Configurable risk score thresholds.
    
    Default thresholds based on industry standards:
    - LOW: 0.0 - 0.4 (minimal investigation)
    - MEDIUM: 0.4 - 0.7 (enhanced due diligence)
    - HIGH: 0.7 - 0.9 (priority review)
    - CRITICAL: 0.9 - 1.0 (immediate escalation)
    """
    low_max: float = 0.4
    medium_max: float = 0.7
    high_max: float = 0.9
    
    def get_category(self, score: float) -> RiskCategory:
        """Get risk category for a given score."""
        if score < self.low_max:
            return RiskCategory.LOW
        elif score < self.medium_max:
            return RiskCategory.MEDIUM
        elif score < self.high_max:
            return RiskCategory.HIGH
        else:
            return RiskCategory.CRITICAL
    
    @classmethod
    def from_env(cls) -> "RiskThresholds":
        """Load thresholds from environment variables."""
        return cls(
            low_max=float(os.getenv("RISK_THRESHOLD_LOW", "0.4")),
            medium_max=float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.7")),
            high_max=float(os.getenv("RISK_THRESHOLD_HIGH", "0.9")),
        )


@dataclass
class AlertRule:
    """Custom alert rule definition."""
    name: str
    description: str
    min_risk_score: float
    max_transactions_per_hour: Optional[int] = None
    min_transaction_value: Optional[float] = None
    required_features: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 1  # 1 = highest
    
    def matches(self, risk_score: float, features: Dict = None) -> bool:
        """Check if transaction matches this alert rule."""
        if not self.enabled:
            return False
        if risk_score < self.min_risk_score:
            return False
        return True


@dataclass  
class ComplianceConfig:
    """
    Compliance configuration for different jurisdictions.
    
    Supports:
    - EU AI Act requirements
    - FinCEN guidelines
    - FATF recommendations
    """
    jurisdiction: str = "US"
    
    # Reporting thresholds (in USD equivalent)
    sar_threshold: float = 10000.0  # Suspicious Activity Report
    ctr_threshold: float = 10000.0  # Currency Transaction Report
    
    # Record keeping (days)
    transaction_retention_days: int = 1825  # 5 years
    audit_log_retention_days: int = 2555    # 7 years
    
    # Model explainability requirements
    require_explanation: bool = True
    min_explanation_features: int = 3
    
    # Human review requirements
    auto_approve_low_risk: bool = True
    require_human_review_high_risk: bool = True
    
    @classmethod
    def for_jurisdiction(cls, jurisdiction: str) -> "ComplianceConfig":
        """Get compliance config for specific jurisdiction."""
        configs = {
            "US": cls(jurisdiction="US", sar_threshold=10000, ctr_threshold=10000),
            "EU": cls(jurisdiction="EU", sar_threshold=15000, ctr_threshold=15000,
                     require_explanation=True, min_explanation_features=5),
            "UK": cls(jurisdiction="UK", sar_threshold=10000, ctr_threshold=10000),
            "APAC": cls(jurisdiction="APAC", sar_threshold=10000, ctr_threshold=10000),
        }
        return configs.get(jurisdiction, cls())


# Default alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="critical_risk",
        description="Critical risk score detected",
        min_risk_score=0.95,
        priority=1
    ),
    AlertRule(
        name="high_risk",
        description="High risk transaction requiring review",
        min_risk_score=0.85,
        priority=2
    ),
    AlertRule(
        name="elevated_risk",
        description="Elevated risk - enhanced monitoring",
        min_risk_score=0.70,
        priority=3
    ),
    AlertRule(
        name="suspicious_pattern",
        description="Transaction matches suspicious pattern indicators",
        min_risk_score=0.60,
        priority=4
    ),
]


class AlertManager:
    """
    Manages alert generation and routing.
    
    Usage:
        manager = AlertManager()
        alerts = manager.evaluate(risk_score=0.92, features={...})
    """
    
    def __init__(self, 
                 rules: List[AlertRule] = None,
                 thresholds: RiskThresholds = None):
        self.rules = rules or DEFAULT_ALERT_RULES
        self.thresholds = thresholds or RiskThresholds.from_env()
    
    def evaluate(self, 
                 transaction_id: str,
                 risk_score: float, 
                 features: Dict = None) -> List[Dict]:
        """
        Evaluate transaction against all alert rules.
        
        Returns list of triggered alerts.
        """
        triggered = []
        
        for rule in self.rules:
            if rule.matches(risk_score, features):
                triggered.append({
                    "transaction_id": transaction_id,
                    "rule_name": rule.name,
                    "rule_description": rule.description,
                    "risk_score": risk_score,
                    "risk_category": self.thresholds.get_category(risk_score).value,
                    "priority": rule.priority,
                    "action_required": rule.priority <= 2
                })
        
        # Sort by priority
        triggered.sort(key=lambda x: x["priority"])
        
        return triggered
    
    def add_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.rules.append(rule)
    
    def remove_rule(self, name: str):
        """Remove an alert rule by name."""
        self.rules = [r for r in self.rules if r.name != name]
    
    def enable_rule(self, name: str):
        """Enable an alert rule."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
    
    def disable_rule(self, name: str):
        """Disable an alert rule."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False


# Export
__all__ = [
    "RiskCategory",
    "RiskThresholds", 
    "AlertRule",
    "AlertManager",
    "ComplianceConfig",
    "DEFAULT_ALERT_RULES"
]
