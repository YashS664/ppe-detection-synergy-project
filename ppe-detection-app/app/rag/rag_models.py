from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ViolationsSummary(BaseModel):
    """Structured daily safety compliance report"""
    report_date: str = Field(description="Date of report")
    total_violations: int = Field(description="Total violations detected")
    hardhat_violations: int = Field(description="Number of hardhat violations")
    vest_violations: int = Field(description="Number of vest violations")
    total_persons: int = Field(description="Total persons detected")
    most_at_risk_persons_id: Optional[int] = Field(description="Person ID with most violations")
    compliance_rate: float = Field(description="Percentage of compliant persons 0-100")
    severity: str = Field(description="Overall severity: LOW/MEDIUM/HIGH/CRITICAL")
    recommendation: str = Field(description="Safety recommendation based on data")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        allowed = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if v not in allowed:
            raise ValueError(f"Severity must be one of {allowed}")
        return v
    
    @field_validator("compliance_rate")
    @classmethod
    def validate_compliance_rate(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Compliance rate must be between 0-100")
        return round(v, 2)

class WorkerRiskProfile(BaseModel):
    """Risk profile for individual worker"""
    person_id: int 
    total_violations: int 
    hardhat_violations: int
    vest_violations: int
    risk_level: str = Field(description="LOW/MEDIUM/HIGH")
    action_required: str = Field(description="Recommended action")

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v):
        allowed = ["LOW", "MEDIUM", "HIGH"]
        if v not in allowed:
            raise ValueError(f"Risk level must be one of {allowed}")
        return v

class QueryRequest(BaseModel):
    """Natural language query request"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about violations"
    )

class QueryResponse(BaseModel):
    """Query response"""
    question: str
    answer: str
