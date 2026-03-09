"""
Pydantic models and schemas for ClinAssist
Defines all request/response structures and validation rules
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ProgressionEnum(str, Enum):
    improving = "improving"
    worsening = "worsening"
    stable = "stable"


class OnsetTypeEnum(str, Enum):
    sudden = "sudden"
    gradual = "gradual"


class RiskLevelEnum(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SessionCreate(BaseModel):
    """Response model for creating a new session"""
    session_id: str
    message: str


class VoiceInput(BaseModel):
    """Request model for voice input (file upload handled separately)"""
    pass


class TextInput(BaseModel):
    """Request model for text input"""
    text: str = Field(..., min_length=0, max_length=1000)


class TranscriptResponse(BaseModel):
    """Response model for transcription"""
    transcript: str


class SymptomRecord(BaseModel):
    """Structured symptom data model"""
    chief_complaint: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[int] = Field(None, ge=1, le=10)
    progression: Optional[ProgressionEnum] = None
    associated_symptoms: Optional[List[str]] = None
    affected_body_part: Optional[str] = None
    onset_type: Optional[OnsetTypeEnum] = None
    aggravating_alleviating_factors: Optional[str] = None
    relevant_medical_history: Optional[str] = None
    
    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v):
        if v is not None and (v < 1 or v > 10):
            raise ValueError('Severity must be between 1 and 10')
        return v


class RiskAssessment(BaseModel):
    """Risk categorization result"""
    risk_level: RiskLevelEnum
    reason: str
    recommended_action: str


class LatencyBreakdown(BaseModel):
    """Latency metrics for a single interaction"""
    stt_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    tts_ms: Optional[float] = None
    total_ms: float


class SessionResponse(BaseModel):
    """Complete response for voice/text interaction"""
    transcript: Optional[str] = None
    response_text: str
    audio_base64: Optional[str] = None
    state: str
    is_complete: bool
    risk_assessment: Optional[RiskAssessment] = None
    latency_breakdown: LatencyBreakdown
    symptom_progress: Optional[Dict[str, bool]] = None
    wellness_tip: Optional[str] = None
    trigger_consult_transition: Optional[bool] = False
    show_report_popup: Optional[bool] = False


class SummaryResponse(BaseModel):
    """Response for session summary endpoint"""
    session_id: str
    symptom_record: SymptomRecord
    risk_assessment: Optional[RiskAssessment] = None
    summary: Optional[str] = None
    state: str


class SessionHistoryItem(BaseModel):
    id: str
    created_at: str
    state: str
    session_type: Optional[str] = None
    chief_complaint: Optional[str] = None
    risk_level: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    sessions: List[SessionHistoryItem]


class FullSessionResponse(BaseModel):
    session_id: str
    created_at: str
    state: str
    conversation_history: List[Dict[str, str]]
    symptom_record: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[RiskAssessment] = None
    wellness_tip: Optional[str] = None


class ExportData(BaseModel):
    """Formatted export data for physician handoff"""
    session_id: str
    timestamp: str
    conversation_history: List[Dict[str, str]]
    structured_data: SymptomRecord
    risk_assessment: Optional[RiskAssessment] = None
    summary: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str


class ConsultRequest(BaseModel):
    """Request model for standalone Health Consult"""
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    context_session_id: Optional[str] = None


class ConsultResponse(BaseModel):
    """Response model for standalone Health Consult"""
    answer: str
    audio_base64: Optional[str] = None
    session_id: str


class ConsultVoiceResponse(BaseModel):
    """Response model for standalone voice Health Consult"""
    answer: str
    transcript: str
    audio_base64: Optional[str] = None
    session_id: str
