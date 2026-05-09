from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4


# Shared schemas
class Metadata(BaseModel):
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    cost_usd: Optional[float] = None
    
    class Config:
        from_attributes = True


class StepBase(BaseModel):
    step_type: str = Field(..., description="think, tool_call, tool_result, observation, decision, error")
    agent_id: Optional[str] = None
    model: Optional[str] = None
    content: str
    metadata: Optional[Metadata] = None
    context_snapshot: Optional[str] = None
    
    @validator('step_type')
    def validate_step_type(cls, v):
        valid_types = {"think", "tool_call", "tool_result", "observation", "decision", "error"}
        if v not in valid_types:
            raise ValueError(f"step_type must be one of {valid_types}")
        return v


# Session schemas
class SessionCreate(BaseModel):
    name: Optional[str] = None
    agent_id: str
    model: str
    metadata: Optional[Dict[str, Any]] = None


class SessionUpdate(BaseModel):
    name: Optional[str] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionStepCreate(StepBase):
    step_number: int


class SessionStepBatchCreate(BaseModel):
    steps: List[SessionStepCreate]


class SessionStepResponse(StepBase):
    id: str
    session_id: str
    step_number: int
    timestamp: datetime
    
    class Config:
        from_attributes = True


class AgentSessionResponse(BaseModel):
    id: str
    name: Optional[str]
    agent_id: str
    model: str
    created_at: datetime
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_steps: int
    total_decisions: int
    metadata: Dict[str, Any]
    steps: List[SessionStepResponse] = []
    
    class Config:
        from_attributes = True


# Timeline schemas
class DecisionCycle(BaseModel):
    cycle_number: int
    start_step: int
    end_step: int
    steps: List[SessionStepResponse]
    latency_ms: Optional[float] = None
    decision_type: Optional[str] = None


class TimelineResponse(BaseModel):
    session_id: str
    total_steps: int
    decision_cycles: List[DecisionCycle]
    step_density: Dict[str, int]  # counts by step_type


# Snapshot schemas
class StateSnapshot(BaseModel):
    step_number: int
    context_window: str
    conversation_history: List[str]
    tool_outputs_received: List[Dict[str, Any]]
    active_goals: List[str]
    memory_state: Optional[Dict[str, Any]] = None
    constraints: Optional[List[str]] = None


# Replay schemas
class ReplayConfig(BaseModel):
    model: str
    temperature: Optional[float] = 0.7
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    tool_config: Optional[Dict[str, Any]] = None
    constraints: Optional[List[str]] = None


class ReplayCreate(BaseModel):
    name: Optional[str] = None
    start_step: int = Field(..., ge=1)
    end_step: int = Field(..., ge=1)
    replay_config: ReplayConfig
    
    @validator('end_step')
    def validate_step_range(cls, v, values):
        if 'start_step' in values and v < values['start_step']:
            raise ValueError("end_step must be >= start_step")
        return v


class ReplayStepDiff(BaseModel):
    step_number: int
    original_content: str
    replayed_content: str
    divergence_score: float
    diff_html: Optional[str] = None


class ReplayMetrics(BaseModel):
    total_steps: int
    steps_diverged: int
    avg_divergence_score: float
    latency_delta_ms: float
    token_delta: int
    outcome_diff: Optional[str] = None


class ReplayComparisonResponse(BaseModel):
    replay_id: str
    baseline_type: str
    baseline_id: Optional[str]
    metrics: ReplayMetrics
    step_diffs: List[ReplayStepDiff]


class ReplayResponse(BaseModel):
    id: str
    session_id: str
    name: Optional[str]
    start_step: int
    end_step: int
    replay_config: ReplayConfig
    created_at: datetime
    completed_at: Optional[datetime]
    total_steps: int
    status: str
    metrics: Optional[ReplayMetrics] = None
    
    class Config:
        from_attributes = True


# Dashboard schemas
class DashboardStats(BaseModel):
    total_sessions: int
    total_steps: int
    total_decisions: int
    avg_steps_per_session: float
    avg_decisions_per_session: float
    most_common_error_patterns: List[Dict[str, Any]]
    step_type_distribution: Dict[str, int]
    recent_sessions: List[AgentSessionResponse]
    active_replays: List[ReplayResponse]


# WebSocket schemas
class WebSocketMessage(BaseModel):
    type: str  # step_added, session_completed, error
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)