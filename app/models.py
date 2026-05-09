from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, Boolean, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class AgentSession(Base):
    """Represents a complete agent session."""
    __tablename__ = "agent_sessions"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    agent_id = Column(String, index=True)
    model = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_steps = Column(Integer, default=0)
    total_decisions = Column(Integer, default=0)
    meta_data = Column("meta_data", JSON, default=dict)
    
    # Relationships
    steps = relationship("SessionStep", back_populates="session", cascade="all, delete-orphan")
    replays = relationship("Replay", back_populates="session", cascade="all, delete-orphan")


class SessionStep(Base):
    """Represents a single step in an agent session."""
    __tablename__ = "session_steps"
    
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("agent_sessions.id", ondelete="CASCADE"), index=True)
    step_number = Column(Integer, nullable=False)
    step_type = Column(String, nullable=False)  # think, tool_call, tool_result, observation, decision, error
    agent_id = Column(String, index=True)
    model = Column(String)
    content = Column(Text, nullable=False)
    meta_data = Column("meta_data", JSON, default=dict)  # tokens_used, latency_ms, tool_name, tool_args, etc.
    context_snapshot = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("AgentSession", back_populates="steps")
    
    # Index
    __table_args__ = (
        (Index('ix_session_steps_session_step_number', 'session_id', 'step_number')),
    )


class DecisionPoint(Base):
    """Represents a significant decision point in a session."""
    __tablename__ = "decision_points"
    
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("agent_sessions.id", ondelete="CASCADE"), index=True)
    step_id = Column(String, ForeignKey("session_steps.id", ondelete="CASCADE"), index=True)
    decision_type = Column(String)  # tool_selection, parameter_choice, goal_change, etc.
    options_considered = Column(JSON, default=list)
    chosen_option = Column(Text)
    reasoning = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Replay(Base):
    """Represents a replay of a session or session segment."""
    __tablename__ = "replays"
    
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("agent_sessions.id", ondelete="CASCADE"), index=True)
    name = Column(String, nullable=True)
    start_step = Column(Integer, nullable=False)
    end_step = Column(Integer, nullable=False)
    replay_config = Column(JSON, nullable=False)  # model, temperature, system_prompt, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    total_steps = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending, running, completed, failed
    
    # Relationships
    session = relationship("AgentSession", back_populates="replays")
    replay_steps = relationship("ReplayStep", back_populates="replay", cascade="all, delete-orphan")
    comparisons = relationship("ReplayComparison", back_populates="replay", cascade="all, delete-orphan")


class ReplayStep(Base):
    """Represents a step in a replay execution."""
    __tablename__ = "replay_steps"
    
    id = Column(String, primary_key=True, index=True)
    replay_id = Column(String, ForeignKey("replays.id", ondelete="CASCADE"), index=True)
    original_step_id = Column(String, ForeignKey("session_steps.id", ondelete="CASCADE"), index=True)
    step_number = Column(Integer, nullable=False)
    step_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    original_content = Column(Text, nullable=False)
    meta_data = Column("meta_data", JSON, default=dict)
    divergence_score = Column(Float, nullable=True)  # 0-1 similarity to original
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    replay = relationship("Replay", back_populates="replay_steps")


class ReplayComparison(Base):
    """Represents a comparison between a replay and its baseline."""
    __tablename__ = "replay_comparisons"
    
    id = Column(String, primary_key=True, index=True)
    replay_id = Column(String, ForeignKey("replays.id", ondelete="CASCADE"), index=True)
    baseline_type = Column(String)  # original, another_replay_id
    baseline_id = Column(String, nullable=True)
    comparison_metrics = Column(JSON, nullable=False)  # step_diffs, latency_delta, token_delta, outcome_diff
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    replay = relationship("Replay", back_populates="comparisons")