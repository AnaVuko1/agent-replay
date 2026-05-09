"""
Test the timeline engine functionality.
"""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AgentSession, SessionStep
from app.engine.timeline import TimelineEngine


@pytest.fixture
def timeline_engine():
    """Fixture for timeline engine."""
    return TimelineEngine()


@pytest.fixture
async def sample_session(db_session: AsyncSession):
    """Create a sample session with steps for testing."""
    session = AgentSession(
        id="test-session-1",
        name="Test Session",
        agent_id="test-agent",
        model="test-model",
        total_steps=0,
        total_decisions=0
    )
    db_session.add(session)
    await db_session.flush()
    
    # Create sample steps
    steps = []
    for i in range(1, 11):
        step_type = "think" if i % 2 == 1 else "tool_call"
        
        step = SessionStep(
            id=f"step-{i}",
            session_id=session.id,
            step_number=i,
            step_type=step_type,
            agent_id="test-agent",
            model="test-model",
            content=f"Step {i} content",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=i)
        )
        steps.append(step)
        db_session.add(step)
    
    session.steps = steps
    session.total_steps = len(steps)
    
    await db_session.commit()
    return session


@pytest.mark.asyncio
async def test_build_timeline_empty(timeline_engine: TimelineEngine):
    """Test building timeline for empty session."""
    session = AgentSession(
        id="empty-session",
        name="Empty Session",
        agent_id="test-agent",
        model="test-model",
        total_steps=0,
        total_decisions=0,
        steps=[]
    )
    
    timeline = await timeline_engine.build_timeline(session)
    
    assert timeline.session_id == "empty-session"
    assert timeline.total_steps == 0
    assert len(timeline.decision_cycles) == 0
    assert timeline.step_density == {}


@pytest.mark.asyncio
async def test_build_timeline_with_steps(timeline_engine: TimelineEngine, sample_session: AgentSession):
    """Test building timeline for session with steps."""
    timeline = await timeline_engine.build_timeline(sample_session)
    
    assert timeline.session_id == sample_session.id
    assert timeline.total_steps == len(sample_session.steps)
    assert len(timeline.decision_cycles) > 0
    assert "think" in timeline.step_density
    assert "tool_call" in timeline.step_density


@pytest.mark.asyncio
async def test_decision_cycle_grouping(timeline_engine: TimelineEngine):
    """Test grouping steps into decision cycles."""
    # Create a session with a clear think → tool_call → tool_result pattern
    session = AgentSession(
        id="cycle-test-session",
        name="Cycle Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=5,
        total_decisions=1,
        steps=[]
    )
    
    # Create steps that should form one complete cycle
    steps = [
        SessionStep(
            id="step-1",
            session_id=session.id,
            step_number=1,
            step_type="think",
            content="Thinking about the problem",
            timestamp=datetime.now(timezone.utc)
        ),
        SessionStep(
            id="step-2",
            session_id=session.id,
            step_number=2,
            step_type="tool_call",
            content="Calling a tool",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=1)
        ),
        SessionStep(
            id="step-3",
            session_id=session.id,
            step_number=3,
            step_type="tool_result",
            content="Tool result",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=2)
        ),
        SessionStep(
            id="step-4",
            session_id=session.id,
            step_number=4,
            step_type="observation",
            content="Observing the result",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=3)
        ),
        SessionStep(
            id="step-5",
            session_id=session.id,
            step_number=5,
            step_type="decision",
            content="Making a decision",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=4)
        ),
    ]
    
    session.steps = steps
    
    timeline = await timeline_engine.build_timeline(session)
    
    # Should create one decision cycle with all steps
    assert len(timeline.decision_cycles) == 1
    cycle = timeline.decision_cycles[0]
    assert cycle.start_step == 1
    assert cycle.end_step == 5
    assert len(cycle.steps) == 5
    assert cycle.decision_type is not None
    assert cycle.latency_ms is not None


@pytest.mark.asyncio
async def test_multiple_decision_cycles(timeline_engine: TimelineEngine):
    """Test grouping steps into multiple decision cycles."""
    session = AgentSession(
        id="multi-cycle-session",
        name="Multi Cycle",
        agent_id="test-agent",
        model="test-model",
        total_steps=8,
        total_decisions=2,
        steps=[]
    )
    
    # Create steps that should form two cycles
    steps = [
        # First cycle
        SessionStep(id="s1", session_id=session.id, step_number=1, step_type="think", content="Think 1", timestamp=datetime.now(timezone.utc)),
        SessionStep(id="s2", session_id=session.id, step_number=2, step_type="decision", content="Decide 1", timestamp=datetime.now(timezone.utc) + timedelta(seconds=1)),
        # Second cycle
        SessionStep(id="s3", session_id=session.id, step_number=3, step_type="think", content="Think 2", timestamp=datetime.now(timezone.utc) + timedelta(seconds=2)),
        SessionStep(id="s4", session_id=session.id, step_number=4, step_type="tool_call", content="Call 2", timestamp=datetime.now(timezone.utc) + timedelta(seconds=3)),
        SessionStep(id="s5", session_id=session.id, step_number=5, step_type="tool_result", content="Result 2", timestamp=datetime.now(timezone.utc) + timedelta(seconds=4)),
        SessionStep(id="s6", session_id=session.id, step_number=6, step_type="decision", content="Decide 2", timestamp=datetime.now(timezone.utc) + timedelta(seconds=5)),
        # Additional steps
        SessionStep(id="s7", session_id=session.id, step_number=7, step_type="observation", content="Observe", timestamp=datetime.now(timezone.utc) + timedelta(seconds=6)),
        SessionStep(id="s8", session_id=session.id, step_number=8, step_type="error", content="Error", timestamp=datetime.now(timezone.utc) + timedelta(seconds=7)),
    ]
    
    session.steps = steps
    
    timeline = await timeline_engine.build_timeline(session)
    
    # Should create at least 2 decision cycles
    assert len(timeline.decision_cycles) >= 2
    
    # Check step density
    assert timeline.step_density["think"] == 2
    assert timeline.step_density["decision"] == 2
    assert timeline.step_density["error"] == 1


@pytest.mark.asyncio
async def test_state_snapshot(timeline_engine: TimelineEngine, sample_session: AgentSession):
    """Test creating state snapshot at a specific step."""
    # Add context snapshot to first step
    sample_session.steps[0].context_snapshot = "System: You are a helpful assistant."
    
    snapshot = await timeline_engine.get_state_snapshot(
        sample_session, 
        step_number=5
    )
    
    assert snapshot.step_number == 5
    assert snapshot.context_window is not None
    assert isinstance(snapshot.conversation_history, list)
    assert isinstance(snapshot.tool_outputs_received, list)
    assert isinstance(snapshot.active_goals, list)
    assert isinstance(snapshot.memory_state, dict)
    assert isinstance(snapshot.constraints, list)


@pytest.mark.asyncio
async def test_state_snapshot_edge_cases(timeline_engine: TimelineEngine):
    """Test state snapshot with edge cases."""
    # Create minimal session
    session = AgentSession(
        id="edge-session",
        name="Edge Case",
        agent_id="test-agent",
        model="test-model",
        total_steps=1,
        total_decisions=0,
        steps=[]
    )
    
    # Single step session
    step = SessionStep(
        id="single-step",
        session_id=session.id,
        step_number=1,
        step_type="think",
        content="Only step",
        timestamp=datetime.now(timezone.utc)
    )
    session.steps = [step]
    
    snapshot = await timeline_engine.get_state_snapshot(session, step_number=1)
    
    assert snapshot.step_number == 1
    assert snapshot.context_window is not None
    assert len(snapshot.conversation_history) == 1


@pytest.mark.asyncio
async def test_state_snapshot_beyond_last_step(timeline_engine: TimelineEngine, sample_session: AgentSession):
    """Test state snapshot for step beyond last recorded step."""
    snapshot = await timeline_engine.get_state_snapshot(
        sample_session,
        step_number=20  # Beyond actual steps
    )
    
    # Should return snapshot at last available step
    assert snapshot.step_number == len(sample_session.steps)


@pytest.mark.asyncio
async def test_identify_patterns(timeline_engine: TimelineEngine, sample_session: AgentSession):
    """Test identifying patterns in session steps."""
    patterns = timeline_engine.identify_patterns(sample_session)
    
    # Should return a dictionary with various pattern analyses
    assert isinstance(patterns, dict)
    assert "loops" in patterns
    assert "redundant_tool_calls" in patterns
    assert "decision_reversals" in patterns
    assert "error_patterns" in patterns
    assert "efficiency_metrics" in patterns
    
    # Check efficiency metrics structure
    metrics = patterns["efficiency_metrics"]
    assert "total_steps" in metrics
    assert "step_type_distribution" in metrics
    assert "tool_success_rate" in metrics


@pytest.mark.asyncio
async def test_identify_loops(timeline_engine: TimelineEngine):
    """Test identifying loops in session steps."""
    session = AgentSession(
        id="loop-session",
        name="Loop Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=6,
        total_decisions=0,
        steps=[]
    )
    
    # Create steps with repeated tool calls
    steps = [
        SessionStep(id="s1", session_id=session.id, step_number=1, step_type="tool_call", content="Search for X", metadata={"tool_name": "search"}),
        SessionStep(id="s2", session_id=session.id, step_number=2, step_type="tool_result", content="Results for X"),
        SessionStep(id="s3", session_id=session.id, step_number=3, step_type="tool_call", content="Search for X", metadata={"tool_name": "search"}),  # Repeated
        SessionStep(id="s4", session_id=session.id, step_number=4, step_type="tool_result", content="Results for X"),
        SessionStep(id="s5", session_id=session.id, step_number=5, step_type="think", content="Thinking"),
        SessionStep(id="s6", session_id=session.id, step_number=6, step_type="decision", content="Deciding"),
    ]
    
    session.steps = steps
    
    patterns = timeline_engine.identify_patterns(session)
    
    # Should identify the loop
    assert len(patterns["loops"]) >= 1
    loop = patterns["loops"][0]
    assert loop["pattern"] == "repeated_tool_call"
    assert "search" in loop.get("content", "").lower()


@pytest.mark.asyncio
async def test_identify_error_patterns(timeline_engine: TimelineEngine):
    """Test identifying error patterns."""
    session = AgentSession(
        id="error-session",
        name="Error Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=8,
        total_decisions=1,
        steps=[]
    )
    
    # Create steps with error and recovery
    steps = [
        SessionStep(id="s1", session_id=session.id, step_number=1, step_type="think", content="Try something"),
        SessionStep(id="s2", session_id=session.id, step_number=2, step_type="tool_call", content="Call risky tool"),
        SessionStep(id="s3", session_id=session.id, step_number=3, step_type="error", content="Tool failed: timeout"),
        SessionStep(id="s4", session_id=session.id, step_number=4, step_type="think", content="Recover from error"),
        SessionStep(id="s5", session_id=session.id, step_number=5, step_type="tool_call", content="Call alternative"),
        SessionStep(id="s6", session_id=session.id, step_number=6, step_type="tool_result", content="Success"),
        SessionStep(id="s7", session_id=session.id, step_number=7, step_type="observation", content="Works now"),
        SessionStep(id="s8", session_id=session.id, step_number=8, step_type="decision", content="Continue"),
    ]
    
    session.steps = steps
    
    patterns = timeline_engine.identify_patterns(session)
    
    # Should identify error patterns
    assert "total_errors" in patterns["error_patterns"]
    assert patterns["error_patterns"]["total_errors"] == 1
    assert "recovery_rate" in patterns["error_patterns"]


@pytest.mark.asyncio
async def test_efficiency_metrics(timeline_engine: TimelineEngine):
    """Test calculating efficiency metrics."""
    session = AgentSession(
        id="efficiency-session",
        name="Efficiency Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=10,
        total_decisions=3,
        steps=[]
    )
    
    # Create steps with metadata for metrics
    steps = []
    for i in range(1, 11):
        step_type = "think" if i % 3 == 1 else "tool_call" if i % 3 == 2 else "decision"
        
        step = SessionStep(
            id=f"step-{i}",
            session_id=session.id,
            step_number=i,
            step_type=step_type,
            content=f"Step {i}",
            metadata={
                "tokens_used": i * 10,
                "latency_ms": i * 100
            } if i != 5 else {"error": "failed"}  # Step 5 is an error
        )
        steps.append(step)
    
    session.steps = steps
    
    patterns = timeline_engine.identify_patterns(session)
    metrics = patterns["efficiency_metrics"]
    
    # Check calculated metrics
    assert metrics["total_steps"] == 10
    assert "think" in metrics["step_type_distribution"]
    assert "decision" in metrics["step_type_distribution"]
    assert "tool_call" in metrics["step_type_distribution"]
    assert metrics["avg_tokens_per_step"] > 0
    assert metrics["decision_density"] > 0
    assert 0 <= metrics["tool_success_rate"] <= 1
    assert 0 <= metrics["step_efficiency"] <= 1


@pytest.mark.asyncio
async def test_context_window_building(timeline_engine: TimelineEngine):
    """Test building context window from steps."""
    session = AgentSession(
        id="context-session",
        name="Context Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=3,
        total_decisions=1,
        steps=[]
    )
    
    # Create steps with context snapshots
    steps = [
        SessionStep(
            id="s1",
            session_id=session.id,
            step_number=1,
            step_type="think",
            content="Initial thought with context",
            context_snapshot="System: You are helpful\nUser: Help me",
            timestamp=datetime.now(timezone.utc)
        ),
        SessionStep(
            id="s2",
            session_id=session.id,
            step_number=2,
            step_type="tool_call",
            content="Calling API",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=1)
        ),
        SessionStep(
            id="s3",
            session_id=session.id,
            step_number=3,
            step_type="decision",
            content="Final decision",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=2)
        ),
    ]
    
    session.steps = steps
    
    snapshot = await timeline_engine.get_state_snapshot(session, step_number=3)
    
    # Context window should include system context and recent steps
    assert snapshot.context_window is not None
    assert "System:" in snapshot.context_window
    assert "Initial thought" in snapshot.context_window
    assert "Final decision" in snapshot.context_window


@pytest.mark.asyncio
async def test_conversation_history_extraction(timeline_engine: TimelineEngine):
    """Test extracting conversation history from steps."""
    session = AgentSession(
        id="conversation-session",
        name="Conversation Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=5,
        total_decisions=2,
        steps=[]
    )
    
    # Create steps that mimic a conversation
    steps = [
        SessionStep(id="s1", session_id=session.id, step_number=1, step_type="think", content="I need to solve this problem"),
        SessionStep(id="s2", session_id=session.id, step_number=2, step_type="observation", content="I see the issue is in the code"),
        SessionStep(id="s3", session_id=session.id, step_number=3, step_type="decision", content="I'll fix the bug first"),
        SessionStep(id="s4", session_id=session.id, step_number=4, step_type="think", content="Now test the fix"),
        SessionStep(id="s5", session_id=session.id, step_number=5, step_type="decision", content="Fix is working, deploy"),
    ]
    
    session.steps = steps
    
    snapshot = await timeline_engine.get_state_snapshot(session, step_number=5)
    
    # Should extract conversation-like entries
    assert len(snapshot.conversation_history) > 0
    for entry in snapshot.conversation_history:
        assert any(marker in entry for marker in ["THINK:", "OBSERVATION:", "DECISION:"])


@pytest.mark.asyncio
async def test_tool_outputs_extraction(timeline_engine: TimelineEngine):
    """Test extracting tool outputs from steps."""
    session = AgentSession(
        id="tools-session",
        name="Tools Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=4,
        total_decisions=1,
        steps=[]
    )
    
    # Create steps with tool calls and results
    steps = [
        SessionStep(
            id="s1",
            session_id=session.id,
            step_number=1,
            step_type="tool_call",
            content="search database",
            metadata={"tool_name": "search"}
        ),
        SessionStep(
            id="s2",
            session_id=session.id,
            step_number=2,
            step_type="tool_result",
            content="Found 42 records",
            metadata={"tool_name": "search"}
        ),
        SessionStep(
            id="s3",
            session_id=session.id,
            step_number=3,
            step_type="tool_call",
            content="calculate stats",
            metadata={"tool_name": "calculator"}
        ),
        SessionStep(
            id="s4",
            session_id=session.id,
            step_number=4,
            step_type="tool_result",
            content="Average: 7.5",
            metadata={"tool_name": "calculator"}
        ),
    ]
    
    session.steps = steps
    
    snapshot = await timeline_engine.get_state_snapshot(session, step_number=4)
    
    # Should extract tool outputs
    assert len(snapshot.tool_outputs_received) == 2
    assert any("search" in str(output.get("tool_name", "")).lower() 
               for output in snapshot.tool_outputs_received)
    assert any("calculator" in str(output.get("tool_name", "")).lower() 
               for output in snapshot.tool_outputs_received)


@pytest.mark.asyncio
async def test_active_goals_inference(timeline_engine: TimelineEngine):
    """Test inferring active goals from step content."""
    session = AgentSession(
        id="goals-session",
        name="Goals Test",
        agent_id="test-agent",
        model="test-model",
        total_steps=3,
        total_decisions=1,
        steps=[]
    )
    
    # Create steps with goal-oriented content
    steps = [
        SessionStep(
            id="s1",
            session_id=session.id,
            step_number=1,
            step_type="think",
            content="My goal is to fix the authentication bug"
        ),
        SessionStep(
            id="s2",
            session_id=session.id,
            step_number=2,
            step_type="decision",
            content="I need to implement OAuth2 for better security"
        ),
        SessionStep(
            id="s3",
            session_id=session.id,
            step_number=3,
            step_type="think",
            content="Objective: complete the fix by EOD"
        ),
    ]
    
    session.steps = steps
    
    snapshot = await timeline_engine.get_state_snapshot(session, step_number=3)
    
    # Should infer goals from content
    assert len(snapshot.active_goals) > 0
    goals_text = " ".join(snapshot.active_goals).lower()
    assert any(word in goals_text for word in ["goal", "objective", "fix", "authentication"])


@pytest.mark.asyncio
async def test_empty_session_patterns(timeline_engine: TimelineEngine):
    """Test pattern identification for empty session."""
    session = AgentSession(
        id="empty-session",
        name="Empty",
        agent_id="test-agent",
        model="test-model",
        total_steps=0,
        total_decisions=0,
        steps=[]
    )
    
    patterns = timeline_engine.identify_patterns(session)
    
    # Should return empty or default patterns
    assert patterns == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])