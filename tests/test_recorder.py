"""
Test the session recorder functionality.
"""
import pytest
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AgentSession, SessionStep
from app.schemas import SessionCreate, SessionStepCreate, Metadata
from app.engine.recorder import SessionRecorder


@pytest.fixture
async def recorder(db_session: AsyncSession):
    """Fixture for session recorder."""
    return SessionRecorder(db_session)


@pytest.mark.asyncio
async def test_create_session(recorder: SessionRecorder):
    """Test creating a new session."""
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    
    session = await recorder.create_session(session_data)
    
    assert session.id is not None
    assert session.name == "Test Session"
    assert session.agent_id == "test-agent"
    assert session.model == "test-model"
    assert session.total_steps == 0
    assert session.total_decisions == 0
    assert session.created_at is not None


@pytest.mark.asyncio
async def test_record_single_step(recorder: SessionRecorder):
    """Test recording a single step."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Record a step
    step_data = SessionStepCreate(
        step_number=1,
        step_type="think",
        agent_id="test-agent",
        model="test-model",
        content="Thinking about the problem",
        metadata=Metadata(tokens_used=45, latency_ms=320)
    )
    
    step = await recorder.record_step(session.id, step_data)
    
    assert step.id is not None
    assert step.session_id == session.id
    assert step.step_number == 1
    assert step.step_type == "think"
    assert step.content == "Thinking about the problem"
    assert step.meta_data["tokens_used"] == 45
    assert step.meta_data["latency_ms"] == 320
    
    # Verify session stats were updated
    session = await recorder.get_session(session.id)
    assert session.total_steps == 1
    assert session.total_decisions == 0  # think step is not a decision


@pytest.mark.asyncio
async def test_record_decision_step(recorder: SessionRecorder):
    """Test recording a decision step updates decision count."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Record a decision step
    step_data = SessionStepCreate(
        step_number=1,
        step_type="decision",
        agent_id="test-agent",
        model="test-model",
        content="Deciding to proceed",
        metadata=Metadata(tokens_used=50, latency_ms=400)
    )
    
    step = await recorder.record_step(session.id, step_data)
    
    # Verify session stats
    session = await recorder.get_session(session.id)
    assert session.total_steps == 1
    assert session.total_decisions == 1


@pytest.mark.asyncio
async def test_record_batch_steps(recorder: SessionRecorder):
    """Test recording multiple steps in a batch."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Record batch of steps
    steps_data = [
        SessionStepCreate(
            step_number=1,
            step_type="think",
            content="Thinking step 1"
        ),
        SessionStepCreate(
            step_number=2,
            step_type="tool_call",
            content="Calling tool"
        ),
        SessionStepCreate(
            step_number=3,
            step_type="decision",
            content="Making decision"
        )
    ]
    
    steps = await recorder.record_batch_steps(session.id, steps_data)
    
    assert len(steps) == 3
    assert steps[0].step_number == 1
    assert steps[1].step_number == 2
    assert steps[2].step_number == 3
    
    # Verify session stats
    session = await recorder.get_session(session.id)
    assert session.total_steps == 3
    assert session.total_decisions == 1


@pytest.mark.asyncio
async def test_invalid_step_sequence(recorder: SessionRecorder):
    """Test recording steps out of sequence."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Try to record step 2 without step 1
    step_data = SessionStepCreate(
        step_number=2,
        step_type="think",
        content="Skipping step 1"
    )
    
    with pytest.raises(ValueError, match="Expected step number 1"):
        await recorder.record_step(session.id, step_data)


@pytest.mark.asyncio
async def test_complete_session(recorder: SessionRecorder):
    """Test marking a session as completed."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Complete the session
    completed_session = await recorder.complete_session(session.id)
    
    assert completed_session.completed_at is not None
    assert completed_session.completed_at <= datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_get_session_steps(recorder: SessionRecorder):
    """Test retrieving steps for a session."""
    # Create session and add steps
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Add some steps
    steps_data = [
        SessionStepCreate(step_number=1, step_type="think", content="Step 1"),
        SessionStepCreate(step_number=2, step_type="tool_call", content="Step 2"),
        SessionStepCreate(step_number=3, step_type="decision", content="Step 3")
    ]
    
    await recorder.record_batch_steps(session.id, steps_data)
    
    # Get steps
    steps = await recorder.get_session_steps(session.id)
    
    assert len(steps) == 3
    assert steps[0].step_number == 1
    assert steps[1].step_number == 2
    assert steps[2].step_number == 3


@pytest.mark.asyncio
async def test_get_session_steps_pagination(recorder: SessionRecorder):
    """Test retrieving steps with pagination."""
    # Create session and add steps
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Add multiple steps
    steps_data = []
    for i in range(1, 11):
        steps_data.append(
            SessionStepCreate(
                step_number=i,
                step_type="think",
                content=f"Step {i}"
            )
        )
    
    await recorder.record_batch_steps(session.id, steps_data)
    
    # Get first 5 steps
    steps = await recorder.get_session_steps(session.id, limit=5)
    assert len(steps) == 5
    assert steps[0].step_number == 1
    assert steps[4].step_number == 5
    
    # Get next 5 steps with offset
    steps = await recorder.get_session_steps(session.id, limit=5, offset=5)
    assert len(steps) == 5
    assert steps[0].step_number == 6
    assert steps[4].step_number == 10


@pytest.mark.asyncio
async def test_auto_detect_step_type(recorder: SessionRecorder):
    """Test auto-detection of step type from content."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Record step without specifying type
    step_data = SessionStepCreate(
        step_number=1,
        step_type="",  # Empty type to trigger auto-detection
        content="I think we should proceed with the plan"
    )
    
    step = await recorder.record_step(session.id, step_data)
    
    # Should auto-detect as "think" based on content
    assert step.step_type == "think"


@pytest.mark.asyncio
async def test_error_step_metadata(recorder: SessionRecorder):
    """Test that error steps update session metadata."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Record an error step
    step_data = SessionStepCreate(
        step_number=1,
        step_type="error",
        content="Failed to execute tool",
        metadata=Metadata(error_message="Tool timeout after 30 seconds")
    )
    
    step = await recorder.record_step(session.id, step_data)
    
    # Get updated session
    session = await recorder.get_session(session.id)
    
    # Check that error was recorded in metadata
    assert "errors" in session.meta_data
    assert len(session.meta_data["errors"]) == 1
    assert session.meta_data["errors"][0]["step"] == 1
    assert "timeout" in session.meta_data["errors"][0]["message"]


@pytest.mark.asyncio
async def test_context_snapshot_truncation(recorder: SessionRecorder):
    """Test that long context snapshots are truncated."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Create a very long context snapshot
    long_context = "x" * 20000  # 20k characters
    
    step_data = SessionStepCreate(
        step_number=1,
        step_type="think",
        content="Thinking",
        context_snapshot=long_context
    )
    
    step = await recorder.record_step(session.id, step_data)
    
    # Context should be truncated to max size (default 10000)
    assert len(step.context_snapshot) <= 10000


@pytest.mark.asyncio
async def test_session_not_found_error(recorder: SessionRecorder):
    """Test error when session doesn't exist."""
    non_existent_id = "non-existent-session-id"
    
    step_data = SessionStepCreate(
        step_number=1,
        step_type="think",
        content="This should fail"
    )
    
    with pytest.raises(ValueError, match="not found"):
        await recorder.record_step(non_existent_id, step_data)


@pytest.mark.asyncio
async def test_batch_step_validation(recorder: SessionRecorder):
    """Test validation in batch step recording."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Try to record batch with wrong step numbers
    steps_data = [
        SessionStepCreate(step_number=1, step_type="think", content="Step 1"),
        SessionStepCreate(step_number=3, step_type="think", content="Step 3")  # Missing step 2
    ]
    
    with pytest.raises(ValueError, match="Expected step number 2"):
        await recorder.record_batch_steps(session.id, steps_data)


@pytest.mark.asyncio
async def test_step_type_validation(recorder: SessionRecorder):
    """Test that invalid step types are rejected."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model"
    )
    session = await recorder.create_session(session_data)
    
    # Try to record step with invalid type
    step_data = SessionStepCreate(
        step_number=1,
        step_type="invalid_type",
        content="Should fail"
    )
    
    # Should be caught by Pydantic validation
    with pytest.raises(ValueError):
        await recorder.record_step(session.id, step_data)


@pytest.mark.asyncio
async def test_metadata_serialization(recorder: SessionRecorder):
    """Test that metadata is properly serialized and deserialized."""
    # Create session first
    session_data = SessionCreate(
        name="Test Session",
        agent_id="test-agent",
        model="test-model",
        metadata={"custom_field": "custom_value"}
    )
    session = await recorder.create_session(session_data)
    
    # Record step with complex metadata
    step_data = SessionStepCreate(
        step_number=1,
        step_type="think",
        content="Test",
        metadata=Metadata(
            tokens_used=100,
            latency_ms=500,
            tool_name="test_tool",
            tool_args={"param1": "value1", "param2": 42},
            cost_usd=0.0025
        )
    )
    
    step = await recorder.record_step(session.id, step_data)
    
    # Check metadata fields
    assert step.meta_data["tokens_used"] == 100
    assert step.meta_data["latency_ms"] == 500
    assert step.meta_data["tool_name"] == "test_tool"
    assert step.meta_data["tool_args"]["param1"] == "value1"
    assert step.meta_data["tool_args"]["param2"] == 42
    assert step.meta_data["cost_usd"] == 0.0025
    
    # Check session metadata
    assert session.meta_data["custom_field"] == "custom_value"