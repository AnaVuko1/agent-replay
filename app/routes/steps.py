from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.database import get_db
from app.models import SessionStep, AgentSession
from app.schemas import SessionStepResponse, StateSnapshot
from app.engine.timeline import TimelineEngine
from app.engine.snapshot import SnapshotEngine

router = APIRouter(prefix="/api/v1/steps", tags=["steps"])


@router.get("/{step_id}")
async def get_step(
    step_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific step by ID."""
    step = await db.get(SessionStep, step_id)
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Step {step_id} not found"
        )
    
    return SessionStepResponse(
        id=step.id,
        session_id=step.session_id,
        step_number=step.step_number,
        step_type=step.step_type,
        agent_id=step.agent_id,
        model=step.model,
        content=step.content,
        metadata=step.meta_data or {},
        context_snapshot=step.context_snapshot,
        timestamp=step.timestamp
    )


@router.get("/session/{session_id}/step/{step_number}")
async def get_step_by_number(
    session_id: str,
    step_number: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific step by session ID and step number."""
    # Check if session exists
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Get step
    query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number == step_number
    )
    
    result = await db.execute(query)
    step = result.scalar_one_or_none()
    
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Step {step_number} not found in session {session_id}"
        )
    
    return SessionStepResponse(
        id=step.id,
        session_id=step.session_id,
        step_number=step.step_number,
        step_type=step.step_type,
        agent_id=step.agent_id,
        model=step.model,
        content=step.content,
        metadata=step.meta_data or {},
        context_snapshot=step.context_snapshot,
        timestamp=step.timestamp
    )


@router.get("/session/{session_id}/around/{step_number}")
async def get_steps_around(
    session_id: str,
    step_number: int,
    window: int = Query(5, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Get steps around a specific step number (context window)."""
    # Check if session exists
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Calculate range
    start_step = max(1, step_number - window)
    end_step = step_number + window
    
    # Get steps in range
    query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number >= start_step,
        SessionStep.step_number <= end_step
    ).order_by(SessionStep.step_number)
    
    result = await db.execute(query)
    steps = result.scalars().all()
    
    # Find target step
    target_step = None
    step_responses = []
    
    for step in steps:
        step_response = SessionStepResponse(
            id=step.id,
            session_id=step.session_id,
            step_number=step.step_number,
            step_type=step.step_type,
            agent_id=step.agent_id,
            model=step.model,
            content=step.content,
            metadata=step.meta_data or {},
            context_snapshot=step.context_snapshot,
            timestamp=step.timestamp
        )
        
        step_responses.append(step_response)
        
        if step.step_number == step_number:
            target_step = step_response
    
    return {
        "target_step": target_step,
        "context_window": window,
        "steps_before": [s for s in step_responses if s.step_number < step_number],
        "steps_after": [s for s in step_responses if s.step_number > step_number],
        "all_steps": step_responses
    }


@router.get("/session/{session_id}/timeline")
async def get_session_timeline(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get timeline with decision cycles for a session."""
    # Get session with steps
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Get all steps
    steps_query = select(SessionStep).where(
        SessionStep.session_id == session_id
    ).order_by(SessionStep.step_number)
    
    steps_result = await db.execute(steps_query)
    session.steps = list(steps_result.scalars().all())
    
    # Build timeline
    timeline_engine = TimelineEngine()
    timeline = await timeline_engine.build_timeline(session)
    
    # Identify patterns
    patterns = timeline_engine.identify_patterns(session)
    
    return {
        "timeline": timeline,
        "patterns": patterns,
        "session_info": {
            "id": session.id,
            "name": session.name,
            "agent_id": session.agent_id,
            "model": session.model,
            "total_steps": session.total_steps,
            "total_decisions": session.total_decisions,
            "status": "completed" if session.completed_at else "in_progress"
        }
    }


@router.get("/session/{session_id}/snapshot")
async def get_state_snapshot(
    session_id: str,
    step: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db)
):
    """Get agent state snapshot at a specific step."""
    # Get session with steps
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Get all steps up to target step
    steps_query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number <= step
    ).order_by(SessionStep.step_number)
    
    steps_result = await db.execute(steps_query)
    steps = list(steps_result.scalars().all())
    
    if not steps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No steps found at or before step {step}"
        )
    
    # Create snapshot
    snapshot_engine = SnapshotEngine()
    snapshot = await snapshot_engine.capture_state(steps, step)
    
    # Get step info for reference
    target_step_query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number == step
    )
    
    target_step_result = await db.execute(target_step_query)
    target_step = target_step_result.scalar_one_or_none()
    
    step_info = None
    if target_step:
        step_info = SessionStepResponse(
            id=target_step.id,
            session_id=target_step.session_id,
            step_number=target_step.step_number,
            step_type=target_step.step_type,
            agent_id=target_step.agent_id,
            model=target_step.model,
            content=target_step.content,
            metadata=target_step.meta_data or {},
            context_snapshot=target_step.context_snapshot,
            timestamp=target_step.timestamp
        )
    
    return {
        "snapshot": snapshot,
        "step_info": step_info,
        "session_info": {
            "id": session.id,
            "name": session.name,
            "agent_id": session.agent_id,
            "model": session.model
        }
    }


@router.get("/session/{session_id}/compare_steps")
async def compare_steps(
    session_id: str,
    step1: int = Query(..., ge=1),
    step2: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db)
):
    """Compare two steps from the same session."""
    # Get session
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Get both steps
    query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number.in_([step1, step2])
    ).order_by(SessionStep.step_number)
    
    result = await db.execute(query)
    steps = {step.step_number: step for step in result.scalars().all()}
    
    if step1 not in steps or step2 not in steps:
        missing = []
        if step1 not in steps:
            missing.append(f"step {step1}")
        if step2 not in steps:
            missing.append(f"step {step2}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Steps not found: {', '.join(missing)}"
        )
    
    step1_obj = steps[step1]
    step2_obj = steps[step2]
    
    # Convert to response models
    step1_response = SessionStepResponse(
        id=step1_obj.id,
        session_id=step1_obj.session_id,
        step_number=step1_obj.step_number,
        step_type=step1_obj.step_type,
        agent_id=step1_obj.agent_id,
        model=step1_obj.model,
        content=step1_obj.content,
        metadata=step1_obj.meta_data or {},
        context_snapshot=step1_obj.context_snapshot,
        timestamp=step1_obj.timestamp
    )
    
    step2_response = SessionStepResponse(
        id=step2_obj.id,
        session_id=step2_obj.session_id,
        step_number=step2_obj.step_number,
        step_type=step2_obj.step_type,
        agent_id=step2_obj.agent_id,
        model=step2_obj.model,
        content=step2_obj.content,
        metadata=step2_obj.meta_data or {},
        context_snapshot=step2_obj.context_snapshot,
        timestamp=step2_obj.timestamp
    )
    
    # Calculate differences
    differences = []
    
    # Compare step types
    if step1_obj.step_type != step2_obj.step_type:
        differences.append({
            "field": "step_type",
            "step1": step1_obj.step_type,
            "step2": step2_obj.step_type
        })
    
    # Compare content length
    len1 = len(step1_obj.content)
    len2 = len(step2_obj.content)
    if len1 != len2:
        differences.append({
            "field": "content_length",
            "step1": len1,
            "step2": len2,
            "difference": len2 - len1
        })
    
    # Compare metadata keys
    metadata1 = set(step1_obj.meta_data.keys() if step1_obj.meta_data else [])
    metadata2 = set(step2_obj.meta_data.keys() if step2_obj.meta_data else [])
    
    if metadata1 != metadata2:
        differences.append({
            "field": "metadata_keys",
            "step1_only": list(metadata1 - metadata2),
            "step2_only": list(metadata2 - metadata1),
            "common": list(metadata1.intersection(metadata2))
        })
    
    # Get context between steps
    if step1 < step2:
        context_query = select(SessionStep).where(
            SessionStep.session_id == session_id,
            SessionStep.step_number > step1,
            SessionStep.step_number < step2
        ).order_by(SessionStep.step_number)
        
        context_result = await db.execute(context_query)
        context_steps = list(context_result.scalars().all())
    else:
        context_steps = []
    
    return {
        "step1": step1_response,
        "step2": step2_response,
        "differences": differences,
        "step_gap": abs(step1 - step2),
        "context_steps": [
            SessionStepResponse(
                id=step.id,
                session_id=step.session_id,
                step_number=step.step_number,
                step_type=step.step_type,
                agent_id=step.agent_id,
                model=step.model,
                content=step.content,
                metadata=step.meta_data or {},
                context_snapshot=step.context_snapshot,
                timestamp=step.timestamp
            )
            for step in context_steps[:10]  # Limit to 10 context steps
        ]
    }


@router.get("/session/{session_id}/search")
async def search_steps(
    session_id: str,
    query: str = Query(..., min_length=1),
    step_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Search steps within a session."""
    # Check if session exists
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Build search query
    sql_query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.content.ilike(f"%{query}%")
    ).order_by(SessionStep.step_number)
    
    if step_type:
        sql_query = sql_query.where(SessionStep.step_type == step_type)
    
    sql_query = sql_query.limit(limit)
    
    result = await db.execute(sql_query)
    steps = result.scalars().all()
    
    # Convert to response models
    step_responses = []
    for step in steps:
        # Highlight search term in content
        content = step.content
        highlighted_content = content.replace(
            query, 
            f"<mark>{query}</mark>"
        )
        
        step_responses.append({
            "step": SessionStepResponse(
                id=step.id,
                session_id=step.session_id,
                step_number=step.step_number,
                step_type=step.step_type,
                agent_id=step.agent_id,
                model=step.model,
                content=step.content,
                metadata=step.meta_data or {},
                context_snapshot=step.context_snapshot,
                timestamp=step.timestamp
            ),
            "highlighted_content": highlighted_content,
            "match_position": step.content.lower().find(query.lower())
        })
    
    return {
        "query": query,
        "step_type_filter": step_type,
        "total_matches": len(step_responses),
        "matches": step_responses,
        "session_info": {
            "id": session.id,
            "name": session.name,
            "agent_id": session.agent_id,
            "model": session.model
        }
    }