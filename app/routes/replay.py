from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from app.database import get_db
from app.models import Replay, ReplayStep, AgentSession
from app.schemas import ReplayCreate, ReplayResponse, ReplayComparisonResponse
from app.engine.replay_engine import ReplayEngine
from app.engine.comparator import Comparator

router = APIRouter(prefix="/api/v1/replays", tags=["replays"])


@router.post("/session/{session_id}/replay", status_code=status.HTTP_201_CREATED)
async def create_replay(
    session_id: str,
    replay_data: ReplayCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create and execute a replay of a session segment."""
    replay_engine = ReplayEngine(db)
    
    try:
        # Create replay
        replay = await replay_engine.create_replay(session_id, replay_data)
        
        # Execute replay in background
        background_tasks.add_task(execute_replay_background, replay.id, db)
        
        await db.commit()
        
        return ReplayResponse(
            id=replay.id,
            session_id=replay.session_id,
            name=replay.name,
            start_step=replay.start_step,
            end_step=replay.end_step,
            replay_config=replay.replay_config,
            created_at=replay.created_at,
            completed_at=replay.completed_at,
            total_steps=replay.total_steps,
            status=replay.status
        )
    except ValueError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create replay: {str(e)}"
        )


async def execute_replay_background(replay_id: str, db: AsyncSession):
    """Background task to execute a replay."""
    async with db.begin():
        replay_engine = ReplayEngine(db)
        try:
            await replay_engine.execute_replay(replay_id)
        except Exception as e:
            # Update replay status to failed
            replay = await db.get(Replay, replay_id)
            if replay:
                replay.status = "failed"
                replay.meta_data = {"error": str(e)}
            await db.commit()


@router.get("/session/{session_id}/replays")
async def list_session_replays(
    session_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all replays for a session."""
    # Check if session exists
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Build query
    query = select(Replay).where(
        Replay.session_id == session_id
    ).order_by(Replay.created_at.desc())
    
    if status_filter:
        query = query.where(Replay.status == status_filter)
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    replays = result.scalars().all()
    
    return [
        ReplayResponse(
            id=replay.id,
            session_id=replay.session_id,
            name=replay.name,
            start_step=replay.start_step,
            end_step=replay.end_step,
            replay_config=replay.replay_config,
            created_at=replay.created_at,
            completed_at=replay.completed_at,
            total_steps=replay.total_steps,
            status=replay.status,
            metrics=replay.meta_data.get("metrics") if replay.meta_data else None
        )
        for replay in replays
    ]


@router.get("/{replay_id}")
async def get_replay(
    replay_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific replay by ID."""
    replay = await db.get(Replay, replay_id)
    if not replay:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay {replay_id} not found"
        )
    
    return ReplayResponse(
        id=replay.id,
        session_id=replay.session_id,
        name=replay.name,
        start_step=replay.start_step,
        end_step=replay.end_step,
        replay_config=replay.replay_config,
        created_at=replay.created_at,
        completed_at=replay.completed_at,
        total_steps=replay.total_steps,
        status=replay.status,
        metrics=replay.meta_data.get("metrics") if replay.meta_data else None
    )


@router.get("/{replay_id}/steps")
async def get_replay_steps(
    replay_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all steps for a replay."""
    replay = await db.get(Replay, replay_id)
    if not replay:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay {replay_id} not found"
        )
    
    # Get replay steps
    steps_query = select(ReplayStep).where(
        ReplayStep.replay_id == replay_id
    ).order_by(ReplayStep.step_number)
    
    steps_result = await db.execute(steps_query)
    steps = steps_result.scalars().all()
    
    return {
        "replay_id": replay_id,
        "total_steps": len(steps),
        "steps": [
            {
                "id": step.id,
                "step_number": step.step_number,
                "step_type": step.step_type,
                "content": step.content,
                "original_content": step.original_content,
                "divergence_score": step.divergence_score,
                "metadata": step.meta_data or {},
                "created_at": step.created_at
            }
            for step in steps
        ]
    }


@router.post("/{replay_id}/execute")
async def execute_replay(
    replay_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Execute a pending replay."""
    replay_engine = ReplayEngine(db)
    
    try:
        replay = await replay_engine.execute_replay(replay_id)
        await db.commit()
        
        return ReplayResponse(
            id=replay.id,
            session_id=replay.session_id,
            name=replay.name,
            start_step=replay.start_step,
            end_step=replay.end_step,
            replay_config=replay.replay_config,
            created_at=replay.created_at,
            completed_at=replay.completed_at,
            total_steps=replay.total_steps,
            status=replay.status,
            metrics=replay.meta_data.get("metrics") if replay.meta_data else None
        )
    except ValueError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute replay: {str(e)}"
        )


@router.get("/{replay_id}/compare")
async def compare_replay(
    replay_id: str,
    baseline_type: str = Query("original", regex="^(original|another_replay)$"),
    baseline_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Compare a replay against its baseline (original or another replay)."""
    replay_engine = ReplayEngine(db)
    
    try:
        comparison = await replay_engine.get_replay_comparison(
            replay_id,
            baseline_type,
            baseline_id
        )
        
        return ReplayComparisonResponse(**comparison)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare replay: {str(e)}"
        )


@router.get("/{replay_id}/analysis")
async def analyze_replay(
    replay_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed analysis of a replay."""
    replay_engine = ReplayEngine(db)
    
    try:
        analysis = await replay_engine.get_replay_analysis(replay_id)
        return analysis
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze replay: {str(e)}"
        )


@router.post("/session/{session_id}/batch_replay")
async def batch_replay(
    session_id: str,
    replay_configs: List[ReplayCreate],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Execute multiple replays in batch."""
    replay_engine = ReplayEngine(db)
    
    try:
        replays = await replay_engine.batch_replay(session_id, replay_configs)
        
        # Execute each replay in background
        for replay in replays:
            if replay.status == "pending":
                background_tasks.add_task(execute_replay_background, replay.id, db)
        
        await db.commit()
        
        return [
            ReplayResponse(
                id=replay.id,
                session_id=replay.session_id,
                name=replay.name,
                start_step=replay.start_step,
                end_step=replay.end_step,
                replay_config=replay.replay_config,
                created_at=replay.created_at,
                completed_at=replay.completed_at,
                total_steps=replay.total_steps,
                status=replay.status,
                metrics=replay.meta_data.get("metrics") if replay.meta_data else None
            )
            for replay in replays
        ]
    except ValueError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute batch replay: {str(e)}"
        )


@router.get("/compare_two_replays")
async def compare_two_replays(
    replay1_id: str,
    replay2_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Compare two replays directly."""
    # Get both replays
    replay1 = await db.get(Replay, replay1_id)
    replay2 = await db.get(Replay, replay2_id)
    
    if not replay1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay {replay1_id} not found"
        )
    
    if not replay2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay {replay2_id} not found"
        )
    
    # Ensure both replays are completed
    if replay1.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Replay {replay1_id} is not completed"
        )
    
    if replay2.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Replay {replay2_id} is not completed"
        )
    
    # Get steps for both replays
    steps1_query = select(ReplayStep).where(
        ReplayStep.replay_id == replay1_id
    ).order_by(ReplayStep.step_number)
    
    steps2_query = select(ReplayStep).where(
        ReplayStep.replay_id == replay2_id
    ).order_by(ReplayStep.step_number)
    
    steps1_result = await db.execute(steps1_query)
    steps2_result = await db.execute(steps2_query)
    
    steps1 = list(steps1_result.scalars().all())
    steps2 = list(steps2_result.scalars().all())
    
    # Compare using comparator
    comparator = Comparator()
    comparison = comparator.compare_replays(replay1, steps1, replay2, steps2)
    
    return comparison


@router.delete("/{replay_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_replay(
    replay_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a replay and all its steps."""
    replay = await db.get(Replay, replay_id)
    if not replay:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay {replay_id} not found"
        )
    
    await db.delete(replay)
    await db.commit()
    
    return None


@router.get("/session/{session_id}/replay_preview")
async def get_replay_preview(
    session_id: str,
    start_step: int = Query(..., ge=1),
    end_step: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db)
):
    """Get preview of what would be replayed without executing."""
    # Check if session exists
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Validate step range
    if end_step < start_step:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_step must be >= start_step"
        )
    
    if end_step > session.total_steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session only has {session.total_steps} steps"
        )
    
    # Get steps in range
    steps_query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number >= start_step,
        SessionStep.step_number <= end_step
    ).order_by(SessionStep.step_number)
    
    steps_result = await db.execute(steps_query)
    steps = list(steps_result.scalars().all())
    
    # Analyze steps
    step_types = {}
    total_tokens = 0
    decision_count = 0
    
    for step in steps:
        # Count step types
        step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
        
        # Count decisions
        if step.step_type == "decision":
            decision_count += 1
        
        # Sum tokens
        if step.meta_data and "tokens_used" in step.meta_data:
            total_tokens += step.meta_data["tokens_used"]
    
    return {
        "session_id": session_id,
        "start_step": start_step,
        "end_step": end_step,
        "total_steps_in_range": len(steps),
        "step_type_distribution": step_types,
        "decision_count": decision_count,
        "estimated_tokens": total_tokens,
        "preview_steps": [
            {
                "step_number": step.step_number,
                "step_type": step.step_type,
                "content_preview": step.content[:100] + ("..." if len(step.content) > 100 else ""),
                "agent_id": step.agent_id,
                "model": step.model
            }
            for step in steps[:5]  # First 5 steps as preview
        ],
        "recommended_replay_config": {
            "model": session.model,
            "temperature": 0.7,
            "system_prompt": f"Replay of steps {start_step}-{end_step} from session '{session.name}'"
        }
    }