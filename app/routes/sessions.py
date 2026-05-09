"""Agent session routes — CRUD + steps + timeline + snapshot."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import uuid

from app.database import get_db
from app.models import AgentSession, SessionStep
from app.schemas import SessionCreate, SessionUpdate, SessionStepCreate, SessionStepBatchCreate
from app.engine.recorder import SessionRecorder
from app.engine.timeline import TimelineEngine

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


def _step_to_dict(step):
    return {
        "id": step.id,
        "session_id": step.session_id,
        "step_number": step.step_number,
        "step_type": step.step_type,
        "agent_id": step.agent_id,
        "model": step.model,
        "content": step.content,
        "metadata": step.meta_data or {},
        "context_snapshot": step.context_snapshot,
        "timestamp": step.timestamp.isoformat() if step.timestamp else None,
    }


def _session_to_dict(session, steps=None):
    return {
        "id": session.id,
        "name": session.name,
        "agent_id": session.agent_id,
        "model": session.model,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        "total_steps": session.total_steps,
        "total_decisions": session.total_decisions,
        "metadata": session.meta_data or {},
        "steps": [_step_to_dict(s) for s in steps] if steps else [],
    }


# ── Sessions CRUD ──────────────────────────────────────────

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_session(session_data: SessionCreate, db: AsyncSession = Depends(get_db)):
    recorder = SessionRecorder(db)
    try:
        session = await recorder.create_session(session_data)
        await db.commit()
        return _session_to_dict(session)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@router.get("/")
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    q = select(AgentSession).order_by(desc(AgentSession.created_at)).offset(skip).limit(limit)
    r = await db.execute(q)
    sessions = r.scalars().all()

    total_q = select(AgentSession)
    total_r = await db.execute(total_q)
    total = len(total_r.scalars().all())

    return {
        "items": [_session_to_dict(s) for s in sessions],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    sq = select(SessionStep).where(SessionStep.session_id == session_id).order_by(SessionStep.step_number)
    sr = await db.execute(sq)
    steps = sr.scalars().all()

    return _session_to_dict(session, steps)


@router.put("/{session_id}")
async def update_session(session_id: str, update: SessionUpdate, db: AsyncSession = Depends(get_db)):
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if update.name is not None:
        session.name = update.name
    if update.completed_at is not None:
        session.completed_at = update.completed_at
    if update.metadata is not None:
        session.meta_data = update.metadata

    await db.flush()
    await db.commit()
    return _session_to_dict(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    await db.delete(session)
    await db.commit()
    return None


# ── Steps ──────────────────────────────────────────────────

@router.post("/{session_id}/steps", status_code=status.HTTP_201_CREATED)
async def add_step(session_id: str, step_data: SessionStepCreate, db: AsyncSession = Depends(get_db)):
    recorder = SessionRecorder(db)
    try:
        step = await recorder.record_step(session_id, step_data)
        await db.commit()
        return _step_to_dict(step)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@router.post("/{session_id}/steps/batch", status_code=status.HTTP_201_CREATED)
async def add_batch_steps(session_id: str, batch: SessionStepBatchCreate, db: AsyncSession = Depends(get_db)):
    recorder = SessionRecorder(db)
    try:
        steps = await recorder.record_batch_steps(session_id, batch.steps)
        await db.commit()
        return {"items": [_step_to_dict(s) for s in steps], "count": len(steps)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@router.get("/{session_id}/steps")
async def list_steps(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    sq = select(SessionStep).where(SessionStep.session_id == session_id).order_by(SessionStep.step_number)
    sr = await db.execute(sq)
    steps = sr.scalars().all()
    return {"items": [_step_to_dict(s) for s in steps], "total": len(steps)}


@router.post("/{session_id}/complete")
async def complete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    from datetime import datetime, timezone
    session.completed_at = datetime.now(timezone.utc)
    await db.flush()
    await db.commit()
    return _session_to_dict(session)


# ── Timeline & Snapshot ────────────────────────────────────

@router.get("/{session_id}/timeline")
async def get_timeline(session_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy.orm import selectinload
    q = select(AgentSession).where(AgentSession.id == session_id).options(selectinload(AgentSession.steps))
    r = await db.execute(q)
    session = r.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    engine = TimelineEngine()
    result = await engine.build_timeline(session)

    return {
        "session_id": session_id,
        "total_steps": result.total_steps,
        "decision_cycles": [
            {
                "cycle_number": c.cycle_number,
                "start_step": c.start_step,
                "end_step": c.end_step,
                "steps": [vars(s) if hasattr(s, '__dict__') else s for s in c.steps],
                "latency_ms": c.latency_ms,
                "decision_type": c.decision_type,
            }
            for c in result.decision_cycles
        ],
        "step_density": result.step_density,
    }


@router.get("/{session_id}/snapshot")
async def get_snapshot(session_id: str, step: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    session = await db.get(AgentSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    sq = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number <= step,
    ).order_by(SessionStep.step_number)
    sr = await db.execute(sq)
    steps = list(sr.scalars().all())

    if not steps:
        raise HTTPException(status_code=404, detail=f"No steps up to {step}")

    from app.engine.snapshot import SnapshotEngine
    engine = SnapshotEngine()
    result = await engine.capture_state(steps, step)

    return {
        "step_number": step,
        "context_window": result.context_window,
        "conversation_history": result.conversation_history,
        "tool_outputs_received": result.tool_outputs_received,
        "active_goals": result.active_goals,
        "memory_state": result.memory_state,
        "constraints": result.constraints,
    }
