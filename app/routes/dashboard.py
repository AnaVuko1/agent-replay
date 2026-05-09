from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from app.database import get_db
from app.models import AgentSession, Replay, SessionStep
from app.schemas import DashboardStats, AgentSessionResponse, ReplayResponse

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(
    time_window_hours: int = 24,
    db: AsyncSession = Depends(get_db)
):
    """Get dashboard statistics and overview."""
    # Calculate time threshold
    time_threshold = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
    
    # Get total sessions
    total_sessions_query = select(func.count(AgentSession.id))
    total_sessions_result = await db.execute(total_sessions_query)
    total_sessions = total_sessions_result.scalar() or 0
    
    # Get recent sessions (last 24 hours)
    recent_sessions_query = select(AgentSession).where(
        AgentSession.created_at >= time_threshold
    ).order_by(desc(AgentSession.created_at)).limit(10)
    
    recent_sessions_result = await db.execute(recent_sessions_query)
    recent_sessions = list(recent_sessions_result.scalars().all())
    
    # Get active replays
    active_replays_query = select(Replay).where(
        Replay.status.in_(["pending", "running"])
    ).order_by(Replay.created_at.desc()).limit(10)
    
    active_replays_result = await db.execute(active_replays_query)
    active_replays = list(active_replays_result.scalars().all())
    
    # Get step statistics
    if total_sessions > 0:
        # Get total steps across all sessions
        total_steps_query = select(func.sum(AgentSession.total_steps))
        total_steps_result = await db.execute(total_steps_query)
        total_steps = total_steps_result.scalar() or 0
        
        # Get total decisions
        total_decisions_query = select(func.sum(AgentSession.total_decisions))
        total_decisions_result = await db.execute(total_decisions_query)
        total_decisions = total_decisions_result.scalar() or 0
        
        # Calculate averages
        avg_steps_per_session = total_steps / total_sessions if total_sessions > 0 else 0
        avg_decisions_per_session = total_decisions / total_sessions if total_sessions > 0 else 0
    else:
        total_steps = 0
        total_decisions = 0
        avg_steps_per_session = 0
        avg_decisions_per_session = 0
    
    # Get step type distribution
    step_type_query = select(
        SessionStep.step_type,
        func.count(SessionStep.id).label("count")
    ).group_by(SessionStep.step_type)
    
    step_type_result = await db.execute(step_type_query)
    step_type_rows = step_type_result.all()
    
    step_type_distribution = {
        row.step_type: row.count
        for row in step_type_rows
    }
    
    # Get most common error patterns (simplified)
    error_patterns = await _get_error_patterns(db, time_threshold)
    
    # Convert recent sessions to response models
    recent_session_responses = []
    for session in recent_sessions:
        # Get steps count (not full steps for dashboard)
        steps_query = select(SessionStep).where(SessionStep.session_id == session.id)
        steps_result = await db.execute(steps_query)
        steps = list(steps_result.scalars().all())
        
        recent_session_responses.append(AgentSessionResponse(
            id=session.id,
            name=session.name,
            agent_id=session.agent_id,
            model=session.model,
            created_at=session.created_at,
            updated_at=session.updated_at,
            completed_at=session.completed_at,
            total_steps=session.total_steps,
            total_decisions=session.total_decisions,
            metadata=session.meta_data or {},
            steps=[
                {
                    "id": step.id,
                    "session_id": step.session_id,
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "agent_id": step.agent_id,
                    "model": step.model,
                    "content": step.content[:100] + ("..." if len(step.content) > 100 else ""),
                    "metadata": step.meta_data or {},
                    "context_snapshot": step.context_snapshot,
                    "timestamp": step.timestamp
                }
                for step in steps[:3]  # Only include first 3 steps for dashboard
            ]
        ))
    
    # Convert active replays to response models
    active_replay_responses = []
    for replay in active_replays:
        active_replay_responses.append(ReplayResponse(
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
        ))
    
    return DashboardStats(
        total_sessions=total_sessions,
        total_steps=total_steps,
        total_decisions=total_decisions,
        avg_steps_per_session=avg_steps_per_session,
        avg_decisions_per_session=avg_decisions_per_session,
        most_common_error_patterns=error_patterns,
        step_type_distribution=step_type_distribution,
        recent_sessions=recent_session_responses,
        active_replays=active_replay_responses
    )


async def _get_error_patterns(db: AsyncSession, time_threshold: datetime) -> List[Dict[str, Any]]:
    """Get most common error patterns from recent sessions."""
    # Get error steps from recent sessions
    error_query = select(SessionStep).where(
        SessionStep.step_type == "error",
        SessionStep.timestamp >= time_threshold
    ).order_by(desc(SessionStep.timestamp)).limit(50)
    
    error_result = await db.execute(error_query)
    error_steps = list(error_result.scalars().all())
    
    if not error_steps:
        return []
    
    # Simple pattern extraction
    patterns = {}
    
    for step in error_steps:
        content = step.content.lower()
        
        # Categorize error patterns
        pattern = "other"
        
        if any(word in content for word in ["timeout", "timed out", "time out"]):
            pattern = "timeout"
        elif any(word in content for word in ["permission", "unauthorized", "forbidden"]):
            pattern = "permission"
        elif any(word in content for word in ["not found", "missing", "does not exist"]):
            pattern = "not_found"
        elif any(word in content for word in ["invalid", "malformed", "bad request"]):
            pattern = "invalid_input"
        elif any(word in content for word in ["network", "connection", "socket"]):
            pattern = "network"
        elif any(word in content for word in ["memory", "out of memory", "oom"]):
            pattern = "memory"
        elif any(word in content for word in ["syntax", "parse", "compilation"]):
            pattern = "syntax"
        elif any(word in content for word in ["type", "attribute", "key error"]):
            pattern = "type_error"
        
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    # Convert to list of dictionaries
    pattern_list = []
    for pattern, count in patterns.items():
        # Get example error
        example = None
        for step in error_steps:
            content_lower = step.content.lower()
            if pattern == "timeout" and any(word in content_lower for word in ["timeout", "timed out"]):
                example = step.content[:100]
                break
            elif pattern == "permission" and any(word in content_lower for word in ["permission", "unauthorized"]):
                example = step.content[:100]
                break
            elif pattern == "other":
                example = step.content[:100]
                break
        
        pattern_list.append({
            "pattern": pattern,
            "count": count,
            "percentage": (count / len(error_steps)) * 100,
            "example": example
        })
    
    # Sort by count descending
    pattern_list.sort(key=lambda x: x["count"], reverse=True)
    
    return pattern_list[:5]  # Return top 5 patterns


@router.get("/activity")
async def get_recent_activity(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get recent activity across sessions and replays."""
    # Get recent sessions
    sessions_query = select(AgentSession).order_by(
        desc(AgentSession.created_at)
    ).limit(limit // 2)
    
    sessions_result = await db.execute(sessions_query)
    sessions = list(sessions_result.scalars().all())
    
    # Get recent replays
    replays_query = select(Replay).order_by(
        desc(Replay.created_at)
    ).limit(limit // 2)
    
    replays_result = await db.execute(replays_query)
    replays = list(replays_result.scalars().all())
    
    # Combine and sort by timestamp
    activities = []
    
    for session in sessions:
        activities.append({
            "type": "session",
            "id": session.id,
            "name": session.name or f"Session {session.id[:8]}",
            "timestamp": session.created_at,
            "agent_id": session.agent_id,
            "model": session.model,
            "status": "completed" if session.completed_at else "in_progress",
            "total_steps": session.total_steps,
            "total_decisions": session.total_decisions
        })
    
    for replay in replays:
        activities.append({
            "type": "replay",
            "id": replay.id,
            "name": replay.name or f"Replay {replay.id[:8]}",
            "timestamp": replay.created_at,
            "session_id": replay.session_id,
            "status": replay.status,
            "step_range": f"{replay.start_step}-{replay.end_step}",
            "model": replay.replay_config.get("model") if replay.replay_config else None
        })
    
    # Sort by timestamp descending
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Limit to requested count
    activities = activities[:limit]
    
    return {
        "total_activities": len(activities),
        "activities": activities
    }


@router.get("/performance")
async def get_performance_metrics(
    time_window_days: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get performance metrics over time."""
    # Calculate time threshold
    time_threshold = datetime.now(timezone.utc) - timedelta(days=time_window_days)
    
    # Get sessions created in time window
    sessions_query = select(AgentSession).where(
        AgentSession.created_at >= time_threshold
    ).order_by(AgentSession.created_at)
    
    sessions_result = await db.execute(sessions_query)
    sessions = list(sessions_result.scalars().all())
    
    # Group by date
    daily_metrics = {}
    
    for session in sessions:
        date_str = session.created_at.date().isoformat()
        
        if date_str not in daily_metrics:
            daily_metrics[date_str] = {
                "date": date_str,
                "session_count": 0,
                "total_steps": 0,
                "total_decisions": 0,
                "avg_steps_per_session": 0,
                "avg_decisions_per_session": 0,
                "completed_sessions": 0
            }
        
        daily = daily_metrics[date_str]
        daily["session_count"] += 1
        daily["total_steps"] += session.total_steps
        daily["total_decisions"] += session.total_decisions
        
        if session.completed_at:
            daily["completed_sessions"] += 1
    
    # Calculate averages
    for date_str, daily in daily_metrics.items():
        if daily["session_count"] > 0:
            daily["avg_steps_per_session"] = daily["total_steps"] / daily["session_count"]
            daily["avg_decisions_per_session"] = daily["total_decisions"] / daily["session_count"]
    
    # Convert to list sorted by date
    daily_list = sorted(daily_metrics.values(), key=lambda x: x["date"])
    
    # Calculate overall metrics
    total_sessions_in_window = len(sessions)
    total_steps_in_window = sum(session.total_steps for session in sessions)
    total_decisions_in_window = sum(session.total_decisions for session in sessions)
    
    completed_sessions = len([s for s in sessions if s.completed_at])
    completion_rate = (completed_sessions / total_sessions_in_window * 100) if total_sessions_in_window > 0 else 0
    
    # Get step type distribution for time window
    step_type_query = select(
        SessionStep.step_type,
        func.count(SessionStep.id).label("count")
    ).join(AgentSession).where(
        AgentSession.created_at >= time_threshold
    ).group_by(SessionStep.step_type)
    
    step_type_result = await db.execute(step_type_query)
    step_type_distribution = {
        row.step_type: row.count
        for row in step_type_result.all()
    }
    
    # Get error rate
    error_count = step_type_distribution.get("error", 0)
    total_steps_in_window_from_db = sum(step_type_distribution.values())
    error_rate = (error_count / total_steps_in_window_from_db * 100) if total_steps_in_window_from_db > 0 else 0
    
    return {
        "time_window_days": time_window_days,
        "start_date": time_threshold.date().isoformat(),
        "end_date": datetime.now(timezone.utc).date().isoformat(),
        "overall_metrics": {
            "total_sessions": total_sessions_in_window,
            "completed_sessions": completed_sessions,
            "completion_rate": completion_rate,
            "total_steps": total_steps_in_window,
            "total_decisions": total_decisions_in_window,
            "avg_steps_per_session": total_steps_in_window / total_sessions_in_window if total_sessions_in_window > 0 else 0,
            "avg_decisions_per_session": total_decisions_in_window / total_sessions_in_window if total_sessions_in_window > 0 else 0,
            "error_count": error_count,
            "error_rate": error_rate
        },
        "daily_metrics": daily_list,
        "step_type_distribution": step_type_distribution
    }


@router.get("/agents")
async def get_agent_metrics(
    db: AsyncSession = Depends(get_db)
):
    """Get metrics grouped by agent."""
    # Get all sessions grouped by agent_id
    agent_query = select(
        AgentSession.agent_id,
        func.count(AgentSession.id).label("session_count"),
        func.sum(AgentSession.total_steps).label("total_steps"),
        func.sum(AgentSession.total_decisions).label("total_decisions"),
        func.avg(AgentSession.total_steps).label("avg_steps"),
        func.avg(AgentSession.total_decisions).label("avg_decisions")
    ).group_by(AgentSession.agent_id)
    
    agent_result = await db.execute(agent_query)
    agent_rows = agent_result.all()
    
    agent_metrics = []
    for row in agent_rows:
        # Get latest session for this agent
        latest_session_query = select(AgentSession).where(
            AgentSession.agent_id == row.agent_id
        ).order_by(desc(AgentSession.created_at)).limit(1)
        
        latest_session_result = await db.execute(latest_session_query)
        latest_session = latest_session_result.scalar_one_or_none()
        
        # Get error rate for this agent
        error_query = select(func.count(SessionStep.id)).where(
            SessionStep.agent_id == row.agent_id,
            SessionStep.step_type == "error"
        )
        
        error_result = await db.execute(error_query)
        error_count = error_result.scalar() or 0
        
        total_steps_for_agent = row.total_steps or 0
        error_rate = (error_count / total_steps_for_agent * 100) if total_steps_for_agent > 0 else 0
        
        agent_metrics.append({
            "agent_id": row.agent_id,
            "session_count": row.session_count,
            "total_steps": row.total_steps,
            "total_decisions": row.total_decisions,
            "avg_steps_per_session": float(row.avg_steps) if row.avg_steps else 0,
            "avg_decisions_per_session": float(row.avg_decisions) if row.avg_decisions else 0,
            "error_count": error_count,
            "error_rate": error_rate,
            "latest_session": {
                "id": latest_session.id if latest_session else None,
                "name": latest_session.name if latest_session else None,
                "created_at": latest_session.created_at.isoformat() if latest_session else None,
                "status": "completed" if latest_session and latest_session.completed_at else "in_progress"
            } if latest_session else None
        })
    
    # Sort by session count descending
    agent_metrics.sort(key=lambda x: x["session_count"], reverse=True)
    
    # Calculate totals
    total_agents = len(agent_metrics)
    total_sessions = sum(m["session_count"] for m in agent_metrics)
    total_steps = sum(m["total_steps"] for m in agent_metrics)
    
    return {
        "total_agents": total_agents,
        "total_sessions": total_sessions,
        "total_steps": total_steps,
        "agent_metrics": agent_metrics
    }


@router.get("/models")
async def get_model_metrics(
    db: AsyncSession = Depends(get_db)
):
    """Get metrics grouped by model."""
    # Get all sessions grouped by model
    model_query = select(
        AgentSession.model,
        func.count(AgentSession.id).label("session_count"),
        func.sum(AgentSession.total_steps).label("total_steps"),
        func.sum(AgentSession.total_decisions).label("total_decisions"),
        func.avg(AgentSession.total_steps).label("avg_steps"),
        func.avg(AgentSession.total_decisions).label("avg_decisions")
    ).group_by(AgentSession.model)
    
    model_result = await db.execute(model_query)
    model_rows = model_result.all()
    
    model_metrics = []
    for row in model_rows:
        # Calculate efficiency score (decisions per step)
        total_steps = row.total_steps or 0
        total_decisions = row.total_decisions or 0
        
        efficiency = (total_decisions / total_steps) if total_steps > 0 else 0
        
        # Get error rate for this model
        error_query = select(func.count(SessionStep.id)).where(
            SessionStep.model == row.model,
            SessionStep.step_type == "error"
        )
        
        error_result = await db.execute(error_query)
        error_count = error_result.scalar() or 0
        
        error_rate = (error_count / total_steps * 100) if total_steps > 0 else 0
        
        model_metrics.append({
            "model": row.model,
            "session_count": row.session_count,
            "total_steps": total_steps,
            "total_decisions": total_decisions,
            "avg_steps_per_session": float(row.avg_steps) if row.avg_steps else 0,
            "avg_decisions_per_session": float(row.avg_decisions) if row.avg_decisions else 0,
            "efficiency_score": efficiency,
            "error_count": error_count,
            "error_rate": error_rate,
            "performance_rating": _rate_model_performance(efficiency, error_rate)
        })
    
    # Sort by session count descending
    model_metrics.sort(key=lambda x: x["session_count"], reverse=True)
    
    # Find best performing model (highest efficiency, lowest error rate)
    if model_metrics:
        # Simple scoring: efficiency * (1 - error_rate/100)
        for metric in model_metrics:
            score = metric["efficiency_score"] * (1 - metric["error_rate"] / 100)
            metric["performance_score"] = score
        
        best_model = max(model_metrics, key=lambda x: x["performance_score"])
    else:
        best_model = None
    
    return {
        "total_models": len(model_metrics),
        "model_metrics": model_metrics,
        "best_performing_model": best_model
    }


def _rate_model_performance(efficiency: float, error_rate: float) -> str:
    """Rate model performance based on efficiency and error rate."""
    if efficiency > 0.3 and error_rate < 5:
        return "excellent"
    elif efficiency > 0.2 and error_rate < 10:
        return "good"
    elif efficiency > 0.1 and error_rate < 20:
        return "average"
    else:
        return "poor"


@router.get("/health")
async def get_system_health(
    db: AsyncSession = Depends(get_db)
):
    """Get system health status."""
    # Check database connection
    try:
        # Simple query to check database
        test_query = select(func.count(AgentSession.id))
        result = await db.execute(test_query)
        session_count = result.scalar() or 0
        
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        session_count = 0
    
    # Get system metrics
    metrics = {
        "database": db_status,
        "total_sessions": session_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": "N/A",  # Would need to track server start time
        "version": "1.0.0"
    }
    
    # Determine overall health
    if "unhealthy" in db_status:
        overall_health = "degraded"
    elif session_count == 0:
        overall_health = "no_data"
    else:
        overall_health = "healthy"
    
    return {
        "status": overall_health,
        "metrics": metrics,
        "checks": {
            "database": db_status,
            "has_data": session_count > 0
        }
    }