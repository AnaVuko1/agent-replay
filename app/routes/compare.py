from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List

from app.database import get_db
from app.models import AgentSession, SessionStep, Replay, ReplayStep
from app.schemas import AgentSessionResponse
from app.engine.comparator import Comparator

router = APIRouter(prefix="/api/v1/compare", tags=["compare"])


@router.get("/sessions")
async def compare_sessions(
    session1_id: str,
    session2_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Compare two different sessions."""
    # Get both sessions
    session1 = await db.get(AgentSession, session1_id)
    session2 = await db.get(AgentSession, session2_id)
    
    if not session1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session1_id} not found"
        )
    
    if not session2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session2_id} not found"
        )
    
    # Get steps for both sessions
    steps1_query = select(SessionStep).where(
        SessionStep.session_id == session1_id
    ).order_by(SessionStep.step_number)
    
    steps2_query = select(SessionStep).where(
        SessionStep.session_id == session2_id
    ).order_by(SessionStep.step_number)
    
    steps1_result = await db.execute(steps1_query)
    steps2_result = await db.execute(steps2_query)
    
    steps1 = list(steps1_result.scalars().all())
    steps2 = list(steps2_result.scalars().all())
    
    # Compare using comparator
    comparator = Comparator()
    comparison = comparator.compare_sessions(steps1, steps2)
    
    # Add session info
    comparison["session_info"] = {
        "session1": {
            "id": session1.id,
            "name": session1.name,
            "agent_id": session1.agent_id,
            "model": session1.model,
            "total_steps": session1.total_steps,
            "total_decisions": session1.total_decisions,
            "status": "completed" if session1.completed_at else "in_progress"
        },
        "session2": {
            "id": session2.id,
            "name": session2.name,
            "agent_id": session2.agent_id,
            "model": session2.model,
            "total_steps": session2.total_steps,
            "total_decisions": session2.total_decisions,
            "status": "completed" if session2.completed_at else "in_progress"
        }
    }
    
    return comparison


@router.get("/session_vs_replay")
async def compare_session_vs_replay(
    session_id: str,
    replay_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Compare original session to a replay."""
    # Get session and replay
    session = await db.get(AgentSession, session_id)
    replay = await db.get(Replay, replay_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    if not replay:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay {replay_id} not found"
        )
    
    if replay.session_id != session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Replay {replay_id} does not belong to session {session_id}"
        )
    
    # Get session steps in replay range
    session_steps_query = select(SessionStep).where(
        SessionStep.session_id == session_id,
        SessionStep.step_number >= replay.start_step,
        SessionStep.step_number <= replay.end_step
    ).order_by(SessionStep.step_number)
    
    session_steps_result = await db.execute(session_steps_query)
    session_steps = list(session_steps_result.scalars().all())
    
    # Get replay steps
    replay_steps_query = select(ReplayStep).where(
        ReplayStep.replay_id == replay_id
    ).order_by(ReplayStep.step_number)
    
    replay_steps_result = await db.execute(replay_steps_query)
    replay_steps = list(replay_steps_result.scalars().all())
    
    # Compare using comparator
    comparator = Comparator()
    comparison = comparator.compare_session_to_replay(session_steps, replay_steps)
    
    # Add metadata
    comparison["metadata"] = {
        "session_id": session_id,
        "replay_id": replay_id,
        "replay_name": replay.name,
        "replay_config": replay.replay_config,
        "step_range": {
            "start": replay.start_step,
            "end": replay.end_step
        },
        "replay_status": replay.status,
        "replay_created_at": replay.created_at.isoformat() if replay.created_at else None,
        "replay_completed_at": replay.completed_at.isoformat() if replay.completed_at else None
    }
    
    return comparison


@router.get("/multiple_replays")
async def compare_multiple_replays(
    replay_ids: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """Compare multiple replays at once."""
    # Parse replay IDs
    id_list = [rid.strip() for rid in replay_ids.split(",") if rid.strip()]
    
    if len(id_list) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 replay IDs required"
        )
    
    if len(id_list) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 replays can be compared at once"
        )
    
    # Get all replays
    replays = []
    replay_steps = {}
    
    for replay_id in id_list:
        replay = await db.get(Replay, replay_id)
        if not replay:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Replay {replay_id} not found"
            )
        
        if replay.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Replay {replay_id} is not completed"
            )
        
        replays.append(replay)
        
        # Get replay steps
        steps_query = select(ReplayStep).where(
            ReplayStep.replay_id == replay_id
        ).order_by(ReplayStep.step_number)
        
        steps_result = await db.execute(steps_query)
        replay_steps[replay_id] = list(steps_result.scalars().all())
    
    # Compare each pair
    comparisons = {}
    comparator = Comparator()
    
    for i in range(len(replays)):
        for j in range(i + 1, len(replays)):
            replay1 = replays[i]
            replay2 = replays[j]
            
            key = f"{replay1.id}_vs_{replay2.id}"
            
            comparison = comparator.compare_replays(
                replay1, replay_steps[replay1.id],
                replay2, replay_steps[replay2.id]
            )
            
            comparisons[key] = comparison
    
    # Calculate aggregated metrics
    aggregated = {
        "total_replays": len(replays),
        "replay_info": [
            {
                "id": replay.id,
                "name": replay.name,
                "session_id": replay.session_id,
                "start_step": replay.start_step,
                "end_step": replay.end_step,
                "model": replay.replay_config.get("model"),
                "temperature": replay.replay_config.get("temperature"),
                "total_steps": len(replay_steps[replay.id]),
                "avg_divergence": sum(
                    step.divergence_score or 0 
                    for step in replay_steps[replay.id]
                ) / len(replay_steps[replay.id]) if replay_steps[replay.id] else 0
            }
            for replay in replays
        ],
        "pairwise_comparisons": comparisons,
        "summary": _generate_multi_replay_summary(replays, replay_steps, comparisons)
    }
    
    return aggregated


def _generate_multi_replay_summary(replays, replay_steps, comparisons):
    """Generate summary for multiple replay comparison."""
    # Calculate average divergence for each replay
    avg_divergences = {}
    for replay in replays:
        steps = replay_steps[replay.id]
        if steps:
            avg_div = sum(step.divergence_score or 0 for step in steps) / len(steps)
            avg_divergences[replay.id] = avg_div
    
    # Find replay with lowest divergence (most similar to original)
    most_similar = min(avg_divergences.items(), key=lambda x: x[1]) if avg_divergences else None
    
    # Find replay with highest divergence (most different)
    most_different = max(avg_divergences.items(), key=lambda x: x[1]) if avg_divergences else None
    
    # Analyze configuration differences
    config_analysis = []
    for i in range(len(replays)):
        for j in range(i + 1, len(replays)):
            config1 = replays[i].replay_config
            config2 = replays[j].replay_config
            
            differences = []
            for key in set(config1.keys()) | set(config2.keys()):
                if config1.get(key) != config2.get(key):
                    differences.append({
                        "parameter": key,
                        f"replay_{i+1}": config1.get(key),
                        f"replay_{j+1}": config2.get(key)
                    })
            
            if differences:
                config_analysis.append({
                    "comparison": f"Replay {i+1} vs Replay {j+1}",
                    "differences": differences
                })
    
    return {
        "most_similar_to_original": {
            "replay_id": most_similar[0] if most_similar else None,
            "replay_name": next((r.name for r in replays if r.id == most_similar[0]), None) if most_similar else None,
            "avg_divergence": most_similar[1] if most_similar else None
        },
        "most_different_from_original": {
            "replay_id": most_different[0] if most_different else None,
            "replay_name": next((r.name for r in replays if r.id == most_different[0]), None) if most_different else None,
            "avg_divergence": most_different[1] if most_different else None
        },
        "configuration_analysis": config_analysis,
        "recommendations": _generate_multi_replay_recommendations(replays, avg_divergences, comparisons)
    }


def _generate_multi_replay_recommendations(replays, avg_divergences, comparisons):
    """Generate recommendations from multiple replay comparison."""
    recommendations = []
    
    if not avg_divergences:
        return recommendations
    
    # Check if any replay has very low divergence
    low_divergence_replays = [
        (replay_id, div) for replay_id, div in avg_divergences.items()
        if div < 0.1
    ]
    
    if low_divergence_replays:
        replay_ids = [rid for rid, _ in low_divergence_replays]
        replay_names = [
            next(r.name for r in replays if r.id == rid)
            for rid in replay_ids
        ]
        
        recommendations.append(
            f"Replays {', '.join(replay_names)} have very low divergence (<0.1). "
            "Their configurations produce results very similar to the original."
        )
    
    # Check if any replay has very high divergence
    high_divergence_replays = [
        (replay_id, div) for replay_id, div in avg_divergences.items()
        if div > 0.5
    ]
    
    if high_divergence_replays:
        replay_ids = [rid for rid, _ in high_divergence_replays]
        replay_names = [
            next(r.name for r in replays if r.id == rid)
            for rid in replay_ids
        ]
        
        recommendations.append(
            f"Replays {', '.join(replay_names)} have high divergence (>0.5). "
            "Their configurations significantly change agent behavior."
        )
    
    # Compare model choices
    models = {}
    for replay in replays:
        model = replay.replay_config.get("model")
        if model:
            models.setdefault(model, []).append(replay.name)
    
    if len(models) > 1:
        model_comparison = ", ".join([f"{model} ({', '.join(names)})" for model, names in models.items()])
        recommendations.append(
            f"Different models used: {model_comparison}. "
            "Compare model performance for this task."
        )
    
    # Check temperature variations
    temperatures = {}
    for replay in replays:
        temp = replay.replay_config.get("temperature")
        if temp is not None:
            temperatures.setdefault(temp, []).append(replay.name)
    
    if len(temperatures) > 1:
        temp_comparison = ", ".join([f"{temp} ({', '.join(names)})" for temp, names in temperatures.items()])
        recommendations.append(
            f"Different temperatures used: {temp_comparison}. "
            "Analyze temperature effects on decision consistency."
        )
    
    # General recommendation
    if len(replays) >= 3:
        recommendations.append(
            f"Compared {len(replays)} replays. "
            "Consider creating a 'best of' configuration combining elements from top-performing replays."
        )
    
    return recommendations


@router.get("/by_pattern")
async def compare_by_pattern(
    pattern: str = Query(..., description="Pattern to search for in step content"),
    session_ids: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Compare sessions/replays by searching for specific patterns."""
    # Parse session IDs if provided
    session_id_list = []
    if session_ids:
        session_id_list = [sid.strip() for sid in session_ids.split(",") if sid.strip()]
    
    # Build query
    query = select(SessionStep).where(
        SessionStep.content.ilike(f"%{pattern}%")
    ).order_by(SessionStep.step_number)
    
    if session_id_list:
        query = query.where(SessionStep.session_id.in_(session_id_list))
    
    query = query.limit(limit)
    
    result = await db.execute(query)
    matching_steps = list(result.scalars().all())
    
    # Group by session
    steps_by_session = {}
    for step in matching_steps:
        session_id = step.session_id
        if session_id not in steps_by_session:
            steps_by_session[session_id] = []
        steps_by_session[session_id].append(step)
    
    # Get session info
    session_info = {}
    for session_id in steps_by_session.keys():
        session = await db.get(AgentSession, session_id)
        if session:
            session_info[session_id] = {
                "id": session.id,
                "name": session.name,
                "agent_id": session.agent_id,
                "model": session.model
            }
    
    # Analyze pattern usage
    pattern_analysis = {}
    for session_id, steps in steps_by_session.items():
        step_types = {}
        step_numbers = []
        
        for step in steps:
            step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
            step_numbers.append(step.step_number)
        
        pattern_analysis[session_id] = {
            "total_matches": len(steps),
            "step_type_distribution": step_types,
            "step_numbers": sorted(step_numbers),
            "first_match_at": min(step_numbers) if step_numbers else None,
            "last_match_at": max(step_numbers) if step_numbers else None,
            "content_samples": [
                {
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "content": step.content[:200]
                }
                for step in steps[:3]  # First 3 matches
            ]
        }
    
    return {
        "pattern": pattern,
        "total_matches": len(matching_steps),
        "sessions_with_matches": len(steps_by_session),
        "session_info": session_info,
        "pattern_analysis": pattern_analysis,
        "cross_session_comparison": _compare_pattern_usage(pattern_analysis, session_info)
    }


def _compare_pattern_usage(pattern_analysis, session_info):
    """Compare pattern usage across sessions."""
    if len(pattern_analysis) < 2:
        return {"message": "Need at least 2 sessions with matches for comparison"}
    
    comparison = {}
    
    # Compare match counts
    match_counts = {
        session_id: data["total_matches"]
        for session_id, data in pattern_analysis.items()
    }
    
    max_matches = max(match_counts.values())
    min_matches = min(match_counts.values())
    
    comparison["match_count_comparison"] = {
        "max_matches": max_matches,
        "max_matches_session": next(
            sid for sid, count in match_counts.items() 
            if count == max_matches
        ),
        "min_matches": min_matches,
        "min_matches_session": next(
            sid for sid, count in match_counts.items() 
            if count == min_matches
        ),
        "ratio": max_matches / min_matches if min_matches > 0 else float('inf')
    }
    
    # Compare step types
    step_type_comparison = {}
    all_step_types = set()
    
    for session_id, data in pattern_analysis.items():
        all_step_types.update(data["step_type_distribution"].keys())
    
    for step_type in all_step_types:
        counts = {}
        for session_id, data in pattern_analysis.items():
            counts[session_id] = data["step_type_distribution"].get(step_type, 0)
        
        if any(count > 0 for count in counts.values()):
            step_type_comparison[step_type] = {
                "counts": counts,
                "most_common_in": max(counts.items(), key=lambda x: x[1])[0] if counts else None
            }
    
    comparison["step_type_comparison"] = step_type_comparison
    
    # Compare when pattern appears
    first_match_comparison = {}
    for session_id, data in pattern_analysis.items():
        first_match = data.get("first_match_at")
        if first_match:
            session_name = session_info.get(session_id, {}).get("name", session_id)
            first_match_comparison[session_name] = {
                "first_match_at": first_match,
                "session_id": session_id
            }
    
    comparison["first_match_timing"] = first_match_comparison
    
    # Generate insights
    insights = []
    
    if max_matches > min_matches * 3:
        insights.append(
            f"Pattern appears {max_matches/min_matches:.1f}x more in one session. "
            "This may indicate different problem-solving approaches."
        )
    
    # Check if pattern appears in different step types
    diverse_step_types = False
    for step_type, data in step_type_comparison.items():
        if len([sid for sid, count in data["counts"].items() if count > 0]) > 1:
            diverse_step_types = True
            break
    
    if diverse_step_types:
        insights.append(
            "Pattern appears in different step types across sessions. "
            "Agents use this pattern in different contexts."
        )
    
    comparison["insights"] = insights
    
    return comparison