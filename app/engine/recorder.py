import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.models import AgentSession, SessionStep
from app.schemas import SessionCreate, SessionStepCreate, Metadata
from app.config import settings


class SessionRecorder:
    """Records agent sessions step-by-step."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_session(self, session_data: SessionCreate) -> AgentSession:
        """Create a new agent session."""
        session_id = str(uuid.uuid4())
        
        session = AgentSession(
            id=session_id,
            name=session_data.name,
            agent_id=session_data.agent_id,
            model=session_data.model,
            meta_data=session_data.metadata or {}
        )
        
        self.db.add(session)
        await self.db.flush()
        return session
    
    async def record_step(
        self,
        session_id: str,
        step_data: SessionStepCreate
    ) -> SessionStep:
        """Record a single step in a session."""
        # Get session to ensure it exists
        session = await self.db.get(AgentSession, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Validate step number
        if step_data.step_number != session.total_steps + 1:
            raise ValueError(
                f"Expected step number {session.total_steps + 1}, "
                f"got {step_data.step_number}"
            )
        
        # Create step ID
        step_id = str(uuid.uuid4())
        
        # Convert metadata to dict
        metadata_dict = {}
        if step_data.metadata:
            metadata_dict = step_data.metadata.model_dump()
        
        # Truncate context snapshot if too large
        context_snapshot = step_data.context_snapshot
        if context_snapshot and len(context_snapshot) > settings.max_context_snapshot_size:
            context_snapshot = context_snapshot[:settings.max_context_snapshot_size]
        
        # Detect step type if not specified
        step_type = step_data.step_type or self._detect_step_type(step_data.content)
        
        # Create step
        step = SessionStep(
            id=step_id,
            session_id=session_id,
            step_number=step_data.step_number,
            step_type=step_type,
            agent_id=step_data.agent_id or session.agent_id,
            model=step_data.model or session.model,
            content=step_data.content,
            meta_data=metadata_dict,
            context_snapshot=context_snapshot
        )
        
        # Update session stats
        session.total_steps += 1
        if step_type == "decision":
            session.total_decisions += 1
        
        self.db.add(step)
        await self.db.flush()
        
        # Update session metadata
        if step_type == "error" and step_data.metadata and step_data.metadata.error_message:
            session.meta_data.setdefault("errors", []).append({
                "step": step_data.step_number,
                "message": step_data.metadata.error_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return step
    
    async def record_batch_steps(
        self,
        session_id: str,
        steps_data: List[SessionStepCreate]
    ) -> List[SessionStep]:
        """Record multiple steps in a batch."""
        session = await self.db.get(AgentSession, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        steps = []
        expected_step_number = session.total_steps + 1
        
        for step_data in steps_data:
            if step_data.step_number != expected_step_number:
                raise ValueError(
                    f"Expected step number {expected_step_number}, "
                    f"got {step_data.step_number}"
                )
            
            step = await self.record_step(session_id, step_data)
            steps.append(step)
            expected_step_number += 1
        
        await self.db.flush()
        return steps
    
    def _detect_step_type(self, content: str) -> str:
        """Auto-detect step type from content structure."""
        content_lower = content.lower()
        
        # Check for tool call patterns
        if any(pattern in content_lower for pattern in ["tool_call", "calling ", "executing ", "function "]):
            return "tool_call"
        
        # Check for tool result patterns
        if any(pattern in content_lower for pattern in ["result:", "output:", "returned:", "tool_result"]):
            return "tool_result"
        
        # Check for error patterns
        if any(pattern in content_lower for pattern in ["error", "exception", "failed", "traceback"]):
            return "error"
        
        # Check for decision patterns
        if any(pattern in content_lower for pattern in ["decide", "choose", "select", "decision"]):
            return "decision"
        
        # Check for observation patterns
        if any(pattern in content_lower for pattern in ["observe", "see", "notice", "observation"]):
            return "observation"
        
        # Default to think
        return "think"
    
    async def complete_session(self, session_id: str) -> AgentSession:
        """Mark a session as completed."""
        session = await self.db.get(AgentSession, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.completed_at = datetime.now(timezone.utc)
        await self.db.flush()
        return session
    
    async def get_session_steps(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[SessionStep]:
        """Get steps for a session with pagination."""
        query = select(SessionStep).where(
            SessionStep.session_id == session_id
        ).order_by(SessionStep.step_number)
        
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID with its steps."""
        session = await self.db.get(AgentSession, session_id)
        if session:
            # Eager load steps
            result = await self.db.execute(
                select(SessionStep)
                .where(SessionStep.session_id == session_id)
                .order_by(SessionStep.step_number)
            )
            session.steps = list(result.scalars().all())
        return session