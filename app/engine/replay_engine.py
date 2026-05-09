import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import difflib

from app.models import SessionStep, Replay, ReplayStep, AgentSession
from app.schemas import ReplayCreate, ReplayConfig, ReplayMetrics, ReplayStepDiff
from app.engine.timeline import TimelineEngine


class ReplayEngine:
    """Simulates replay of agent decisions with modified parameters."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.timeline_engine = TimelineEngine()
    
    async def create_replay(
        self,
        session_id: str,
        replay_data: ReplayCreate
    ) -> Replay:
        """Create a new replay configuration."""
        # Get session
        session = await self.db.get(AgentSession, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Validate step range
        if replay_data.end_step > session.total_steps:
            raise ValueError(
                f"Session only has {session.total_steps} steps, "
                f"requested end_step {replay_data.end_step}"
            )
        
        # Create replay
        replay_id = str(uuid.uuid4())
        
        replay = Replay(
            id=replay_id,
            session_id=session_id,
            name=replay_data.name or f"Replay of steps {replay_data.start_step}-{replay_data.end_step}",
            start_step=replay_data.start_step,
            end_step=replay_data.end_step,
            replay_config=replay_data.replay_config.model_dump(),
            status="pending",
            total_steps=replay_data.end_step - replay_data.start_step + 1
        )
        
        self.db.add(replay)
        await self.db.flush()
        
        return replay
    
    async def execute_replay(self, replay_id: str) -> Replay:
        """Execute a replay (simulated - uses recorded data with modifications)."""
        replay = await self.db.get(Replay, replay_id)
        if not replay:
            raise ValueError(f"Replay {replay_id} not found")
        
        if replay.status == "completed":
            return replay
        
        # Get session and steps
        session = await self.db.get(AgentSession, replay.session_id)
        if not session:
            raise ValueError(f"Session {replay.session_id} not found")
        
        # Get steps in replay range
        result = await self.db.execute(
            select(SessionStep)
            .where(SessionStep.session_id == replay.session_id)
            .where(SessionStep.step_number >= replay.start_step)
            .where(SessionStep.step_number <= replay.end_step)
            .order_by(SessionStep.step_number)
        )
        original_steps = list(result.scalars().all())
        
        if not original_steps:
            raise ValueError(f"No steps found in range {replay.start_step}-{replay.end_step}")
        
        # Update replay status
        replay.status = "running"
        await self.db.flush()
        
        # Execute simulation
        replay_steps = []
        replay_config = replay.replay_config
        
        for original_step in original_steps:
            replayed_content = self._simulate_replayed_step(
                original_step, 
                replay_config,
                replay_steps  # For context
            )
            
            # Calculate divergence
            divergence_score = self._calculate_divergence(
                original_step.content,
                replayed_content
            )
            
            # Create replay step
            replay_step = ReplayStep(
                id=str(uuid.uuid4()),
                replay_id=replay_id,
                original_step_id=original_step.id,
                step_number=original_step.step_number,
                step_type=original_step.step_type,
                content=replayed_content,
                original_content=original_step.content,
                metadata=original_step.meta_data or {},
                divergence_score=divergence_score
            )
            
            self.db.add(replay_step)
            replay_steps.append(replay_step)
        
        # Complete replay
        replay.status = "completed"
        replay.completed_at = datetime.now(timezone.utc)
        
        # Calculate metrics
        metrics = self._calculate_replay_metrics(replay_steps)
        replay.meta_data = {"metrics": metrics.model_dump()}
        
        await self.db.flush()
        
        return replay
    
    def _simulate_replayed_step(
        self,
        original_step: SessionStep,
        replay_config: Dict[str, Any],
        previous_replay_steps: List[ReplayStep]
    ) -> str:
        """Simulate what the step would look like with replay configuration."""
        step_type = original_step.step_type
        
        if step_type in ["think", "decision"]:
            # Apply model and temperature changes to cognitive steps
            return self._modify_cognitive_step(
                original_step.content,
                replay_config,
                previous_replay_steps
            )
        elif step_type == "tool_call":
            # Apply constraints to tool calls
            return self._modify_tool_call(
                original_step.content,
                replay_config,
                original_step.meta_data or {}
            )
        else:
            # tool_result, observation, error - usually unchanged
            return original_step.content
    
    def _modify_cognitive_step(
        self,
        original_content: str,
        replay_config: Dict[str, Any],
        previous_steps: List[ReplayStep]
    ) -> str:
        """Modify think/decision steps based on replay configuration."""
        content = original_content
        
        # Apply system prompt if present
        system_prompt = replay_config.get("system_prompt")
        if system_prompt:
            # Check if content references system instructions
            if "system:" in content.lower() or "instructions:" in content.lower():
                # Replace system instructions
                lines = content.split('\n')
                modified_lines = []
                for line in lines:
                    if "system:" in line.lower() or "instructions:" in line.lower():
                        modified_lines.append(f"System: {system_prompt}")
                    else:
                        modified_lines.append(line)
                content = '\n'.join(modified_lines)
            else:
                # Prepend system context
                content = f"[System context: {system_prompt}]\n\n{content}"
        
        # Apply temperature effects for decisions
        temperature = replay_config.get("temperature", 0.7)
        original_temp = 0.7  # Assumed default
        
        if temperature != original_temp:
            # Simulate temperature effect on decision confidence
            if "confident" in content.lower() or "certain" in content.lower():
                if temperature > original_temp:
                    # Higher temp = less certain
                    content = content.replace("confident", "somewhat confident")
                    content = content.replace("certain", "fairly certain")
                    content = content.replace("definitely", "probably")
                else:
                    # Lower temp = more certain
                    content = content.replace("somewhat confident", "confident")
                    content = content.replace("fairly certain", "certain")
                    content = content.replace("probably", "definitely")
        
        # Apply constraints if present
        constraints = replay_config.get("constraints")
        if constraints:
            constraint_text = " [Constraints: " + "; ".join(constraints) + "]"
            
            # Append constraints to end of thought
            if content.endswith(('.', '!', '?')):
                content = content[:-1] + constraint_text + content[-1]
            else:
                content += constraint_text
        
        return content
    
    def _modify_tool_call(
        self,
        original_content: str,
        replay_config: Dict[str, Any],
        original_metadata: Dict[str, Any]
    ) -> str:
        """Modify tool calls based on replay configuration."""
        content = original_content
        
        # Apply tool configuration
        tool_config = replay_config.get("tool_config")
        if tool_config:
            # Check if tool call matches configurable tools
            tool_name = original_metadata.get("tool_name", "").lower()
            
            for tool, config in tool_config.items():
                if tool.lower() in tool_name or tool_name in tool.lower():
                    # Apply tool-specific modifications
                    if "params" in config:
                        # Modify parameters in content
                        for param, value in config["params"].items():
                            if f"{param}=" in content:
                                # Replace parameter value
                                import re
                                pattern = f"{param}=([^,\\s}}]+)"
                                content = re.sub(pattern, f"{param}={value}", content)
        
        # Apply constraints
        constraints = replay_config.get("constraints")
        if constraints:
            # Check if tool call violates constraints
            content_lower = content.lower()
            for constraint in constraints:
                constraint_lower = constraint.lower()
                if any(word in content_lower for word in constraint_lower.split()):
                    # Add constraint note
                    content = f"[Constrained: {constraint}] {content}"
        
        return content
    
    def _calculate_divergence(self, original: str, replayed: str) -> float:
        """Calculate divergence score between original and replayed content (0-1)."""
        if original == replayed:
            return 0.0
        
        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, original, replayed).ratio()
        divergence = 1.0 - similarity
        
        # Boost divergence for structural changes
        lines_original = original.split('\n')
        lines_replayed = replayed.split('\n')
        
        if len(lines_original) != len(lines_replayed):
            divergence = min(1.0, divergence + 0.2)
        
        # Boost for keyword changes
        keywords = ["system:", "constraint", "confident", "certain", "definitely"]
        original_lower = original.lower()
        replayed_lower = replayed.lower()
        
        keyword_changes = 0
        for keyword in keywords:
            in_original = keyword in original_lower
            in_replayed = keyword in replayed_lower
            if in_original != in_replayed:
                keyword_changes += 1
        
        divergence = min(1.0, divergence + (keyword_changes * 0.05))
        
        return round(divergence, 3)
    
    def _calculate_replay_metrics(self, replay_steps: List[ReplayStep]) -> ReplayMetrics:
        """Calculate metrics for a completed replay."""
        if not replay_steps:
            return ReplayMetrics(
                total_steps=0,
                steps_diverged=0,
                avg_divergence_score=0.0,
                latency_delta_ms=0.0,
                token_delta=0,
                outcome_diff=None
            )
        
        # Count diverged steps
        steps_diverged = sum(1 for step in replay_steps if step.divergence_score > 0.1)
        
        # Calculate average divergence
        total_divergence = sum(step.divergence_score or 0.0 for step in replay_steps)
        avg_divergence = total_divergence / len(replay_steps)
        
        # Calculate token delta (simulated)
        token_delta = 0
        for step in replay_steps:
            original_len = len(step.original_content.split())
            replayed_len = len(step.content.split())
            token_delta += replayed_len - original_len
        
        # Simulate latency delta (based on model changes)
        latency_delta_ms = 0.0
        for step in replay_steps:
            if step.divergence_score > 0.3:
                # High divergence might indicate different processing time
                latency_delta_ms += step.divergence_score * 100  # milliseconds
        
        # Determine outcome difference
        outcome_diff = None
        if steps_diverged > 0:
            # Check last step for significant divergence
            last_step = replay_steps[-1]
            if last_step.divergence_score > 0.5:
                outcome_diff = "different_final_decision"
            elif last_step.divergence_score > 0.2:
                outcome_diff = "modified_final_decision"
            else:
                outcome_diff = "similar_outcome"
        
        return ReplayMetrics(
            total_steps=len(replay_steps),
            steps_diverged=steps_diverged,
            avg_divergence_score=avg_divergence,
            latency_delta_ms=latency_delta_ms,
            token_delta=token_delta,
            outcome_diff=outcome_diff
        )
    
    async def get_replay_comparison(
        self,
        replay_id: str,
        baseline_type: str = "original",
        baseline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare a replay against its baseline."""
        replay = await self.db.get(Replay, replay_id)
        if not replay:
            raise ValueError(f"Replay {replay_id} not found")
        
        if replay.status != "completed":
            raise ValueError(f"Replay {replay_id} is not completed")
        
        # Get replay steps
        result = await self.db.execute(
            select(ReplayStep)
            .where(ReplayStep.replay_id == replay_id)
            .order_by(ReplayStep.step_number)
        )
        replay_steps = list(result.scalars().all())
        
        # Get baseline steps
        if baseline_type == "original":
            baseline_steps = await self._get_original_steps(
                replay.session_id,
                replay.start_step,
                replay.end_step
            )
        elif baseline_type == "another_replay":
            if not baseline_id:
                raise ValueError("baseline_id required for another_replay comparison")
            
            baseline_replay = await self.db.get(Replay, baseline_id)
            if not baseline_replay:
                raise ValueError(f"Baseline replay {baseline_id} not found")
            
            baseline_steps = await self._get_replay_steps(baseline_id)
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")
        
        # Calculate diff for each step
        step_diffs = []
        for replay_step, baseline_step in zip(replay_steps, baseline_steps):
            diff_html = self._generate_diff_html(
                baseline_step.get("content", ""),
                replay_step.content
            )
            
            # Calculate divergence if not already calculated
            divergence = replay_step.divergence_score
            if divergence is None:
                divergence = self._calculate_divergence(
                    baseline_step.get("content", ""),
                    replay_step.content
                )
            
            step_diffs.append(ReplayStepDiff(
                step_number=replay_step.step_number,
                original_content=baseline_step.get("content", ""),
                replayed_content=replay_step.content,
                divergence_score=divergence,
                diff_html=diff_html
            ))
        
        # Calculate comparison metrics
        metrics = self._calculate_replay_metrics(replay_steps)
        
        # Add baseline comparison if available
        if baseline_type == "another_replay" and baseline_id:
            baseline_replay = await self.db.get(Replay, baseline_id)
            if baseline_replay and "metrics" in (baseline_replay.meta_data or {}):
                baseline_metrics = baseline_replay.meta_data["metrics"]
                metrics.replay_metrics = baseline_metrics
        
        return {
            "replay_id": replay_id,
            "baseline_type": baseline_type,
            "baseline_id": baseline_id,
            "metrics": metrics,
            "step_diffs": step_diffs
        }
    
    async def _get_original_steps(
        self,
        session_id: str,
        start_step: int,
        end_step: int
    ) -> List[Dict[str, Any]]:
        """Get original steps for a range."""
        result = await self.db.execute(
            select(SessionStep)
            .where(SessionStep.session_id == session_id)
            .where(SessionStep.step_number >= start_step)
            .where(SessionStep.step_number <= end_step)
            .order_by(SessionStep.step_number)
        )
        
        steps = list(result.scalars().all())
        return [
            {
                "id": step.id,
                "step_number": step.step_number,
                "content": step.content,
                "step_type": step.step_type,
                "metadata": step.meta_data or {}
            }
            for step in steps
        ]
    
    async def _get_replay_steps(self, replay_id: str) -> List[Dict[str, Any]]:
        """Get steps from another replay."""
        result = await self.db.execute(
            select(ReplayStep)
            .where(ReplayStep.replay_id == replay_id)
            .order_by(ReplayStep.step_number)
        )
        
        steps = list(result.scalars().all())
        return [
            {
                "id": step.id,
                "step_number": step.step_number,
                "content": step.content,
                "step_type": step.step_type,
                "metadata": step.meta_data or {}
            }
            for step in steps
        ]
    
    def _generate_diff_html(self, original: str, replayed: str) -> str:
        """Generate HTML diff between original and replayed content."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            replayed.splitlines(keepends=True),
            fromfile='original',
            tofile='replayed',
            lineterm=''
        )
        
        html_lines = ['<div class="diff-container">']
        
        for line in diff:
            if line.startswith('---') or line.startswith('+++'):
                continue
            elif line.startswith('@@'):
                html_lines.append(f'<div class="diff-header">{line}</div>')
            elif line.startswith('+'):
                html_lines.append(f'<div class="diff-added">{line[1:]}</div>')
            elif line.startswith('-'):
                html_lines.append(f'<div class="diff-removed">{line[1:]}</div>')
            else:
                html_lines.append(f'<div class="diff-context">{line}</div>')
        
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    async def batch_replay(
        self,
        session_id: str,
        replay_configs: List[ReplayCreate]
    ) -> List[Replay]:
        """Execute multiple replays in batch."""
        replays = []
        
        for config in replay_configs:
            # Create replay
            replay = await self.create_replay(session_id, config)
            
            # Execute replay
            try:
                replay = await self.execute_replay(replay.id)
                replays.append(replay)
            except Exception as e:
                replay.status = "failed"
                replay.meta_data = {"error": str(e)}
                await self.db.flush()
                replays.append(replay)
        
        return replays
    
    async def get_replay_analysis(self, replay_id: str) -> Dict[str, Any]:
        """Get detailed analysis of a replay."""
        replay = await self.db.get(Replay, replay_id)
        if not replay:
            raise ValueError(f"Replay {replay_id} not found")
        
        # Get replay steps
        result = await self.db.execute(
            select(ReplayStep)
            .where(ReplayStep.replay_id == replay_id)
            .order_by(ReplayStep.step_number)
        )
        replay_steps = list(result.scalars().all())
        
        # Analyze divergence patterns
        high_divergence_steps = [
            step for step in replay_steps 
            if step.divergence_score and step.divergence_score > 0.3
        ]
        
        divergence_by_type = {}
        for step in replay_steps:
            if step.divergence_score and step.divergence_score > 0:
                step_type = step.step_type
                divergence_by_type[step_type] = divergence_by_type.get(step_type, 0) + 1
        
        # Calculate when divergence started
        first_divergence_step = None
        for step in replay_steps:
            if step.divergence_score and step.divergence_score > 0.1:
                first_divergence_step = step.step_number
                break
        
        # Analyze impact of configuration changes
        config_impact = self._analyze_config_impact(replay.replay_config, replay_steps)
        
        return {
            "replay_id": replay_id,
            "total_steps": len(replay_steps),
            "steps_diverged": len(high_divergence_steps),
            "first_divergence_at": first_divergence_step,
            "divergence_by_step_type": divergence_by_type,
            "high_divergence_steps": [
                {
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "divergence_score": step.divergence_score,
                    "original_preview": step.original_content[:100],
                    "replayed_preview": step.content[:100]
                }
                for step in high_divergence_steps[:5]  # Top 5
            ],
            "config_impact": config_impact,
            "recommendations": self._generate_recommendations(
                replay.replay_config,
                high_divergence_steps
            )
        }
    
    def _analyze_config_impact(
        self,
        replay_config: Dict[str, Any],
        replay_steps: List[ReplayStep]
    ) -> Dict[str, Any]:
        """Analyze which configuration changes caused the most divergence."""
        impact = {
            "model_change": False,
            "temperature_change": False,
            "system_prompt_change": False,
            "tool_config_change": False,
            "constraints_added": False
        }
        
        # Check model change
        if "model" in replay_config:
            impact["model_change"] = True
        
        # Check temperature change
        if "temperature" in replay_config and replay_config["temperature"] != 0.7:
            impact["temperature_change"] = True
        
        # Check system prompt
        if "system_prompt" in replay_config and replay_config["system_prompt"]:
            impact["system_prompt_change"] = True
        
        # Check tool config
        if "tool_config" in replay_config and replay_config["tool_config"]:
            impact["tool_config_change"] = True
        
        # Check constraints
        if "constraints" in replay_config and replay_config["constraints"]:
            impact["constraints_added"] = True
        
        # Correlate with divergence
        config_causes = []
        for step in replay_steps:
            if step.divergence_score and step.divergence_score > 0.3:
                step_type = step.step_type
                
                if step_type in ["think", "decision"]:
                    if impact["system_prompt_change"]:
                        config_causes.append("system_prompt")
                    if impact["temperature_change"]:
                        config_causes.append("temperature")
                
                if step_type == "tool_call":
                    if impact["tool_config_change"]:
                        config_causes.append("tool_config")
                    if impact["constraints_added"]:
                        config_causes.append("constraints")
        
        # Count causes
        cause_counts = {}
        for cause in config_causes:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        
        impact["primary_cause"] = max(cause_counts.items(), key=lambda x: x[1])[0] if cause_counts else None
        impact["cause_distribution"] = cause_counts
        
        return impact
    
    def _generate_recommendations(
        self,
        replay_config: Dict[str, Any],
        high_divergence_steps: List[ReplayStep]
    ) -> List[str]:
        """Generate recommendations based on replay analysis."""
        recommendations = []
        
        if not high_divergence_steps:
            recommendations.append("No significant divergence detected. Current configuration produces similar results.")
            return recommendations
        
        # Check system prompt impact
        if "system_prompt" in replay_config:
            system_prompt_steps = [
                step for step in high_divergence_steps
                if step.step_type in ["think", "decision"]
            ]
            if system_prompt_steps:
                recommendations.append(
                    f"System prompt caused divergence in {len(system_prompt_steps)} cognitive steps. "
                    "Consider whether the new prompt improves decision quality."
                )
        
        # Check temperature impact
        if "temperature" in replay_config:
            temp = replay_config["temperature"]
            if temp < 0.5:
                recommendations.append(
                    f"Low temperature ({temp}) may be causing overly deterministic decisions. "
                    "Consider increasing slightly for more creative solutions."
                )
            elif temp > 0.9:
                recommendations.append(
                    f"High temperature ({temp}) may be causing inconsistent decisions. "
                    "Consider decreasing for more reliable outcomes."
                )
        
        # Check constraints
        if "constraints" in replay_config and replay_config["constraints"]:
            constrained_steps = [
                step for step in high_divergence_steps
                if step.step_type == "tool_call" and "Constrained" in step.content
            ]
            if constrained_steps:
                recommendations.append(
                    f"Constraints affected {len(constrained_steps)} tool calls. "
                    "Verify constraints are not preventing necessary actions."
                )
        
        # General recommendation
        if len(high_divergence_steps) > len(high_divergence_steps) / 2:
            recommendations.append(
                "High divergence across many steps. Consider whether the new configuration "
                "improves overall agent performance or introduces undesirable changes."
            )
        else:
            recommendations.append(
                "Divergence is limited to specific steps. Review those steps to understand "
                "whether the changes are beneficial."
            )
        
        return recommendations