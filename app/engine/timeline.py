from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import re

from app.models import SessionStep, AgentSession
from app.schemas import DecisionCycle, TimelineResponse, StateSnapshot
from app.config import settings


class TimelineEngine:
    """Builds interactive timelines from recorded steps."""
    
    def __init__(self):
        self.cycle_patterns = [
            # Classic think → tool_call → tool_result → observation → decision
            ["think", "tool_call", "tool_result", "observation", "decision"],
            # Simple think → decision
            ["think", "decision"],
            # Tool call with immediate result
            ["tool_call", "tool_result", "observation"],
            # Error recovery pattern
            ["think", "tool_call", "error", "think", "tool_call", "tool_result"],
        ]
    
    async def build_timeline(self, session: AgentSession) -> TimelineResponse:
        """Build a complete timeline from session steps."""
        if not session.steps:
            return TimelineResponse(
                session_id=session.id,
                total_steps=0,
                decision_cycles=[],
                step_density={}
            )
        
        # Sort steps by step number
        steps = sorted(session.steps, key=lambda s: s.step_number)
        
        # Calculate step density
        step_density = defaultdict(int)
        for step in steps:
            step_density[step.step_type] += 1
        
        # Group steps into decision cycles
        decision_cycles = self._group_into_cycles(steps)
        
        return TimelineResponse(
            session_id=session.id,
            total_steps=len(steps),
            decision_cycles=decision_cycles,
            step_density=dict(step_density)
        )
    
    def _group_into_cycles(self, steps: List[SessionStep]) -> List[DecisionCycle]:
        """Group steps into natural decision cycles."""
        cycles = []
        current_cycle = []
        cycle_number = 1
        
        for i, step in enumerate(steps):
            current_cycle.append(step)
            
            # Check if this step completes a cycle
            if self._is_cycle_complete(current_cycle):
                cycles.append(self._create_decision_cycle(
                    cycle_number, current_cycle, steps, i
                ))
                current_cycle = []
                cycle_number += 1
        
        # Handle any remaining steps as the last cycle
        if current_cycle:
            cycles.append(self._create_decision_cycle(
                cycle_number, current_cycle, steps, len(steps) - 1
            ))
        
        return cycles
    
    def _is_cycle_complete(self, cycle_steps: List[SessionStep]) -> bool:
        """Check if a set of steps forms a complete decision cycle."""
        step_types = [step.step_type for step in cycle_steps]
        
        # Check against known patterns
        for pattern in self.cycle_patterns:
            if len(step_types) < len(pattern):
                continue
            
            # Check if the end of current steps matches a pattern
            if step_types[-len(pattern):] == pattern:
                return True
        
        # Check for decision markers
        if "decision" in step_types:
            return True
        
        # Check for error followed by think (recovery attempt)
        if "error" in step_types and "think" in step_types[-2:]:
            return True
        
        # Long cycles complete after 8 steps max
        if len(cycle_steps) >= 8:
            return True
        
        return False
    
    def _create_decision_cycle(
        self,
        cycle_number: int,
        cycle_steps: List[SessionStep],
        all_steps: List[SessionStep],
        end_index: int
    ) -> DecisionCycle:
        """Create a DecisionCycle object from steps."""
        start_step = cycle_steps[0].step_number
        end_step = cycle_steps[-1].step_number
        
        # Calculate latency
        latency_ms = None
        if len(cycle_steps) > 1:
            first_time = cycle_steps[0].timestamp
            last_time = cycle_steps[-1].timestamp
            if first_time and last_time:
                latency_ms = (last_time - first_time).total_seconds() * 1000
        
        # Determine decision type
        decision_type = self._determine_decision_type(cycle_steps)
        
        # Extract step schemas
        step_schemas = []
        for step in cycle_steps:
            metadata = {}
            if step.meta_data:
                metadata = step.meta_data
            
            step_schemas.append(
                {
                    "id": step.id,
                    "session_id": step.session_id,
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "agent_id": step.agent_id,
                    "model": step.model,
                    "content": step.content,
                    "metadata": metadata,
                    "timestamp": step.timestamp
                }
            )
        
        return DecisionCycle(
            cycle_number=cycle_number,
            start_step=start_step,
            end_step=end_step,
            steps=step_schemas,
            latency_ms=latency_ms,
            decision_type=decision_type
        )
    
    def _determine_decision_type(self, cycle_steps: List[SessionStep]) -> Optional[str]:
        """Determine the type of decision being made in this cycle."""
        step_types = [step.step_type for step in cycle_steps]
        content = " ".join(step.content.lower() for step in cycle_steps)
        
        # Tool selection
        if "tool_call" in step_types:
            if any(word in content for word in ["search", "query", "lookup"]):
                return "information_retrieval"
            elif any(word in content for word in ["write", "create", "edit", "code"]):
                return "content_generation"
            elif any(word in content for word in ["calculate", "compute", "analyze"]):
                return "computation"
            elif any(word in content for word in ["call", "execute", "run"]):
                return "execution"
            return "tool_selection"
        
        # Goal/strategy change
        if "decision" in step_types:
            if any(word in content for word in ["goal", "objective", "target"]):
                return "goal_change"
            elif any(word in content for word in ["strategy", "approach", "method"]):
                return "strategy_selection"
            elif any(word in content for word in ["parameter", "setting", "config"]):
                return "parameter_tuning"
            return "general_decision"
        
        # Error recovery
        if "error" in step_types:
            return "error_recovery"
        
        return None
    
    async def get_state_snapshot(
        self,
        session: AgentSession,
        step_number: int
    ) -> StateSnapshot:
        """Reconstruct agent state at a specific step."""
        if not session.steps:
            raise ValueError("Session has no steps")
        
        # Get all steps up to and including the target step
        steps = sorted(session.steps, key=lambda s: s.step_number)
        target_steps = [s for s in steps if s.step_number <= step_number]
        
        if not target_steps:
            raise ValueError(f"No steps found at or before step {step_number}")
        
        # Reconstruct context window
        context_window = self._build_context_window(target_steps)
        
        # Extract conversation history
        conversation_history = self._extract_conversation(target_steps)
        
        # Extract tool outputs received so far
        tool_outputs_received = self._extract_tool_outputs(target_steps)
        
        # Infer active goals
        active_goals = self._infer_active_goals(target_steps)
        
        # Infer constraints
        constraints = self._infer_constraints(target_steps)
        
        # Simulate memory state
        memory_state = self._simulate_memory_state(target_steps)
        
        return StateSnapshot(
            step_number=step_number,
            context_window=context_window,
            conversation_history=conversation_history,
            tool_outputs_received=tool_outputs_received,
            active_goals=active_goals,
            memory_state=memory_state,
            constraints=constraints
        )
    
    def _build_context_window(self, steps: List[SessionStep]) -> str:
        """Build a simulated context window from steps."""
        context_parts = []
        
        # Add system prompt if present in first step
        if steps and steps[0].context_snapshot:
            # Try to extract system message
            snapshot = steps[0].context_snapshot
            if "system" in snapshot.lower() or "you are" in snapshot.lower():
                context_parts.append(snapshot)
        
        # Add recent thoughts and observations
        recent_steps = steps[-10:]  # Last 10 steps
        for step in recent_steps:
            if step.step_type in ["think", "observation", "decision"]:
                context_parts.append(f"[{step.step_type.upper()}] {step.content}")
        
        # Truncate if too large
        context = "\n\n".join(context_parts)
        if len(context) > settings.max_context_snapshot_size:
            context = context[:settings.max_context_snapshot_size]
        
        return context
    
    def _extract_conversation(self, steps: List[SessionStep]) -> List[str]:
        """Extract conversation-like exchanges from steps."""
        conversation = []
        
        for step in steps:
            if step.step_type in ["think", "observation", "decision"]:
                # Clean and truncate
                content = step.content.strip()
                if len(content) > 200:
                    content = content[:197] + "..."
                conversation.append(f"{step.step_type}: {content}")
        
        return conversation[-20:]  # Keep last 20 entries
    
    def _extract_tool_outputs(self, steps: List[SessionStep]) -> List[Dict[str, Any]]:
        """Extract tool outputs from tool_result steps."""
        outputs = []
        
        for step in steps:
            if step.step_type == "tool_result":
                output = {
                    "step": step.step_number,
                    "tool_name": step.meta_data.get("tool_name") if step.meta_data else "unknown",
                    "content": step.content[:500]  # Truncate
                }
                outputs.append(output)
        
        return outputs[-10:]  # Keep last 10 outputs
    
    def _infer_active_goals(self, steps: List[SessionStep]) -> List[str]:
        """Infer active goals from step content."""
        goals = []
        goal_keywords = ["goal", "objective", "task", "need to", "must", "should"]
        
        for step in steps:
            if step.step_type in ["think", "decision"]:
                content_lower = step.content.lower()
                for keyword in goal_keywords:
                    if keyword in content_lower:
                        # Extract sentence containing goal
                        sentences = re.split(r'[.!?]', step.content)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                goals.append(sentence.strip())
                                break
        
        # Deduplicate and return unique goals
        return list(dict.fromkeys(goals))[-5:]  # Keep last 5 goals
    
    def _infer_constraints(self, steps: List[SessionStep]) -> List[str]:
        """Infer constraints from step content."""
        constraints = []
        constraint_keywords = ["cannot", "must not", "should not", "avoid", "limit", "constraint"]
        
        for step in steps:
            content_lower = step.content.lower()
            for keyword in constraint_keywords:
                if keyword in content_lower:
                    sentences = re.split(r'[.!?]', step.content)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            constraints.append(sentence.strip())
                            break
        
        return list(dict.fromkeys(constraints))
    
    def _simulate_memory_state(self, steps: List[SessionStep]) -> Dict[str, Any]:
        """Simulate a simple memory/vector store state."""
        # Extract key information from steps
        memory = {
            "facts": [],
            "decisions": [],
            "errors": []
        }
        
        for step in steps:
            if step.step_type == "decision":
                memory["decisions"].append({
                    "step": step.step_number,
                    "content": step.content[:200]
                })
            elif step.step_type == "error":
                memory["errors"].append({
                    "step": step.step_number,
                    "content": step.content[:200]
                })
            elif step.step_type == "observation":
                # Extract potential facts
                if len(step.content) < 100 and any(word in step.content.lower() for word in ["is ", "are ", "has ", "have "]):
                    memory["facts"].append({
                        "step": step.step_number,
                        "content": step.content
                    })
        
        # Keep only recent entries
        for key in memory:
            memory[key] = memory[key][-20:]
        
        return memory
    
    def identify_patterns(self, session: AgentSession) -> Dict[str, Any]:
        """Identify patterns in session steps."""
        if not session.steps:
            return {}
        
        steps = sorted(session.steps, key=lambda s: s.step_number)
        
        patterns = {
            "loops": self._find_loops(steps),
            "redundant_tool_calls": self._find_redundant_tool_calls(steps),
            "decision_reversals": self._find_decision_reversals(steps),
            "error_patterns": self._analyze_error_patterns(steps),
            "efficiency_metrics": self._calculate_efficiency_metrics(steps)
        }
        
        return patterns
    
    def _find_loops(self, steps: List[SessionStep]) -> List[Dict[str, Any]]:
        """Find loops (repeated sequences) in steps."""
        loops = []
        
        # Look for repeated tool calls
        tool_calls = [(s.step_number, s.content) for s in steps if s.step_type == "tool_call"]
        
        for i in range(len(tool_calls) - 1):
            for j in range(i + 1, len(tool_calls)):
                if tool_calls[i][1] == tool_calls[j][1]:
                    loops.append({
                        "start_step": tool_calls[i][0],
                        "end_step": tool_calls[j][0],
                        "pattern": "repeated_tool_call",
                        "content": tool_calls[i][1][:200]
                    })
        
        return loops[:5]  # Return top 5 loops
    
    def _find_redundant_tool_calls(self, steps: List[SessionStep]) -> List[Dict[str, Any]]:
        """Find redundant tool calls (similar calls close together)."""
        redundancies = []
        
        tool_calls = [(s.step_number, s.content, s.meta_data) for s in steps if s.step_type == "tool_call"]
        
        for i in range(len(tool_calls) - 1):
            step1_num, content1, meta1 = tool_calls[i]
            step2_num, content2, meta2 = tool_calls[i + 1]
            
            # Check if calls are within 3 steps and similar
            if step2_num - step1_num <= 3:
                # Simple similarity check (could be enhanced)
                words1 = set(content1.lower().split())
                words2 = set(content2.lower().split())
                similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
                
                if similarity > 0.7:
                    redundancies.append({
                        "steps": [step1_num, step2_num],
                        "similarity": similarity,
                        "content_samples": [content1[:100], content2[:100]]
                    })
        
        return redundancies
    
    def _find_decision_reversals(self, steps: List[SessionStep]) -> List[Dict[str, Any]]:
        """Find decision reversals (contradictory decisions)."""
        reversals = []
        
        decisions = [(s.step_number, s.content) for s in steps if s.step_type == "decision"]
        decision_keywords = ["yes", "no", "true", "false", "accept", "reject", "approve", "deny"]
        
        for i in range(len(decisions) - 1):
            step1_num, content1 = decisions[i]
            step2_num, content2 = decisions[i + 1]
            
            if step2_num - step1_num <= 5:
                content1_lower = content1.lower()
                content2_lower = content2.lower()
                
                # Check for contradictory keywords
                pairs = [("yes", "no"), ("true", "false"), ("accept", "reject"), ("approve", "deny")]
                
                for pos, neg in pairs:
                    if (pos in content1_lower and neg in content2_lower) or \
                       (neg in content1_lower and pos in content2_lower):
                        reversals.append({
                            "steps": [step1_num, step2_num],
                            "contradiction": f"{pos}/{neg}",
                            "content": [content1[:150], content2[:150]]
                        })
        
        return reversals
    
    def _analyze_error_patterns(self, steps: List[SessionStep]) -> Dict[str, Any]:
        """Analyze error patterns and recovery attempts."""
        errors = [s for s in steps if s.step_type == "error"]
        
        if not errors:
            return {"total_errors": 0, "recovery_rate": 1.0}
        
        total_errors = len(errors)
        recovery_attempts = 0
        successful_recoveries = 0
        
        for i, error_step in enumerate(errors):
            # Look at next few steps after error
            next_steps = [s for s in steps if s.step_number > error_step.step_number]
            next_steps = next_steps[:5]  # Look at next 5 steps
            
            recovery_attempts += 1
            
            # Check if there's a successful tool call or decision after error
            for next_step in next_steps:
                if next_step.step_type in ["tool_result", "decision"]:
                    successful_recoveries += 1
                    break
        
        recovery_rate = successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0
        
        return {
            "total_errors": total_errors,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": recovery_rate
        }
    
    def _calculate_efficiency_metrics(self, steps: List[SessionStep]) -> Dict[str, Any]:
        """Calculate efficiency metrics for the session."""
        if not steps:
            return {}
        
        # Calculate step type distribution
        type_counts = {}
        for step in steps:
            type_counts[step.step_type] = type_counts.get(step.step_type, 0) + 1
        
        # Calculate average tokens per step (if metadata available)
        total_tokens = 0
        steps_with_tokens = 0
        
        for step in steps:
            if step.meta_data and "tokens_used" in step.meta_data:
                total_tokens += step.meta_data["tokens_used"]
                steps_with_tokens += 1
        
        avg_tokens_per_step = total_tokens / steps_with_tokens if steps_with_tokens > 0 else None
        
        # Calculate decision density
        total_decisions = type_counts.get("decision", 0)
        decision_density = total_decisions / len(steps) if steps else 0
        
        # Calculate tool call success rate
        tool_calls = [s for s in steps if s.step_type == "tool_call"]
        tool_results = [s for s in steps if s.step_type == "tool_result"]
        
        tool_success_rate = len(tool_results) / len(tool_calls) if tool_calls else 1.0
        
        return {
            "total_steps": len(steps),
            "step_type_distribution": type_counts,
            "avg_tokens_per_step": avg_tokens_per_step,
            "decision_density": decision_density,
            "tool_success_rate": tool_success_rate,
            "step_efficiency": tool_success_rate * decision_density if decision_density > 0 else 0
        }