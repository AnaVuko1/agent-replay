from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from collections import defaultdict

from app.models import SessionStep
from app.schemas import StateSnapshot


class SnapshotEngine:
    """Captures agent state at any point in the timeline."""
    
    def __init__(self):
        self.context_window_size = 10  # Number of recent steps to include
        self.max_history_items = 20
        self.max_tool_outputs = 10
    
    async def capture_state(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> StateSnapshot:
        """Capture agent state at a specific step number."""
        if not steps:
            raise ValueError("No steps provided")
        
        # Filter steps up to target step
        relevant_steps = [s for s in steps if s.step_number <= target_step_number]
        
        if not relevant_steps:
            raise ValueError(f"No steps found at or before step {target_step_number}")
        
        # Get the target step
        target_step = next((s for s in relevant_steps if s.step_number == target_step_number), None)
        if not target_step:
            # Use last step before target
            target_step = relevant_steps[-1]
            target_step_number = target_step.step_number
        
        # Build context window
        context_window = self._build_context_window(relevant_steps, target_step_number)
        
        # Extract conversation history
        conversation_history = self._extract_conversation_history(relevant_steps)
        
        # Extract tool outputs
        tool_outputs_received = self._extract_tool_outputs(relevant_steps)
        
        # Infer active goals
        active_goals = self._infer_active_goals(relevant_steps, target_step_number)
        
        # Infer constraints
        constraints = self._infer_constraints(relevant_steps)
        
        # Build memory state
        memory_state = self._build_memory_state(relevant_steps, target_step_number)
        
        # Build task state
        task_state = self._build_task_state(relevant_steps, target_step_number)
        
        return StateSnapshot(
            step_number=target_step_number,
            context_window=context_window,
            conversation_history=conversation_history,
            tool_outputs_received=tool_outputs_received,
            active_goals=active_goals,
            memory_state=memory_state,
            constraints=constraints,
            task_state=task_state
        )
    
    def _build_context_window(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> str:
        """Build the context window as the agent would see it."""
        context_parts = []
        
        # Get recent steps (up to context_window_size)
        recent_steps = [s for s in steps if s.step_number <= target_step_number]
        recent_steps = recent_steps[-self.context_window_size:]
        
        # Build context from recent steps
        for step in recent_steps:
            if step.step_type in ["think", "observation", "decision", "tool_result"]:
                # Format based on step type
                if step.step_type == "think":
                    prefix = "Thought: "
                elif step.step_type == "observation":
                    prefix = "Observation: "
                elif step.step_type == "decision":
                    prefix = "Decision: "
                elif step.step_type == "tool_result":
                    tool_name = step.meta_data.get("tool_name", "tool") if step.meta_data else "tool"
                    prefix = f"{tool_name.capitalize()} result: "
                else:
                    prefix = ""
                
                # Add to context
                content = step.content.strip()
                if len(content) > 300:
                    content = content[:297] + "..."
                
                context_parts.append(f"{prefix}{content}")
        
        # Add system context if available in early steps
        if steps and steps[0].context_snapshot:
            system_context = steps[0].context_snapshot
            if len(system_context) < 500:  # Don't add huge system prompts
                context_parts.insert(0, f"System: {system_context}")
        
        return "\n\n".join(context_parts)
    
    def _extract_conversation_history(self, steps: List[SessionStep]) -> List[str]:
        """Extract conversation-like history from steps."""
        conversation = []
        
        for step in steps:
            if step.step_type in ["think", "observation", "decision"]:
                # Format conversation entry
                timestamp = step.timestamp.strftime("%H:%M:%S") if step.timestamp else "??:??:??"
                entry = f"[{timestamp}] {step.step_type.upper()}: {step.content[:200]}"
                
                if len(step.content) > 200:
                    entry += "..."
                
                conversation.append(entry)
        
        # Keep only recent history
        return conversation[-self.max_history_items:]
    
    def _extract_tool_outputs(self, steps: List[SessionStep]) -> List[Dict[str, Any]]:
        """Extract tool outputs received so far."""
        outputs = []
        
        for step in steps:
            if step.step_type == "tool_result":
                output = {
                    "step": step.step_number,
                    "tool_name": step.meta_data.get("tool_name") if step.meta_data else "unknown",
                    "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                    "content_preview": step.content[:200],
                    "success": "error" not in step.content.lower() and "failed" not in step.content.lower()
                }
                
                # Add full metadata if available
                if step.meta_data:
                    output["metadata"] = {
                        k: v for k, v in step.meta_data.items()
                        if k not in ["tool_name"] and isinstance(v, (str, int, float, bool, list, dict))
                    }
                
                outputs.append(output)
        
        # Return most recent outputs
        return outputs[-self.max_tool_outputs:]
    
    def _infer_active_goals(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> List[str]:
        """Infer active goals from step content."""
        goals = []
        goal_keywords = [
            "goal", "objective", "task", "need to", "must", "should", 
            "purpose", "aim", "target", "achieve", "solve", "fix"
        ]
        
        # Look for goal statements in recent steps
        recent_steps = [s for s in steps if s.step_number <= target_step_number]
        recent_steps = recent_steps[-10:]  # Last 10 steps
        
        for step in recent_steps:
            if step.step_type in ["think", "decision"]:
                content_lower = step.content.lower()
                
                # Check for goal-related keywords
                for keyword in goal_keywords:
                    if keyword in content_lower:
                        # Extract the sentence containing the goal
                        sentences = re.split(r'[.!?]', step.content)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                clean_sentence = sentence.strip()
                                if len(clean_sentence) > 10:  # Avoid very short sentences
                                    goals.append(clean_sentence)
                                break
        
        # Also look for explicit goal statements in context snapshots
        for step in recent_steps:
            if step.context_snapshot:
                snapshot_lower = step.context_snapshot.lower()
                if "goal:" in snapshot_lower or "objective:" in snapshot_lower:
                    # Extract goal from snapshot
                    lines = step.context_snapshot.split('\n')
                    for line in lines:
                        if "goal:" in line.lower() or "objective:" in line.lower():
                            goals.append(line.strip())
        
        # Deduplicate and return
        unique_goals = []
        seen = set()
        for goal in goals:
            # Simple deduplication by first 50 chars
            key = goal[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_goals.append(goal)
        
        return unique_goals[-5:]  # Keep up to 5 most recent goals
    
    def _infer_constraints(self, steps: List[SessionStep]) -> List[str]:
        """Infer constraints from step content."""
        constraints = []
        constraint_keywords = [
            "cannot", "must not", "should not", "avoid", "limit", "constraint",
            "restriction", "prohibited", "forbidden", "don't", "do not"
        ]
        
        for step in steps:
            if step.step_type in ["think", "decision", "observation"]:
                content_lower = step.content.lower()
                
                for keyword in constraint_keywords:
                    if keyword in content_lower:
                        # Extract constraint sentence
                        sentences = re.split(r'[.!?]', step.content)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                clean_sentence = sentence.strip()
                                if len(clean_sentence) > 10:
                                    constraints.append(clean_sentence)
                                break
        
        # Also look in metadata
        for step in steps:
            if step.meta_data and "constraints" in step.meta_data:
                if isinstance(step.meta_data["constraints"], list):
                    constraints.extend(step.meta_data["constraints"])
        
        # Deduplicate
        unique_constraints = []
        seen = set()
        for constraint in constraints:
            key = constraint[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_constraints.append(constraint)
        
        return unique_constraints[-10:]  # Keep up to 10 constraints
    
    def _build_memory_state(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> Dict[str, Any]:
        """Build simulated memory/vector store state."""
        memory = {
            "facts": [],
            "decisions": [],
            "errors": [],
            "tool_knowledge": [],
            "patterns": []
        }
        
        relevant_steps = [s for s in steps if s.step_number <= target_step_number]
        
        for step in relevant_steps:
            # Extract facts from observations
            if step.step_type == "observation":
                fact = self._extract_fact_from_observation(step.content)
                if fact:
                    memory["facts"].append({
                        "step": step.step_number,
                        "fact": fact,
                        "confidence": self._estimate_confidence(step.content)
                    })
            
            # Record decisions
            elif step.step_type == "decision":
                memory["decisions"].append({
                    "step": step.step_number,
                    "content": step.content[:200],
                    "reasoning": self._extract_reasoning(step.content),
                    "timestamp": step.timestamp.isoformat() if step.timestamp else None
                })
            
            # Record errors
            elif step.step_type == "error":
                memory["errors"].append({
                    "step": step.step_number,
                    "message": step.content[:200],
                    "recovered": self._check_error_recovery(relevant_steps, step.step_number),
                    "timestamp": step.timestamp.isoformat() if step.timestamp else None
                })
            
            # Extract tool knowledge
            elif step.step_type == "tool_result":
                tool_knowledge = self._extract_tool_knowledge(step)
                if tool_knowledge:
                    memory["tool_knowledge"].append(tool_knowledge)
        
        # Identify patterns
        memory["patterns"] = self._identify_patterns(relevant_steps)
        
        # Limit memory size
        for key in memory:
            if isinstance(memory[key], list):
                memory[key] = memory[key][-20:]  # Keep last 20 items
        
        return memory
    
    def _build_task_state(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> Dict[str, Any]:
        """Build task/progress state."""
        task_state = {
            "progress": self._estimate_progress(steps, target_step_number),
            "current_subtask": self._identify_current_subtask(steps, target_step_number),
            "completed_subtasks": self._identify_completed_subtasks(steps, target_step_number),
            "pending_subtasks": self._identify_pending_subtasks(steps, target_step_number),
            "blockers": self._identify_blockers(steps, target_step_number),
            "next_actions": self._predict_next_actions(steps, target_step_number)
        }
        
        return task_state
    
    def _extract_fact_from_observation(self, content: str) -> Optional[str]:
        """Extract factual statement from observation."""
        # Look for statements of fact
        fact_patterns = [
            r"([A-Z][^.!?]*(is|are|was|were|has|have|contains|shows|indicates)[^.!?]*[.!?])",
            r"([A-Z][^.!?]*(found|discovered|learned|determined)[^.!?]*[.!?])",
            r"([A-Z][^.!?]*(\d+|[0-9.]+%|true|false|yes|no)[^.!?]*[.!?])"
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    sentence = match[0]
                else:
                    sentence = match
                
                # Clean up
                sentence = sentence.strip()
                if len(sentence) > 10 and len(sentence) < 200:
                    return sentence
        
        return None
    
    def _estimate_confidence(self, content: str) -> float:
        """Estimate confidence level from content."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["certain", "definitely", "confirmed", "verified"]):
            return 0.9
        elif any(word in content_lower for word in ["probably", "likely", "seems", "appears"]):
            return 0.7
        elif any(word in content_lower for word in ["maybe", "perhaps", "possibly", "might"]):
            return 0.5
        elif any(word in content_lower for word in ["uncertain", "unsure", "unknown", "unclear"]):
            return 0.3
        else:
            return 0.6  # Default
    
    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning from decision content."""
        # Look for reasoning indicators
        reasoning_patterns = [
            r"because[^.!?]*[.!?]",
            r"since[^.!?]*[.!?]",
            r"therefore[^.!?]*[.!?]",
            r"so[^.!?]*[.!?]",
            r"reason:[^.!?]*[.!?]",
            r"rationale:[^.!?]*[.!?]"
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                reasoning = match.group(0).strip()
                if len(reasoning) > 10:
                    return reasoning
        
        return None
    
    def _check_error_recovery(self, steps: List[SessionStep], error_step_number: int) -> bool:
        """Check if an error was recovered from."""
        # Look at steps after error
        steps_after = [s for s in steps if s.step_number > error_step_number]
        steps_after = steps_after[:5]  # Look at next 5 steps
        
        for step in steps_after:
            if step.step_type in ["tool_result", "decision"]:
                # Check if it shows recovery
                if "success" in step.content.lower() or "fixed" in step.content.lower():
                    return True
        
        return False
    
    def _extract_tool_knowledge(self, step: SessionStep) -> Optional[Dict[str, Any]]:
        """Extract knowledge about tool usage."""
        if step.step_type != "tool_result":
            return None
        
        knowledge = {
            "tool_name": step.meta_data.get("tool_name") if step.meta_data else "unknown",
            "step": step.step_number,
            "timestamp": step.timestamp.isoformat() if step.timestamp else None
        }
        
        # Extract what was learned
        content_lower = step.content.lower()
        
        if "error" in content_lower or "failed" in content_lower:
            knowledge["type"] = "error"
            knowledge["lesson"] = f"Tool error: {step.content[:100]}"
        elif any(word in content_lower for word in ["data", "result", "output", "found"]):
            knowledge["type"] = "data"
            knowledge["data_preview"] = step.content[:150]
        elif any(word in content_lower for word in ["success", "completed", "finished"]):
            knowledge["type"] = "success"
            knowledge["outcome"] = "Tool executed successfully"
        
        return knowledge
    
    def _identify_patterns(self, steps: List[SessionStep]) -> List[Dict[str, Any]]:
        """Identify patterns in agent behavior."""
        patterns = []
        
        # Check for repeated tool calls
        tool_calls = [(s.step_number, s.content, s.meta_data) 
                     for s in steps if s.step_type == "tool_call"]
        
        tool_names = {}
        for step_num, content, metadata in tool_calls:
            tool_name = metadata.get("tool_name") if metadata else "unknown"
            if tool_name not in tool_names:
                tool_names[tool_name] = []
            tool_names[tool_name].append(step_num)
        
        for tool_name, calls in tool_names.items():
            if len(calls) > 2:
                patterns.append({
                    "type": "repeated_tool_use",
                    "tool": tool_name,
                    "call_count": len(calls),
                    "steps": calls[:5]  # First 5 calls
                })
        
        # Check for decision chains
        decisions = [s for s in steps if s.step_type == "decision"]
        if len(decisions) >= 3:
            # Check if decisions are related
            decision_content = [d.content.lower() for d in decisions[-3:]]
            common_words = set()
            for content in decision_content:
                words = set(content.split()[:10])  # First 10 words
                if not common_words:
                    common_words = words
                else:
                    common_words = common_words.intersection(words)
            
            if len(common_words) >= 2:
                patterns.append({
                    "type": "decision_chain",
                    "steps": [d.step_number for d in decisions[-3:]],
                    "common_themes": list(common_words)[:5]
                })
        
        # Check for error-recovery patterns
        errors = [s for s in steps if s.step_type == "error"]
        for error in errors:
            # Look for recovery attempts
            steps_after = [s for s in steps if s.step_number > error.step_number]
            steps_after = steps_after[:3]
            
            recovery_attempts = []
            for step in steps_after:
                if step.step_type in ["think", "tool_call"]:
                    if "retry" in step.content.lower() or "fix" in step.content.lower():
                        recovery_attempts.append(step.step_number)
            
            if recovery_attempts:
                patterns.append({
                    "type": "error_recovery",
                    "error_step": error.step_number,
                    "recovery_attempts": recovery_attempts
                })
        
        return patterns[:5]  # Return top 5 patterns
    
    def _estimate_progress(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> Dict[str, Any]:
        """Estimate task progress."""
        if not steps:
            return {"percentage": 0, "stage": "not_started"}
        
        # Simple heuristic based on step types and patterns
        total_steps = len([s for s in steps if s.step_number <= target_step_number])
        
        # Count completion indicators
        completion_indicators = 0
        for step in steps:
            if step.step_number > target_step_number:
                continue
            
            if step.step_type == "decision":
                if "complete" in step.content.lower() or "finished" in step.content.lower():
                    completion_indicators += 1
            
            if step.meta_data and step.meta_data.get("status") == "completed":
                completion_indicators += 1
        
        # Estimate percentage (simplistic)
        if total_steps < 5:
            percentage = min(30, total_steps * 10)
            stage = "early"
        elif total_steps < 15:
            percentage = min(70, 30 + (total_steps - 5) * 5)
            stage = "middle"
        else:
            percentage = min(100, 70 + (total_steps - 15) * 2)
            stage = "late"
        
        # Adjust based on completion indicators
        if completion_indicators > 0:
            percentage = min(100, percentage + (completion_indicators * 10))
            if completion_indicators >= 2:
                stage = "completing"
        
        return {
            "percentage": percentage,
            "stage": stage,
            "total_steps_so_far": total_steps,
            "completion_indicators": completion_indicators
        }
    
    def _identify_current_subtask(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> Optional[Dict[str, Any]]:
        """Identify current subtask being worked on."""
        # Look at recent steps
        recent_steps = [s for s in steps if s.step_number <= target_step_number]
        recent_steps = recent_steps[-5:]  # Last 5 steps
        
        for step in reversed(recent_steps):
            if step.step_type in ["think", "decision"]:
                # Look for task-related language
                content_lower = step.content.lower()
                
                task_keywords = ["working on", "currently", "now", "next", "task", "subtask"]
                for keyword in task_keywords:
                    if keyword in content_lower:
                        # Extract task description
                        sentences = re.split(r'[.!?]', step.content)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                return {
                                    "description": sentence.strip(),
                                    "step": step.step_number,
                                    "step_type": step.step_type
                                }
        
        return None
    
    def _identify_completed_subtasks(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> List[Dict[str, Any]]:
        """Identify completed subtasks."""
        completed = []
        
        for step in steps:
            if step.step_number > target_step_number:
                continue
            
            if step.step_type == "decision":
                if "completed" in step.content.lower() or "finished" in step.content.lower():
                    completed.append({
                        "step": step.step_number,
                        "description": step.content[:100],
                        "timestamp": step.timestamp.isoformat() if step.timestamp else None
                    })
            
            if step.meta_data and step.meta_data.get("status") == "completed":
                completed.append({
                    "step": step.step_number,
                    "description": step.meta_data.get("task_description", "Completed task"),
                    "timestamp": step.timestamp.isoformat() if step.timestamp else None
                })
        
        return completed[-5:]  # Last 5 completed subtasks
    
    def _identify_pending_subtasks(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> List[Dict[str, Any]]:
        """Identify pending subtasks."""
        pending = []
        
        # Look for future references in recent steps
        recent_steps = [s for s in steps if s.step_number <= target_step_number]
        recent_steps = recent_steps[-10:]
        
        for step in recent_steps:
            if step.step_type in ["think", "decision"]:
                content_lower = step.content.lower()
                
                future_keywords = ["need to", "should", "must", "next", "then", "after", "still need"]
                for keyword in future_keywords:
                    if keyword in content_lower:
                        # Extract future task
                        sentences = re.split(r'[.!?]', step.content)
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                pending.append({
                                    "step": step.step_number,
                                    "description": sentence.strip(),
                                    "urgency": self._estimate_urgency(sentence)
                                })
                                break
        
        # Deduplicate
        unique_pending = []
        seen = set()
        for task in pending:
            key = task["description"][:50].lower()
            if key not in seen:
                seen.add(key)
                unique_pending.append(task)
        
        return unique_pending[:5]  # Top 5 pending tasks
    
    def _identify_blockers(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> List[Dict[str, Any]]:
        """Identify current blockers."""
        blockers = []
        
        # Look for errors and issues
        recent_steps = [s for s in steps if s.step_number <= target_step_number]
        recent_steps = recent_steps[-10:]
        
        for step in recent_steps:
            if step.step_type == "error":
                blockers.append({
                    "type": "error",
                    "step": step.step_number,
                    "description": step.content[:150],
                    "severity": "high"
                })
            
            elif step.step_type in ["think", "observation"]:
                content_lower = step.content.lower()
                
                blocker_keywords = ["blocked", "stuck", "cannot", "unable", "problem", "issue", "difficulty"]
                for keyword in blocker_keywords:
                    if keyword in content_lower:
                        blockers.append({
                            "type": "blocker",
                            "step": step.step_number,
                            "description": step.content[:150],
                            "severity": "medium"
                        })
                        break
        
        return blockers[:3]  # Top 3 blockers
    
    def _predict_next_actions(
        self,
        steps: List[SessionStep],
        target_step_number: int
    ) -> List[Dict[str, Any]]:
        """Predict likely next actions."""
        next_actions = []
        
        # Look at recent patterns
        recent_steps = [s for s in steps if s.step_number <= target_step_number]
        if not recent_steps:
            return next_actions
        
        last_step = recent_steps[-1]
        
        # Predict based on last step type
        if last_step.step_type == "think":
            # Likely to make a decision or call a tool
            next_actions.append({
                "type": "decision_or_tool_call",
                "confidence": 0.7,
                "reason": "Thinking usually precedes decision or action"
            })
        
        elif last_step.step_type == "tool_call":
            # Expect tool result next
            next_actions.append({
                "type": "tool_result",
                "confidence": 0.9,
                "reason": "Tool call should be followed by result"
            })
        
        elif last_step.step_type == "error":
            # Likely recovery attempt
            next_actions.append({
                "type": "recovery_attempt",
                "confidence": 0.8,
                "reason": "Errors usually trigger recovery attempts"
            })
        
        elif last_step.step_type == "tool_result":
            # Likely observation or next action
            next_actions.append({
                "type": "observation_or_next_action",
                "confidence": 0.6,
                "reason": "Tool results are typically observed and acted upon"
            })
        
        # Also consider patterns
        if len(recent_steps) >= 3:
            last_three_types = [s.step_type for s in recent_steps[-3:]]
            
            # Check for patterns
            if last_three_types == ["think", "tool_call", "tool_result"]:
                next_actions.append({
                    "type": "observation",
                    "confidence": 0.8,
                    "reason": "Pattern suggests observation next"
                })
            
            elif "error" in last_three_types:
                next_actions.append({
                    "type": "think",  # About recovery
                    "confidence": 0.7,
                    "reason": "Error recovery pattern"
                })
        
        return next_actions
    
    def _estimate_urgency(self, sentence: str) -> str:
        """Estimate urgency of a task."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ["urgent", "immediately", "now", "asap", "critical"]):
            return "high"
        elif any(word in sentence_lower for word in ["soon", "next", "then", "after"]):
            return "medium"
        else:
            return "low"