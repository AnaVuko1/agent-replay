from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timezone
import re

from app.schemas import SessionStepCreate, Metadata


class TraceIngestionAdapter:
    """Adapter for ingesting traces from various AI platforms."""
    
    @staticmethod
    def from_openai(openai_trace: Dict[str, Any]) -> List[SessionStepCreate]:
        """Convert OpenAI trace to Agent Replay steps."""
        steps = []
        step_number = 1
        
        # Extract messages
        messages = openai_trace.get("messages", [])
        for msg in messages:
            step_type = TraceIngestionAdapter._detect_openai_step_type(msg)
            
            step = SessionStepCreate(
                step_number=step_number,
                step_type=step_type,
                agent_id="openai-agent",
                model=openai_trace.get("model", "gpt-unknown"),
                content=TraceIngestionAdapter._extract_openai_content(msg),
                metadata=TraceIngestionAdapter._extract_openai_metadata(openai_trace, msg),
                context_snapshot=TraceIngestionAdapter._build_openai_context(messages, msg)
            )
            
            steps.append(step)
            step_number += 1
        
        # Extract tool calls if present
        if "tool_calls" in openai_trace:
            for tool_call in openai_trace["tool_calls"]:
                step = SessionStepCreate(
                    step_number=step_number,
                    step_type="tool_call",
                    agent_id="openai-agent",
                    model=openai_trace.get("model", "gpt-unknown"),
                    content=json.dumps(tool_call, indent=2),
                    metadata={
                        "tool_name": tool_call.get("function", {}).get("name"),
                        "tool_args": tool_call.get("function", {}).get("arguments"),
                        "call_id": tool_call.get("id")
                    }
                )
                
                steps.append(step)
                step_number += 1
        
        return steps
    
    @staticmethod
    def from_anthropic(anthropic_trace: Dict[str, Any]) -> List[SessionStepCreate]:
        """Convert Anthropic trace to Agent Replay steps."""
        steps = []
        step_number = 1
        
        # Extract messages
        messages = anthropic_trace.get("messages", [])
        for msg in messages:
            step_type = TraceIngestionAdapter._detect_anthropic_step_type(msg)
            
            step = SessionStepCreate(
                step_number=step_number,
                step_type=step_type,
                agent_id="anthropic-agent",
                model=anthropic_trace.get("model", "claude-unknown"),
                content=TraceIngestionAdapter._extract_anthropic_content(msg),
                metadata=TraceIngestionAdapter._extract_anthropic_metadata(anthropic_trace, msg),
                context_snapshot=TraceIngestionAdapter._build_anthropic_context(messages, msg)
            )
            
            steps.append(step)
            step_number += 1
        
        # Extract tool uses if present
        if "tool_uses" in anthropic_trace:
            for tool_use in anthropic_trace["tool_uses"]:
                step = SessionStepCreate(
                    step_number=step_number,
                    step_type="tool_call",
                    agent_id="anthropic-agent",
                    model=anthropic_trace.get("model", "claude-unknown"),
                    content=json.dumps(tool_use, indent=2),
                    metadata={
                        "tool_name": tool_use.get("name"),
                        "tool_args": tool_use.get("input"),
                        "use_id": tool_use.get("id")
                    }
                )
                
                steps.append(step)
                step_number += 1
        
        return steps
    
    @staticmethod
    def from_langsmith(langsmith_trace: Dict[str, Any]) -> List[SessionStepCreate]:
        """Convert LangSmith trace to Agent Replay steps."""
        steps = []
        
        # LangSmith traces can be complex - extract runs
        runs = langsmith_trace.get("runs", [])
        
        for run in runs:
            step_type = TraceIngestionAdapter._detect_langsmith_step_type(run)
            
            # Create step for each run
            step = SessionStepCreate(
                step_number=run.get("execution_order", 1),
                step_type=step_type,
                agent_id=run.get("name", "langsmith-agent"),
                model=run.get("extra", {}).get("model", "unknown"),
                content=TraceIngestionAdapter._extract_langsmith_content(run),
                metadata=TraceIngestionAdapter._extract_langsmith_metadata(run),
                context_snapshot=TraceIngestionAdapter._extract_langsmith_context(run)
            )
            
            steps.append(step)
        
        # Sort by step number
        steps.sort(key=lambda x: x.step_number)
        
        # Re-number sequentially
        for i, step in enumerate(steps, 1):
            step.step_number = i
        
        return steps
    
    @staticmethod
    def from_generic(trace: Dict[str, Any]) -> List[SessionStepCreate]:
        """Convert generic trace format to Agent Replay steps."""
        steps = []
        
        # Try to extract steps from common patterns
        if "steps" in trace:
            # Direct steps array
            for i, step_data in enumerate(trace["steps"], 1):
                step = SessionStepCreate(
                    step_number=i,
                    step_type=step_data.get("type", "unknown"),
                    agent_id=step_data.get("agent_id", "generic-agent"),
                    model=step_data.get("model", "unknown"),
                    content=str(step_data.get("content", "")),
                    metadata=step_data.get("metadata", {}),
                    context_snapshot=step_data.get("context", "")
                )
                steps.append(step)
        
        elif "events" in trace:
            # Events array
            for i, event in enumerate(trace["events"], 1):
                step_type = TraceIngestionAdapter._detect_generic_step_type(event)
                
                step = SessionStepCreate(
                    step_number=i,
                    step_type=step_type,
                    agent_id=event.get("agent", "generic-agent"),
                    model=event.get("model", "unknown"),
                    content=str(event.get("data", event.get("message", ""))),
                    metadata=event.get("metadata", {}),
                    context_snapshot=event.get("context", "")
                )
                steps.append(step)
        
        elif "messages" in trace:
            # Chat messages
            for i, msg in enumerate(trace["messages"], 1):
                step_type = TraceIngestionAdapter._detect_message_step_type(msg)
                
                step = SessionStepCreate(
                    step_number=i,
                    step_type=step_type,
                    agent_id=msg.get("role", "user"),
                    model=trace.get("model", "unknown"),
                    content=msg.get("content", ""),
                    metadata={
                        "role": msg.get("role"),
                        "tokens": msg.get("token_count")
                    },
                    context_snapshot=json.dumps(trace["messages"][:i], indent=2)
                )
                steps.append(step)
        
        else:
            # Single step trace
            step = SessionStepCreate(
                step_number=1,
                step_type="decision",
                agent_id=trace.get("agent_id", "single-step-agent"),
                model=trace.get("model", "unknown"),
                content=json.dumps(trace, indent=2),
                metadata=trace.get("metadata", {}),
                context_snapshot=trace.get("context", "")
            )
            steps.append(step)
        
        return steps
    
    @staticmethod
    def _detect_openai_step_type(msg: Dict[str, Any]) -> str:
        """Detect step type from OpenAI message."""
        role = msg.get("role", "").lower()
        content = str(msg.get("content", "")).lower()
        
        if role == "assistant":
            if "tool_call" in str(msg):
                return "tool_call"
            elif any(word in content for word in ["think", "reason", "analyze"]):
                return "think"
            elif any(word in content for word in ["decide", "choose", "select"]):
                return "decision"
            elif any(word in content for word in ["observe", "see", "notice"]):
                return "observation"
            else:
                return "decision"  # Default for assistant messages
        
        elif role == "tool":
            return "tool_result"
        
        elif role == "user":
            return "observation"
        
        elif role == "system":
            return "think"  # System prompts as initial thought
        
        else:
            return "unknown"
    
    @staticmethod
    def _extract_openai_content(msg: Dict[str, Any]) -> str:
        """Extract content from OpenAI message."""
        content = msg.get("content")
        
        if content:
            if isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            text_parts.append("[Image input]")
                return "\n".join(text_parts)
            else:
                return str(content)
        
        # Check for tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            return json.dumps(tool_calls, indent=2)
        
        return ""
    
    @staticmethod
    def _extract_openai_metadata(trace: Dict[str, Any], msg: Dict[str, Any]) -> Metadata:
        """Extract metadata from OpenAI trace."""
        metadata = {}
        
        # Extract usage if available
        usage = trace.get("usage", {})
        if usage:
            metadata["tokens_used"] = usage.get("total_tokens")
            metadata["prompt_tokens"] = usage.get("prompt_tokens")
            metadata["completion_tokens"] = usage.get("completion_tokens")
        
        # Extract timing if available
        if "created" in trace:
            metadata["timestamp"] = datetime.fromtimestamp(trace["created"], tz=timezone.utc).isoformat()
        
        # Extract model info
        metadata["model"] = trace.get("model")
        
        # Extract tool call info
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            metadata["tool_call_count"] = len(tool_calls)
            if tool_calls:
                first_tool = tool_calls[0]
                metadata["tool_name"] = first_tool.get("function", {}).get("name")
        
        return Metadata(**metadata)
    
    @staticmethod
    def _build_openai_context(messages: List[Dict[str, Any]], current_msg: Dict[str, Any]) -> str:
        """Build context window from OpenAI messages."""
        # Get messages up to current message
        context_messages = []
        for msg in messages:
            if msg == current_msg:
                break
            context_messages.append(msg)
        
        # Format context
        context_parts = []
        for msg in context_messages[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = TraceIngestionAdapter._extract_openai_content(msg)
            
            if content:
                context_parts.append(f"{role.upper()}: {content[:200]}")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def _detect_anthropic_step_type(msg: Dict[str, Any]) -> str:
        """Detect step type from Anthropic message."""
        role = msg.get("role", "").lower()
        content = str(msg.get("content", "")).lower()
        
        if role == "assistant":
            if "tool_use" in str(msg):
                return "tool_call"
            elif any(word in content for word in ["think", "reason", "analyze"]):
                return "think"
            elif any(word in content for word in ["decide", "choose", "select"]):
                return "decision"
            elif any(word in content for word in ["observe", "see", "notice"]):
                return "observation"
            else:
                return "decision"
        
        elif role == "user":
            return "observation"
        
        else:
            return "unknown"
    
    @staticmethod
    def _extract_anthropic_content(msg: Dict[str, Any]) -> str:
        """Extract content from Anthropic message."""
        content = msg.get("content")
        
        if isinstance(content, list):
            # Handle complex content
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "tool_use":
                        tool_use = item
                        text_parts.append(
                            f"Tool use: {tool_use.get('name')}\n"
                            f"Input: {json.dumps(tool_use.get('input'), indent=2)}"
                        )
            return "\n".join(text_parts)
        
        elif content:
            return str(content)
        
        return ""
    
    @staticmethod
    def _extract_anthropic_metadata(trace: Dict[str, Any], msg: Dict[str, Any]) -> Metadata:
        """Extract metadata from Anthropic trace."""
        metadata = {}
        
        # Extract usage if available
        usage = trace.get("usage", {})
        if usage:
            metadata["tokens_used"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            metadata["input_tokens"] = usage.get("input_tokens")
            metadata["output_tokens"] = usage.get("output_tokens")
        
        # Extract model info
        metadata["model"] = trace.get("model")
        
        # Check for tool use
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    metadata["tool_name"] = item.get("name")
                    metadata["tool_use_id"] = item.get("id")
                    break
        
        return Metadata(**metadata)
    
    @staticmethod
    def _build_anthropic_context(messages: List[Dict[str, Any]], current_msg: Dict[str, Any]) -> str:
        """Build context window from Anthropic messages."""
        context_messages = []
        for msg in messages:
            if msg == current_msg:
                break
            context_messages.append(msg)
        
        context_parts = []
        for msg in context_messages[-5:]:
            role = msg.get("role", "unknown")
            content = TraceIngestionAdapter._extract_anthropic_content(msg)
            
            if content:
                context_parts.append(f"{role.upper()}: {content[:200]}")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def _detect_langsmith_step_type(run: Dict[str, Any]) -> str:
        """Detect step type from LangSmith run."""
        run_type = run.get("run_type", "").lower()
        name = run.get("name", "").lower()
        
        # Map common LangSmith run types to our step types
        type_map = {
            "chain": "think",
            "llm": "decision",
            "retriever": "tool_call",
            "tool": "tool_call",
            "agent": "decision",
            "chat": "think",
            "prompt": "think",
            "embedding": "tool_call",
            "parser": "tool_result",
            "reranker": "tool_call"
        }
        
        if run_type in type_map:
            return type_map[run_type]
        
        # Try to infer from name
        if any(word in name for word in ["think", "reason", "analyze"]):
            return "think"
        elif any(word in name for word in ["decide", "choose", "select"]):
            return "decision"
        elif any(word in name for word in ["tool", "call", "execute"]):
            return "tool_call"
        elif any(word in name for word in ["result", "output", "response"]):
            return "tool_result"
        elif any(word in name for word in ["observe", "see", "notice"]):
            return "observation"
        elif any(word in name for word in ["error", "exception", "fail"]):
            return "error"
        
        return "unknown"
    
    @staticmethod
    def _extract_langsmith_content(run: Dict[str, Any]) -> str:
        """Extract content from LangSmith run."""
        # Try various content fields
        content_fields = ["outputs", "output", "inputs", "input", "serialized", "extra"]
        
        for field in content_fields:
            if field in run:
                value = run[field]
                if isinstance(value, dict) or isinstance(value, list):
                    return json.dumps(value, indent=2)
                elif value:
                    return str(value)
        
        # Fallback to run name and type
        return f"{run.get('run_type', 'unknown')}: {run.get('name', 'unnamed')}"
    
    @staticmethod
    def _extract_langsmith_metadata(run: Dict[str, Any]) -> Metadata:
        """Extract metadata from LangSmith run."""
        metadata = {}
        
        # Extract timing
        if "start_time" in run:
            metadata["start_time"] = run["start_time"]
        if "end_time" in run:
            metadata["end_time"] = run["end_time"]
            if "start_time" in run:
                try:
                    start = datetime.fromisoformat(run["start_time"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(run["end_time"].replace("Z", "+00:00"))
                    metadata["latency_ms"] = (end - start).total_seconds() * 1000
                except (ValueError, TypeError):
                    pass
        
        # Extract tokens if available
        extra = run.get("extra", {})
        if "token_usage" in extra:
            token_usage = extra["token_usage"]
            metadata["tokens_used"] = token_usage.get("total_tokens")
            metadata["prompt_tokens"] = token_usage.get("prompt_tokens")
            metadata["completion_tokens"] = token_usage.get("completion_tokens")
        
        # Extract run info
        metadata["run_type"] = run.get("run_type")
        metadata["run_name"] = run.get("name")
        metadata["run_id"] = run.get("id")
        
        # Extract error if present
        if run.get("error"):
            metadata["error_message"] = str(run["error"])
        
        return Metadata(**metadata)
    
    @staticmethod
    def _extract_langsmith_context(run: Dict[str, Any]) -> str:
        """Extract context from LangSmith run."""
        # Build context from parent runs or inputs
        context_parts = []
        
        # Add parent info
        if "parent_run_id" in run:
            context_parts.append(f"Parent run: {run['parent_run_id']}")
        
        # Add tags if present
        tags = run.get("tags", [])
        if tags:
            context_parts.append(f"Tags: {', '.join(tags)}")
        
        # Add inputs preview
        inputs = run.get("inputs", {})
        if inputs:
            inputs_str = json.dumps(inputs, indent=2)
            context_parts.append(f"Inputs: {inputs_str[:200]}")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def _detect_generic_step_type(event: Dict[str, Any]) -> str:
        """Detect step type from generic event."""
        event_type = str(event.get("type", "")).lower()
        
        type_map = {
            "think": "think",
            "thought": "think",
            "reason": "think",
            "analyze": "think",
            "decision": "decision",
            "choose": "decision",
            "select": "decision",
            "tool_call": "tool_call",
            "tool": "tool_call",
            "action": "tool_call",
            "tool_result": "tool_result",
            "result": "tool_result",
            "output": "tool_result",
            "observation": "observation",
            "observe": "observation",
            "see": "observation",
            "error": "error",
            "exception": "error",
            "fail": "error"
        }
        
        if event_type in type_map:
            return type_map[event_type]
        
        # Try to infer from content
        content = str(event.get("data", event.get("content", ""))).lower()
        
        if any(word in content for word in ["think", "reason", "analyze"]):
            return "think"
        elif any(word in content for word in ["decide", "choose", "select"]):
            return "decision"
        elif any(word in content for word in ["call", "execute", "run", "tool"]):
            return "tool_call"
        elif any(word in content for word in ["result", "output", "return"]):
            return "tool_result"
        elif any(word in content for word in ["observe", "see", "notice"]):
            return "observation"
        elif any(word in content for word in ["error", "exception", "failed"]):
            return "error"
        
        return "unknown"
    
    @staticmethod
    def _detect_message_step_type(msg: Dict[str, Any]) -> str:
        """Detect step type from generic message."""
        role = str(msg.get("role", "")).lower()
        
        if role == "assistant" or role == "ai":
            content = str(msg.get("content", "")).lower()
            if any(word in content for word in ["think", "reason", "analyze"]):
                return "think"
            elif any(word in content for word in ["decide", "choose", "select"]):
                return "decision"
            elif "tool" in content or "function" in content:
                return "tool_call"
            else:
                return "decision"
        
        elif role == "user" or role == "human":
            return "observation"
        
        elif role == "system":
            return "think"
        
        elif role == "function" or role == "tool":
            return "tool_result"
        
        else:
            return "unknown"
    
    @staticmethod
    def auto_detect_and_convert(trace_data: Dict[str, Any]) -> List[SessionStepCreate]:
        """Auto-detect trace format and convert to steps."""
        # Check for OpenAI format
        if "choices" in trace_data or "messages" in trace_data and "model" in trace_data:
            try:
                return TraceIngestionAdapter.from_openai(trace_data)
            except Exception:
                pass
        
        # Check for Anthropic format
        if "content" in trace_data and "model" in trace_data:
            try:
                return TraceIngestionAdapter.from_anthropic(trace_data)
            except Exception:
                pass
        
        # Check for LangSmith format
        if "runs" in trace_data or "run_type" in trace_data:
            try:
                return TraceIngestionAdapter.from_langsmith(trace_data)
            except Exception:
                pass
        
        # Try generic conversion
        try:
            return TraceIngestionAdapter.from_generic(trace_data)
        except Exception as e:
            raise ValueError(f"Could not convert trace data: {e}")
    
    @staticmethod
    def validate_and_normalize(steps: List[SessionStepCreate]) -> List[SessionStepCreate]:
        """Validate and normalize steps."""
        normalized = []
        
        for i, step in enumerate(steps, 1):
            # Ensure step number is sequential
            step.step_number = i
            
            # Validate step type
            if step.step_type not in ["think", "tool_call", "tool_result", "observation", "decision", "error"]:
                # Try to auto-detect
                content_lower = step.content.lower()
                
                if any(word in content_lower for word in ["think", "reason", "analyze"]):
                    step.step_type = "think"
                elif any(word in content_lower for word in ["decide", "choose", "select"]):
                    step.step_type = "decision"
                elif any(word in content_lower for word in ["call", "execute", "run", "tool"]):
                    step.step_type = "tool_call"
                elif any(word in content_lower for word in ["result", "output", "return"]):
                    step.step_type = "tool_result"
                elif any(word in content_lower for word in ["observe", "see", "notice"]):
                    step.step_type = "observation"
                elif any(word in content_lower for word in ["error", "exception", "failed"]):
                    step.step_type = "error"
                else:
                    # Default based on agent
                    if step.agent_id and "tool" in step.agent_id.lower():
                        step.step_type = "tool_call"
                    else:
                        step.step_type = "think"
            
            # Ensure metadata is proper Metadata object
            if step.meta_data and not isinstance(step.meta_data, Metadata):
                if isinstance(step.meta_data, dict):
                    step.meta_data = Metadata(**step.meta_data)
                else:
                    step.meta_data = Metadata()
            
            normalized.append(step)
        
        return normalized