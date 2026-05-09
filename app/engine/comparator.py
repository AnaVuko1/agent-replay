from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import difflib
from collections import defaultdict

from app.models import SessionStep, Replay, ReplayStep
from app.schemas import ReplayComparisonResponse, ReplayMetrics, ReplayStepDiff


class Comparator:
    """Compares replays and sessions to identify differences."""
    
    def compare_replays(
        self,
        replay1: Replay,
        replay1_steps: List[ReplayStep],
        replay2: Replay,
        replay2_steps: List[ReplayStep]
    ) -> Dict[str, Any]:
        """Compare two replays of the same session."""
        if not replay1_steps or not replay2_steps:
            raise ValueError("Both replays must have steps")
        
        # Ensure steps are sorted
        replay1_steps.sort(key=lambda s: s.step_number)
        replay2_steps.sort(key=lambda s: s.step_number)
        
        # Calculate metrics for each replay
        metrics1 = self._calculate_replay_metrics(replay1_steps)
        metrics2 = self._calculate_replay_metrics(replay2_steps)
        
        # Compare step by step
        step_diffs = self._compare_step_sequences(replay1_steps, replay2_steps)
        
        # Calculate aggregated differences
        aggregated_diff = self._aggregate_differences(metrics1, metrics2, step_diffs)
        
        # Determine which replay is "better" (lower divergence from original)
        replay1_divergence = sum(s.divergence_score or 0 for s in replay1_steps) / len(replay1_steps)
        replay2_divergence = sum(s.divergence_score or 0 for s in replay2_steps) / len(replay2_steps)
        
        if replay1_divergence < replay2_divergence:
            better_replay = {"id": replay1.id, "name": replay1.name, "reason": "closer_to_original"}
        elif replay2_divergence < replay1_divergence:
            better_replay = {"id": replay2.id, "name": replay2.name, "reason": "closer_to_original"}
        else:
            # Compare other metrics
            if metrics1.avg_divergence_score < metrics2.avg_divergence_score:
                better_replay = {"id": replay1.id, "name": replay1.name, "reason": "lower_average_divergence"}
            elif metrics2.avg_divergence_score < metrics1.avg_divergence_score:
                better_replay = {"id": replay2.id, "name": replay2.name, "reason": "lower_average_divergence"}
            elif metrics1.token_delta < metrics2.token_delta:
                better_replay = {"id": replay1.id, "name": replay1.name, "reason": "more_efficient"}
            elif metrics2.token_delta < metrics1.token_delta:
                better_replay = {"id": replay2.id, "name": replay2.name, "reason": "more_efficient"}
            else:
                better_replay = {"id": replay1.id, "name": replay1.name, "reason": "equal_performance"}
        
        return {
            "replays": {
                "first": {"id": replay1.id, "name": replay1.name, "config": replay1.replay_config},
                "second": {"id": replay2.id, "name": replay2.name, "config": replay2.replay_config},
                "better": better_replay
            },
            "metrics": {
                "first": metrics1.model_dump(),
                "second": metrics2.model_dump(),
                "deltas": {
                    "avg_divergence_delta": metrics1.avg_divergence_score - metrics2.avg_divergence_score,
                    "steps_diverged_delta": metrics1.steps_diverged - metrics2.steps_diverged,
                    "latency_delta_ms": metrics1.latency_delta_ms - metrics2.latency_delta_ms,
                    "token_delta": metrics1.token_delta - metrics2.token_delta
                }
            },
            "step_diffs": step_diffs,
            "aggregated_diff": aggregated_diff,
            "insights": self._generate_comparison_insights(
                replay1.replay_config,
                replay2.replay_config,
                step_diffs,
                aggregated_diff
            )
        }
    
    def compare_session_to_replay(
        self,
        session_steps: List[SessionStep],
        replay_steps: List[ReplayStep]
    ) -> Dict[str, Any]:
        """Compare original session to a replay."""
        if not session_steps or not replay_steps:
            raise ValueError("Both session and replay must have steps")
        
        # Sort steps
        session_steps.sort(key=lambda s: s.step_number)
        replay_steps.sort(key=lambda s: s.step_number)
        
        # Ensure same number of steps
        min_steps = min(len(session_steps), len(replay_steps))
        session_steps = session_steps[:min_steps]
        replay_steps = replay_steps[:min_steps]
        
        # Calculate replay metrics (divergence already calculated)
        replay_metrics = self._calculate_replay_metrics(replay_steps)
        
        # Compare step by step
        step_diffs = []
        for session_step, replay_step in zip(session_steps, replay_steps):
            # Use pre-calculated divergence or compute
            divergence = replay_step.divergence_score
            if divergence is None:
                divergence = self._calculate_divergence(
                    session_step.content,
                    replay_step.content
                )
            
            diff_html = self._generate_diff_html(
                session_step.content,
                replay_step.content
            )
            
            step_diffs.append({
                "step_number": session_step.step_number,
                "step_type": session_step.step_type,
                "original_content": session_step.content[:500],
                "replayed_content": replay_step.content[:500],
                "divergence_score": divergence,
                "diff_html": diff_html
            })
        
        # Calculate session metrics
        session_metrics = self._calculate_session_metrics(session_steps)
        
        # Analyze change impact
        change_impact = self._analyze_change_impact(step_diffs)
        
        return {
            "metrics": {
                "session": session_metrics,
                "replay": replay_metrics.model_dump()
            },
            "step_diffs": step_diffs,
            "change_impact": change_impact,
            "summary": self._generate_session_replay_summary(step_diffs, change_impact)
        }
    
    def compare_sessions(
        self,
        session1_steps: List[SessionStep],
        session2_steps: List[SessionStep]
    ) -> Dict[str, Any]:
        """Compare two different sessions (for similar tasks)."""
        if not session1_steps or not session2_steps:
            raise ValueError("Both sessions must have steps")
        
        # Sort steps
        session1_steps.sort(key=lambda s: s.step_number)
        session2_steps.sort(key=lambda s: s.step_number)
        
        # Calculate metrics for each session
        metrics1 = self._calculate_session_metrics(session1_steps)
        metrics2 = self._calculate_session_metrics(session2_steps)
        
        # Align steps by type and approximate time
        aligned_diffs = self._align_and_compare_sessions(session1_steps, session2_steps)
        
        # Calculate aggregated differences
        aggregated_diff = self._aggregate_session_differences(metrics1, metrics2, aligned_diffs)
        
        # Determine which session was more efficient
        efficiency_comparison = self._compare_session_efficiency(metrics1, metrics2)
        
        return {
            "metrics": {
                "first": metrics1,
                "second": metrics2
            },
            "aligned_diffs": aligned_diffs,
            "aggregated_diff": aggregated_diff,
            "efficiency_comparison": efficiency_comparison,
            "recommendations": self._generate_session_comparison_recommendations(
                metrics1, metrics2, aligned_diffs
            )
        }
    
    def _calculate_replay_metrics(self, replay_steps: List[ReplayStep]) -> ReplayMetrics:
        """Calculate metrics for replay steps."""
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
        steps_diverged = sum(1 for step in replay_steps if step.divergence_score and step.divergence_score > 0.1)
        
        # Calculate average divergence
        total_divergence = sum(step.divergence_score or 0.0 for step in replay_steps)
        avg_divergence = total_divergence / len(replay_steps)
        
        # Calculate token delta (simulated)
        token_delta = 0
        for step in replay_steps:
            original_tokens = len(step.original_content.split()) if step.original_content else 0
            replayed_tokens = len(step.content.split())
            token_delta += replayed_tokens - original_tokens
        
        # Simulate latency delta
        latency_delta_ms = 0.0
        for step in replay_steps:
            if step.divergence_score and step.divergence_score > 0.3:
                latency_delta_ms += step.divergence_score * 100
        
        # Determine outcome
        outcome_diff = None
        if steps_diverged > 0:
            last_step = replay_steps[-1]
            if last_step.divergence_score and last_step.divergence_score > 0.5:
                outcome_diff = "different_final_decision"
            elif last_step.divergence_score and last_step.divergence_score > 0.2:
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
    
    def _calculate_session_metrics(self, session_steps: List[SessionStep]) -> Dict[str, Any]:
        """Calculate metrics for session steps."""
        if not session_steps:
            return {}
        
        # Count step types
        type_counts = defaultdict(int)
        for step in session_steps:
            type_counts[step.step_type] += 1
        
        # Calculate efficiency metrics
        total_tokens = 0
        steps_with_tokens = 0
        total_latency_ms = 0
        steps_with_latency = 0
        
        for step in session_steps:
            metadata = step.meta_data or {}
            if "tokens_used" in metadata:
                total_tokens += metadata["tokens_used"]
                steps_with_tokens += 1
            if "latency_ms" in metadata:
                total_latency_ms += metadata["latency_ms"]
                steps_with_latency += 1
        
        avg_tokens = total_tokens / steps_with_tokens if steps_with_tokens > 0 else None
        avg_latency = total_latency_ms / steps_with_latency if steps_with_latency > 0 else None
        
        # Calculate success rate
        tool_calls = len([s for s in session_steps if s.step_type == "tool_call"])
        tool_errors = len([s for s in session_steps if s.step_type == "error" and "tool" in (s.meta_data or {}).get("tool_name", "").lower()])
        
        tool_success_rate = (tool_calls - tool_errors) / tool_calls if tool_calls > 0 else 1.0
        
        # Calculate decision density
        decisions = type_counts.get("decision", 0)
        decision_density = decisions / len(session_steps) if session_steps else 0
        
        return {
            "total_steps": len(session_steps),
            "step_type_distribution": dict(type_counts),
            "avg_tokens_per_step": avg_tokens,
            "avg_latency_ms_per_step": avg_latency,
            "tool_success_rate": tool_success_rate,
            "decision_density": decision_density,
            "efficiency_score": tool_success_rate * decision_density if decision_density > 0 else 0
        }
    
    def _compare_step_sequences(
        self,
        steps1: List[ReplayStep],
        steps2: List[ReplayStep]
    ) -> List[Dict[str, Any]]:
        """Compare two sequences of replay steps."""
        diffs = []
        
        # Match steps by step_number
        step_numbers = sorted(set(s.step_number for s in steps1) | set(s.step_number for s in steps2))
        
        for step_number in step_numbers:
            step1 = next((s for s in steps1 if s.step_number == step_number), None)
            step2 = next((s for s in steps2 if s.step_number == step_number), None)
            
            if not step1 or not step2:
                # Step missing in one replay
                diffs.append({
                    "step_number": step_number,
                    "type": "missing_step",
                    "missing_in": "first" if not step1 else "second",
                    "content": step2.content[:200] if step2 else step1.content[:200] if step1 else ""
                })
                continue
            
            # Compare content
            content_similarity = difflib.SequenceMatcher(
                None, 
                step1.content,
                step2.content
            ).ratio()
            
            # Compare divergence scores
            divergence1 = step1.divergence_score or 0
            divergence2 = step2.divergence_score or 0
            divergence_diff = abs(divergence1 - divergence2)
            
            # Determine difference level
            if content_similarity > 0.9:
                diff_level = "minor"
            elif content_similarity > 0.7:
                diff_level = "moderate"
            else:
                diff_level = "major"
            
            diffs.append({
                "step_number": step_number,
                "step_type": step1.step_type,
                "type": "content_diff",
                "diff_level": diff_level,
                "content_similarity": content_similarity,
                "divergence_diff": divergence_diff,
                "content1_preview": step1.content[:200],
                "content2_preview": step2.content[:200],
                "diff_html": self._generate_diff_html(step1.content, step2.content)
            })
        
        return diffs
    
    def _align_and_compare_sessions(
        self,
        session1_steps: List[SessionStep],
        session2_steps: List[SessionStep]
    ) -> List[Dict[str, Any]]:
        """Align steps from two sessions and compare them."""
        aligned = []
        
        # Simple alignment by step type sequence
        i, j = 0, 0
        
        while i < len(session1_steps) and j < len(session2_steps):
            step1 = session1_steps[i]
            step2 = session2_steps[j]
            
            # Try to align by step type
            if step1.step_type == step2.step_type:
                # Compare aligned steps
                similarity = difflib.SequenceMatcher(
                    None,
                    step1.content,
                    step2.content
                ).ratio()
                
                aligned.append({
                    "step1_number": step1.step_number,
                    "step2_number": step2.step_number,
                    "step_type": step1.step_type,
                    "similarity": similarity,
                    "content1_preview": step1.content[:200],
                    "content2_preview": step2.content[:200],
                    "alignment_score": 1.0 if similarity > 0.7 else 0.5
                })
                
                i += 1
                j += 1
            else:
                # Step types don't match - look ahead
                look_ahead = 3
                found_match = False
                
                for offset in range(1, min(look_ahead + 1, len(session1_steps) - i, len(session2_steps) - j)):
                    if session1_steps[i + offset].step_type == step2.step_type:
                        # Step1 skipped some steps
                        for k in range(offset):
                            aligned.append({
                                "step1_number": session1_steps[i + k].step_number,
                                "step2_number": None,
                                "step_type": session1_steps[i + k].step_type,
                                "similarity": 0,
                                "content1_preview": session1_steps[i + k].content[:200],
                                "content2_preview": "",
                                "alignment_score": 0
                            })
                        
                        i += offset + 1
                        j += 1
                        found_match = True
                        break
                    
                    elif step1.step_type == session2_steps[j + offset].step_type:
                        # Step2 skipped some steps
                        for k in range(offset):
                            aligned.append({
                                "step1_number": None,
                                "step2_number": session2_steps[j + k].step_number,
                                "step_type": session2_steps[j + k].step_type,
                                "similarity": 0,
                                "content1_preview": "",
                                "content2_preview": session2_steps[j + k].content[:200],
                                "alignment_score": 0
                            })
                        
                        i += 1
                        j += offset + 1
                        found_match = True
                        break
                
                if not found_match:
                    # No match found - advance both
                    aligned.append({
                        "step1_number": step1.step_number,
                        "step2_number": step2.step_number,
                        "step_type": f"{step1.step_type}/{step2.step_type}",
                        "similarity": 0,
                        "content1_preview": step1.content[:200],
                        "content2_preview": step2.content[:200],
                        "alignment_score": 0
                    })
                    
                    i += 1
                    j += 1
        
        # Handle remaining steps
        while i < len(session1_steps):
            step1 = session1_steps[i]
            aligned.append({
                "step1_number": step1.step_number,
                "step2_number": None,
                "step_type": step1.step_type,
                "similarity": 0,
                "content1_preview": step1.content[:200],
                "content2_preview": "",
                "alignment_score": 0
            })
            i += 1
        
        while j < len(session2_steps):
            step2 = session2_steps[j]
            aligned.append({
                "step1_number": None,
                "step2_number": step2.step_number,
                "step_type": step2.step_type,
                "similarity": 0,
                "content1_preview": "",
                "content2_preview": step2.content[:200],
                "alignment_score": 0
            })
            j += 1
        
        return aligned
    
    def _aggregate_differences(
        self,
        metrics1: ReplayMetrics,
        metrics2: ReplayMetrics,
        step_diffs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate differences between two replays."""
        # Count diff levels
        diff_counts = defaultdict(int)
        for diff in step_diffs:
            if "diff_level" in diff:
                diff_counts[diff["diff_level"]] += 1
        
        # Calculate average similarity
        similarities = [d.get("content_similarity", 0) for d in step_diffs if "content_similarity" in d]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Find steps with highest divergence
        high_divergence_steps = []
        for diff in step_diffs:
            if "divergence_diff" in diff and diff["divergence_diff"] > 0.3:
                high_divergence_steps.append({
                    "step_number": diff["step_number"],
                    "divergence_diff": diff["divergence_diff"],
                    "step_type": diff.get("step_type", "unknown")
                })
        
        return {
            "diff_counts": dict(diff_counts),
            "avg_similarity": avg_similarity,
            "high_divergence_steps": high_divergence_steps[:5],  # Top 5
            "total_differences": len([d for d in step_diffs if d.get("type") != "missing_step"]),
            "missing_steps": len([d for d in step_diffs if d.get("type") == "missing_step"]),
            "metrics_comparison": {
                "avg_divergence_ratio": metrics1.avg_divergence_score / metrics2.avg_divergence_score 
                    if metrics2.avg_divergence_score > 0 else float('inf'),
                "steps_diverged_ratio": metrics1.steps_diverged / metrics2.steps_diverged 
                    if metrics2.steps_diverged > 0 else float('inf'),
                "token_efficiency_ratio": metrics2.token_delta / metrics1.token_delta 
                    if metrics1.token_delta != 0 else float('inf')
            }
        }
    
    def _aggregate_session_differences(
        self,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any],
        aligned_diffs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate differences between two sessions."""
        # Calculate alignment quality
        alignment_scores = [d["alignment_score"] for d in aligned_diffs]
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        # Count aligned vs misaligned
        well_aligned = len([d for d in aligned_diffs if d["alignment_score"] > 0.7])
        poorly_aligned = len([d for d in aligned_diffs if d["alignment_score"] < 0.3])
        
        # Calculate content similarity for aligned steps
        similarities = [d["similarity"] for d in aligned_diffs if d["similarity"] > 0]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Compare metrics
        metric_diffs = {}
        for key in metrics1:
            if key in metrics2 and isinstance(metrics1[key], (int, float)) and isinstance(metrics2[key], (int, float)):
                if metrics1[key] != 0:
                    metric_diffs[key] = {
                        "first": metrics1[key],
                        "second": metrics2[key],
                        "ratio": metrics2[key] / metrics1[key],
                        "delta": metrics2[key] - metrics1[key]
                    }
        
        return {
            "alignment_quality": avg_alignment,
            "well_aligned_steps": well_aligned,
            "poorly_aligned_steps": poorly_aligned,
            "avg_content_similarity": avg_similarity,
            "metric_diffs": metric_diffs,
            "step_count_difference": abs(metrics1.get("total_steps", 0) - metrics2.get("total_steps", 0))
        }
    
    def _analyze_change_impact(self, step_diffs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of changes in a replay."""
        # Group by step type
        diff_by_type = defaultdict(list)
        for diff in step_diffs:
            step_type = diff.get("step_type", "unknown")
            diff_by_type[step_type].append(diff)
        
        # Calculate impact scores
        impact_scores = {}
        for step_type, diffs in diff_by_type.items():
            if not diffs:
                continue
            
            avg_divergence = sum(d.get("divergence_score", 0) for d in diffs) / len(diffs)
            max_divergence = max(d.get("divergence_score", 0) for d in diffs)
            
            # Weight by step type importance
            weights = {
                "decision": 2.0,
                "tool_call": 1.5,
                "think": 1.0,
                "observation": 0.8,
                "tool_result": 0.5,
                "error": 1.2
            }
            
            weight = weights.get(step_type, 1.0)
            impact_score = avg_divergence * weight
            
            impact_scores[step_type] = {
                "count": len(diffs),
                "avg_divergence": avg_divergence,
                "max_divergence": max_divergence,
                "weight": weight,
                "impact_score": impact_score
            }
        
        # Determine overall impact
        total_impact = sum(info["impact_score"] for info in impact_scores.values())
        
        if total_impact > 2.0:
            overall_impact = "high"
        elif total_impact > 1.0:
            overall_impact = "medium"
        elif total_impact > 0.3:
            overall_impact = "low"
        else:
            overall_impact = "negligible"
        
        # Find most impacted step types
        most_impacted = []
        for step_type, info in impact_scores.items():
            if info["impact_score"] > 0.5:
                most_impacted.append({
                    "step_type": step_type,
                    "impact_score": info["impact_score"],
                    "avg_divergence": info["avg_divergence"]
                })
        
        most_impacted.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return {
            "impact_scores": impact_scores,
            "total_impact": total_impact,
            "overall_impact": overall_impact,
            "most_impacted_step_types": most_impacted[:3],
            "high_impact_steps": [
                diff for diff in step_diffs 
                if diff.get("divergence_score", 0) > 0.5
            ][:5]  # Top 5
        }
    
    def _compare_session_efficiency(
        self,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare efficiency of two sessions."""
        efficiency1 = metrics1.get("efficiency_score", 0)
        efficiency2 = metrics2.get("efficiency_score", 0)
        
        if efficiency1 > efficiency2:
            more_efficient = "first"
            efficiency_ratio = efficiency1 / efficiency2 if efficiency2 > 0 else float('inf')
        else:
            more_efficient = "second"
            efficiency_ratio = efficiency2 / efficiency1 if efficiency1 > 0 else float('inf')
        
        # Compare token usage
        tokens1 = metrics1.get("avg_tokens_per_step")
        tokens2 = metrics2.get("avg_tokens_per_step")
        
        token_efficiency = None
        if tokens1 is not None and tokens2 is not None:
            if tokens1 < tokens2:
                token_efficient = "first"
                token_ratio = tokens2 / tokens1
            else:
                token_efficient = "second"
                token_ratio = tokens1 / tokens2
        
        # Compare latency
        latency1 = metrics1.get("avg_latency_ms_per_step")
        latency2 = metrics2.get("avg_latency_ms_per_step")
        
        latency_efficiency = None
        if latency1 is not None and latency2 is not None:
            if latency1 < latency2:
                latency_efficient = "first"
                latency_ratio = latency2 / latency1
            else:
                latency_efficient = "second"
                latency_ratio = latency1 / latency2
        
        return {
            "more_efficient": more_efficient,
            "efficiency_ratio": efficiency_ratio,
            "token_efficient": token_efficient if token_efficiency else None,
            "token_ratio": token_ratio if token_efficiency else None,
            "latency_efficient": latency_efficient if latency_efficiency else None,
            "latency_ratio": latency_ratio if latency_efficiency else None,
            "summary": self._generate_efficiency_summary(
                more_efficient, efficiency_ratio,
                token_efficient if token_efficiency else None,
                latency_efficient if latency_efficiency else None
            )
        }
    
    def _calculate_divergence(self, original: str, replayed: str) -> float:
        """Calculate divergence between two texts."""
        if original == replayed:
            return 0.0
        
        similarity = difflib.SequenceMatcher(None, original, replayed).ratio()
        divergence = 1.0 - similarity
        
        # Boost for structural changes
        lines_original = original.split('\n')
        lines_replayed = replayed.split('\n')
        
        if len(lines_original) != len(lines_replayed):
            divergence = min(1.0, divergence + 0.2)
        
        return round(divergence, 3)
    
    def _generate_diff_html(self, original: str, replayed: str) -> str:
        """Generate HTML diff between two texts."""
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
    
    def _generate_comparison_insights(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        step_diffs: List[Dict[str, Any]],
        aggregated_diff: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from comparison."""
        insights = []
        
        # Compare configurations
        config_diffs = {}
        for key in set(config1.keys()) | set(config2.keys()):
            if config1.get(key) != config2.get(key):
                config_diffs[key] = {
                    "first": config1.get(key),
                    "second": config2.get(key)
                }
        
        if config_diffs:
            insights.append(f"Configuration differences: {', '.join(config_diffs.keys())}")
        
        # Analyze diff patterns
        major_diffs = aggregated_diff.get("diff_counts", {}).get("major", 0)
        if major_diffs > 0:
            insights.append(
                f"Found {major_diffs} major differences in step content. "
                "Configuration changes significantly affected agent behavior."
            )
        
        # Check for missing steps
        missing_steps = aggregated_diff.get("missing_steps", 0)
        if missing_steps > 0:
            insights.append(
                f"{missing_steps} steps are missing in one replay. "
                "Check if configuration changes caused steps to be skipped."
            )
        
        # Overall similarity
        avg_similarity = aggregated_diff.get("avg_similarity", 0)
        if avg_similarity > 0.8:
            insights.append("Replays are very similar despite configuration changes.")
        elif avg_similarity > 0.5:
            insights.append("Replays show moderate differences from configuration changes.")
        else:
            insights.append("Replays are significantly different due to configuration changes.")
        
        return insights
    
    def _generate_session_replay_summary(
        self,
        step_diffs: List[Dict[str, Any]],
        change_impact: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of session vs replay comparison."""
        # Calculate statistics
        total_steps = len(step_diffs)
        diverged_steps = len([d for d in step_diffs if d.get("divergence_score", 0) > 0.1])
        high_divergence_steps = len([d for d in step_diffs if d.get("divergence_score", 0) > 0.5])
        
        # Determine overall impact
        overall_impact = change_impact.get("overall_impact", "negligible")
        
        # Generate summary text
        if overall_impact == "high":
            summary_text = f"High impact: {high_divergence_steps} steps significantly changed. Replay configuration dramatically altered agent behavior."
        elif overall_impact == "medium":
            summary_text = f"Medium impact: {diverged_steps} steps diverged. Configuration changes noticeably affected agent decisions."
        elif overall_impact == "low":
            summary_text = f"Low impact: {diverged_steps} steps slightly changed. Configuration had minor effects on agent behavior."
        else:
            summary_text = f"Negligible impact: Only {diverged_steps} steps diverged. Configuration changes had little effect."
        
        return {
            "total_steps": total_steps,
            "diverged_steps": diverged_steps,
            "high_divergence_steps": high_divergence_steps,
            "divergence_percentage": (diverged_steps / total_steps * 100) if total_steps > 0 else 0,
            "overall_impact": overall_impact,
            "summary_text": summary_text,
            "recommended_action": self._get_recommended_action(overall_impact, high_divergence_steps)
        }
    
    def _generate_session_comparison_recommendations(
        self,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any],
        aligned_diffs: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations from session comparison."""
        recommendations = []
        
        # Compare efficiency
        efficiency1 = metrics1.get("efficiency_score", 0)
        efficiency2 = metrics2.get("efficiency_score", 0)
        
        if efficiency1 > efficiency2 * 1.2:
            recommendations.append("First session was significantly more efficient. Consider adopting its approach.")
        elif efficiency2 > efficiency1 * 1.2:
            recommendations.append("Second session was significantly more efficient. Consider adopting its approach.")
        
        # Compare tool success rates
        success1 = metrics1.get("tool_success_rate", 1.0)
        success2 = metrics2.get("tool_success_rate", 1.0)
        
        if success1 > success2 + 0.2:
            recommendations.append("First session had higher tool success rate. Its tool selection/usage may be better.")
        elif success2 > success1 + 0.2:
            recommendations.append("Second session had higher tool success rate. Its tool selection/usage may be better.")
        
        # Compare decision density
        density1 = metrics1.get("decision_density", 0)
        density2 = metrics2.get("decision_density", 0)
        
        if density1 > density2 * 1.5:
            recommendations.append("First session made decisions more frequently. May indicate more active problem-solving.")
        elif density2 > density1 * 1.5:
            recommendations.append("Second session made decisions more frequently. May indicate more active problem-solving.")
        
        # Check alignment
        avg_alignment = sum(d["alignment_score"] for d in aligned_diffs) / len(aligned_diffs)
        if avg_alignment < 0.5:
            recommendations.append("Low step alignment between sessions. Agents used very different approaches to the same task.")
        
        # Check for errors
        errors1 = metrics1.get("step_type_distribution", {}).get("error", 0)
        errors2 = metrics2.get("step_type_distribution", {}).get("error", 0)
        
        if errors1 > errors2 * 2:
            recommendations.append(f"First session had {errors1} errors vs {errors2} in second. Second session's approach may be more robust.")
        elif errors2 > errors1 * 2:
            recommendations.append(f"Second session had {errors2} errors vs {errors1} in first. First session's approach may be more robust.")
        
        return recommendations
    
    def _generate_efficiency_summary(
        self,
        more_efficient: str,
        efficiency_ratio: float,
        token_efficient: Optional[str],
        latency_efficient: Optional[str]
    ) -> str:
        """Generate efficiency summary text."""
        summary_parts = []
        
        if efficiency_ratio > 1.5:
            summary_parts.append(f"{more_efficient.capitalize()} session was {efficiency_ratio:.1f}x more efficient overall.")
        elif efficiency_ratio > 1.1:
            summary_parts.append(f"{more_efficient.capitalize()} session was slightly more efficient.")
        
        if token_efficient:
            summary_parts.append(f"{token_efficient.capitalize()} session used tokens more efficiently.")
        
        if latency_efficient:
            summary_parts.append(f"{latency_efficient.capitalize()} session had lower latency per step.")
        
        if not summary_parts:
            summary_parts.append("Sessions were similarly efficient.")
        
        return " ".join(summary_parts)
    
    def _get_recommended_action(self, overall_impact: str, high_divergence_steps: int) -> str:
        """Get recommended action based on comparison results."""
        if overall_impact == "high":
            if high_divergence_steps > 5:
                return "Review all high-divergence steps carefully. Configuration changes had major effects."
            else:
                return "Review specific high-divergence steps. Configuration changes significantly affected key decisions."
        elif overall_impact == "medium":
            return "Review divergent steps. Consider whether configuration changes improved or worsened outcomes."
        elif overall_impact == "low":
            return "Configuration changes had minimal effect. Consider more significant changes if seeking different outcomes."
        else:
            return "Configuration changes had negligible effect. Current configuration is stable."