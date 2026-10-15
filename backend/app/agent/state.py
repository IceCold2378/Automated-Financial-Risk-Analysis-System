"""
Agent state management for the LangGraph underwriting agent.

AgentState   — The typed state flowing through the graph
AgentConfig  — Runtime configuration for the agent
AgentTrace   — Records each reasoning step for audit/explainability
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict



# ── Agent Step Status ──────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── Trace (Audit Trail) ───────────────────────────────────────────

@dataclass
class TraceStep:
    """A single recorded step in the agent's reasoning chain."""
    step_number: int
    node_name: str
    status: StepStatus = StepStatus.PENDING
    input_summary: str = ""
    output_summary: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "step": self.step_number,
            "node": self.node_name,
            "status": self.status.value,
            "input": self.input_summary,
            "output": self.output_summary,
            "tool_calls": self.tool_calls,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
        }


@dataclass
class AgentTrace:
    """Full trace of the agent's execution for audit and explainability."""
    steps: list[TraceStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    iterations: int = 0
    final_status: StepStatus = StepStatus.PENDING

    def add_step(self, node_name: str, step_number: int | None = None) -> TraceStep:
        num = step_number if step_number is not None else len(self.steps) + 1
        step = TraceStep(step_number=num, node_name=node_name)
        self.steps.append(step)
        return step

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_ms": round(self.total_duration_ms, 1),
            "iterations": self.iterations,
            "status": self.final_status.value if isinstance(self.final_status, StepStatus) else self.final_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentTrace:
        if not data:
            return cls()
        
        trace = cls()
        trace.total_duration_ms = data.get("total_duration_ms", 0.0)
        trace.iterations = data.get("iterations", 0)
        
        status_val = data.get("status", StepStatus.PENDING.value)
        try:
            trace.final_status = StepStatus(status_val)
        except ValueError:
            trace.final_status = StepStatus.PENDING
            
        steps = []
        for s in data.get("steps", []):
            step = TraceStep(
                step_number=s.get("step", 0),
                node_name=s.get("node", ""),
            )
            try:
                step.status = StepStatus(s.get("status", "pending"))
            except ValueError:
                step.status = StepStatus.PENDING
                
            step.input_summary = s.get("input", "")
            step.output_summary = s.get("output", "")
            step.tool_calls = s.get("tool_calls", [])
            step.duration_ms = s.get("duration_ms", 0.0)
            step.error = s.get("error")
            steps.append(step)
            
        trace.steps = steps
        return trace


# ── Agent Config ──────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Runtime configuration for a single agent run."""
    max_iterations: int = 5
    enable_tools: bool = True
    enable_compliance_check: bool = True
    enable_risk_calculation: bool = True
    enable_market_lookup: bool = False  # Requires external API
    temperature: float = 0.1
    verbose: bool = False


# ── LangGraph Agent State ─────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Typed state dictionary flowing through the LangGraph StateGraph.

    Each node reads from / writes to this state. LangGraph manages
    the state transitions between nodes automatically.
    """
    # ── Input ──
    document_text: str                      # Raw text extracted from uploaded document
    document_filename: str                  # Original filename for reference
    config: dict                            # Serialized AgentConfig

    # ── Retrieved Context ──
    retrieved_guidelines: list[str]         # Chunks retrieved from vector DB
    retrieval_query: str                    # The query used for retrieval
    retrieval_count: int                    # Number of chunks retrieved

    # ── Analysis State ──
    extracted_facts: dict                   # Structured facts from the document
    risk_flags: list[dict]                  # Risk flags found so far
    compliance_results: list[dict]          # Compliance check outcomes
    risk_score: int                         # Calculated risk score (0-100)
    risk_level: str                         # low / moderate / high / critical

    # ── Agent Reasoning ──
    current_step: str                       # Name of the current graph node
    reasoning: list[str]                    # Intermediate reasoning log
    iteration: int                          # Current iteration count
    should_continue: bool                   # Whether agent needs more steps
    agent_messages: list[dict]              # LLM message history

    # ── Output ──
    summary: str                            # Final analysis summary
    recommendation: str                     # APPROVE / DENY / MANUAL_REVIEW
    detailed_analysis: str                  # Full analysis text
    guidelines_checked: int                 # Count of guidelines evaluated
    analysis_result: dict | None            # Final serialized AnalysisResult

    # ── Trace ──
    trace: dict                             # Serialized AgentTrace
    error: str | None                       # Error message if failed
