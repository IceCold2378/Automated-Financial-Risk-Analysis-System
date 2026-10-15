"""
LangGraph Agent Orchestrator — multi-step underwriting analysis.

This module defines the agent as a directed graph (StateGraph) where each node
performs a focused task (retrieve, extract, analyze, check compliance, etc.)
and edges control the flow based on state conditions.

Flow:
  START → extract_facts → retrieve_context → llm_analyze → validate_citations
        → calculate_risk → generate_output → END
"""

import json
import logging
import time

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState, AgentConfig, AgentTrace, StepStatus
from app.agent.tools import (
    build_retrieval_queries,
    calculate_risk_score,
    determine_recommendation,
    extract_document_facts,
    format_sources,
    retrieve_sources,
    validate_citations,
)
from app.models.schemas import AnalysisResult, RiskFlag, RiskLevel

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Graph Node Functions
# ══════════════════════════════════════════════════════════════════

def node_extract_facts(state: AgentState) -> dict:
    """
    Node 1: Extract structured facts from the raw document using regex
    pattern matching. This is deterministic pre-processing before LLM.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("extract_facts")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        document_text = state["document_text"]
        step.input_summary = f"Document: {len(document_text)} chars"

        facts = extract_document_facts(document_text)

        found_facts = {k: v for k, v in facts.items() if v is not None and v is not False}
        step.output_summary = f"Extracted {len(found_facts)} facts"
        step.tool_calls = [{"tool": "extract_document_facts", "result_count": len(found_facts)}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        return {
            "extracted_facts": facts,
            "reasoning": state.get("reasoning", []) + [
                f"Extracted {len(found_facts)} structured facts from document"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("extract_facts node failed: %s", e)
        return {
            "extracted_facts": {},
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_retrieve_context(state: AgentState) -> dict:
    """
    Node 2: Retrieve relevant handbook excerpts (with page/section metadata) using
    several focused queries built from the extracted facts and topics in the document.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("retrieve_context")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        from app.services.vector_service import get_retriever
        retriever = get_retriever()

        queries = build_retrieval_queries(state.get("extracted_facts", {}), state["document_text"])
        sources = retrieve_sources(queries, retriever)

        step.input_summary = f"{len(queries)} queries"
        step.output_summary = f"Retrieved {len(sources)} excerpts from pages {sorted({s['page'] for s in sources})}"
        step.tool_calls = [{"tool": "retrieve_sources", "queries": len(queries), "chunks": len(sources)}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        return {
            "retrieved_sources": sources,
            "retrieved_guidelines": [s["text"] for s in sources],
            "retrieval_query": " | ".join(queries),
            "retrieval_count": len(sources),
            "reasoning": state.get("reasoning", []) + [
                f"Retrieved {len(sources)} handbook excerpts using {len(queries)} queries"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("retrieve_context node failed: %s", e)
        return {
            "retrieved_sources": [],
            "retrieved_guidelines": [],
            "retrieval_count": 0,
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_llm_analyze(state: AgentState) -> dict:
    """
    Node 3: Ask the LLM to analyze the application against the retrieved excerpts.
    Each flag must cite one of the excerpt pages; citations are verified in the next node.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("llm_analyze")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        from app.services.ai_service import get_analysis_chain

        document_text = state["document_text"]
        sources = state.get("retrieved_sources", [])
        if not sources:
            raise RuntimeError("No guideline excerpts were retrieved; cannot analyze without grounding")
        step.input_summary = f"Document ({len(document_text)} chars) + {len(sources)} excerpts"

        raw_answer = get_analysis_chain().invoke(
            {"context": format_sources(sources), "input": document_text}
        )

        step.output_summary = f"LLM response: {len(raw_answer)} chars"
        step.tool_calls = [{"tool": "analysis_chain.invoke", "response_length": len(raw_answer)}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        text = raw_answer.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

        llm_ok, llm_flags, llm_summary, llm_detailed, llm_score = True, [], raw_answer[:500], raw_answer, None
        try:
            data = json.loads(text)
            llm_flags = data.get("risk_flags", [])
            llm_summary = data.get("summary", "")
            llm_detailed = data.get("detailed_analysis", raw_answer)
            llm_score = int(data["overall_risk_score"]) if "overall_risk_score" in data else None
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("LLM did not return valid JSON, using raw text")
            llm_ok = False

        return {
            "summary": llm_summary,
            "detailed_analysis": llm_detailed,
            "llm_ok": llm_ok,
            "llm_risk_score": llm_score,
            "risk_flags": [
                {
                    "category": f.get("category", "Unknown"),
                    "description": f.get("description", ""),
                    "severity": f.get("severity", "moderate"),
                    "guideline_reference": f.get("guideline_reference", ""),
                    "confidence": min(1.0, max(0.0, float(f.get("confidence", 0.5)))),
                }
                for f in llm_flags
            ],
            "reasoning": state.get("reasoning", []) + [
                f"LLM analysis complete — found {len(llm_flags)} risk flags"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("llm_analyze node failed: %s", e)
        return {
            "summary": "Analysis could not be completed due to an AI service error.",
            "detailed_analysis": str(e),
            "risk_flags": [],
            "llm_ok": False,
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_validate_citations(state: AgentState) -> dict:
    """
    Node 4: Verify that every flag's citation points at an excerpt that was actually retrieved.
    Ungrounded citations are marked and their confidence capped.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("validate_citations")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        flags, stats = validate_citations(state.get("risk_flags", []), state.get("retrieved_sources", []))
        step.input_summary = f"{stats['flags']} flags"
        step.output_summary = f"{stats['grounded']}/{stats['flags']} citations grounded in retrieved pages"
        step.tool_calls = [{"tool": "validate_citations", **stats}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000
        return {
            "risk_flags": flags,
            "citation_stats": stats,
            "guidelines_checked": len({s["page"] for s in state.get("retrieved_sources", [])}),
            "reasoning": state.get("reasoning", []) + [
                f"Citation check: {stats['grounded']}/{stats['flags']} flags cite retrieved handbook pages"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("validate_citations node failed: %s", e)
        return {"trace": trace.to_dict(), "error": str(e)}


def node_calculate_risk(state: AgentState) -> dict:
    """
    Node 5: Score the application from grounded flags (+ the LLM's own score) and
    recommend. If the LLM step failed, fail safe to MANUAL_REVIEW rather than approve.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("calculate_risk")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        flags = state.get("risk_flags", [])
        if state.get("llm_ok", False):
            risk_score, risk_level = calculate_risk_score(flags, state.get("llm_risk_score"))
            recommendation = determine_recommendation(risk_score, flags)
        else:
            risk_score, risk_level, recommendation = 50, RiskLevel.MODERATE.value, "MANUAL_REVIEW"

        step.input_summary = f"{len(flags)} flags"
        step.output_summary = f"Score={risk_score}, Level={risk_level}, Rec={recommendation}"
        step.tool_calls = [
            {"tool": "calculate_risk_score", "score": risk_score, "level": risk_level},
            {"tool": "determine_recommendation", "result": recommendation},
        ]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "reasoning": state.get("reasoning", []) + [
                f"Risk score: {risk_score}/100 ({risk_level}) → Recommendation: {recommendation}"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("calculate_risk node failed: %s", e)
        return {
            "risk_score": 50,
            "risk_level": RiskLevel.MODERATE.value,
            "recommendation": "MANUAL_REVIEW",
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_generate_output(state: AgentState) -> dict:
    """
    Node 6 (Final): Assemble the complete AnalysisResult from all
    previous nodes' output. This is the terminal node.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("generate_output")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        risk_flags = []
        for rf in state.get("risk_flags", []):
            try:
                risk_flags.append(RiskFlag(
                    category=rf.get("category", "Unknown"),
                    description=rf.get("description", ""),
                    severity=rf.get("severity", "moderate"),
                    guideline_reference=rf.get("guideline_reference", ""),
                    confidence=min(1.0, max(0.0, float(rf.get("confidence", 0.5)))),
                ))
            except Exception as e:
                logger.warning("Skipping malformed risk flag: %s", e)

        result = AnalysisResult(
            summary=state.get("summary", "Analysis completed"),
            overall_risk_score=state.get("risk_score", 50),
            overall_risk_level=state.get("risk_level", RiskLevel.MODERATE.value),
            risk_flags=risk_flags,
            recommendation=state.get("recommendation", "MANUAL_REVIEW"),
            detailed_analysis=state.get("detailed_analysis", ""),
            guidelines_checked=state.get("guidelines_checked", 0),
            processing_time_seconds=0.0,  # Set by caller
        )

        step.output_summary = f"Final result: score={result.overall_risk_score}, flags={len(risk_flags)}"
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000
        trace.final_status = StepStatus.COMPLETE

        return {
            "analysis_result": result.model_dump(),
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        trace.final_status = StepStatus.FAILED
        logger.exception("generate_output node failed: %s", e)
        return {
            "analysis_result": None,
            "trace": trace.to_dict(),
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════
#  Build the LangGraph
# ══════════════════════════════════════════════════════════════════

def _build_graph() -> StateGraph:
    """
    Construct the agent graph:

        START → extract_facts → retrieve_context → llm_analyze
              → validate_citations → calculate_risk → generate_output → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("extract_facts", node_extract_facts)
    graph.add_node("retrieve_context", node_retrieve_context)
    graph.add_node("llm_analyze", node_llm_analyze)
    graph.add_node("validate_citations", node_validate_citations)
    graph.add_node("calculate_risk", node_calculate_risk)
    graph.add_node("generate_output", node_generate_output)

    # Define edges (linear pipeline for v1; conditional edges in future)
    graph.add_edge(START, "extract_facts")
    graph.add_edge("extract_facts", "retrieve_context")
    graph.add_edge("retrieve_context", "llm_analyze")
    graph.add_edge("llm_analyze", "validate_citations")
    graph.add_edge("validate_citations", "calculate_risk")
    graph.add_edge("calculate_risk", "generate_output")
    graph.add_edge("generate_output", END)

    return graph


# Compiled graph (singleton)
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph().compile()
    return _compiled_graph


# ══════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════

async def run_agent_analysis(
    document_text: str,
    filename: str = "unknown",
    config: AgentConfig | None = None,
) -> tuple[AnalysisResult, dict]:
    """
    Run the full agentic analysis pipeline.

    Args:
        document_text: Extracted text from the uploaded document
        filename: Original filename for reference
        config: Optional agent configuration

    Returns:
        (AnalysisResult, trace_dict) — the structured result + audit trail
    """
    if config is None:
        config = AgentConfig()

    start_time = time.time()
    logger.info(
        "Starting agent analysis: file=%s, text=%d chars, max_iter=%d",
        filename, len(document_text), config.max_iterations,
    )

    # Prepare initial state
    initial_state: AgentState = {
        "document_text": document_text,
        "document_filename": filename,
        "config": {
            "max_iterations": config.max_iterations,
            "enable_tools": config.enable_tools,
            "temperature": config.temperature,
        },
        "extracted_facts": {},
        "retrieved_guidelines": [],
        "retrieved_sources": [],
        "retrieval_query": "",
        "retrieval_count": 0,
        "risk_flags": [],
        "risk_score": 50,
        "risk_level": "moderate",
        "current_step": "start",
        "reasoning": [],
        "iteration": 0,
        "should_continue": True,
        "agent_messages": [],
        "summary": "",
        "recommendation": "MANUAL_REVIEW",
        "detailed_analysis": "",
        "guidelines_checked": 0,
        "analysis_result": None,
        "trace": AgentTrace().to_dict(),
        "error": None,
    }

    # Run the graph
    try:
        graph = _get_graph()
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.exception("Agent graph execution failed: %s", e)
        # Return a fallback result
        fallback = AnalysisResult(
            summary="Agent analysis failed — falling back to basic result",
            overall_risk_score=50,
            overall_risk_level=RiskLevel.MODERATE,
            risk_flags=[],
            recommendation="MANUAL_REVIEW",
            detailed_analysis=f"Agent error: {str(e)}",
            guidelines_checked=0,
            processing_time_seconds=round(time.time() - start_time, 2),
        )
        return fallback, {"error": str(e), "steps": []}

    # Build the final AnalysisResult
    result_data = final_state.get("analysis_result")
    if result_data:
        result = AnalysisResult(**result_data)
    else:
        result = AnalysisResult(
            summary=final_state.get("summary", "Analysis completed"),
            overall_risk_score=final_state.get("risk_score", 50),
            overall_risk_level=final_state.get("risk_level", "moderate"),
            risk_flags=[],
            recommendation=final_state.get("recommendation", "MANUAL_REVIEW"),
            detailed_analysis=final_state.get("detailed_analysis", ""),
            guidelines_checked=final_state.get("guidelines_checked", 0),
            processing_time_seconds=0.0,
        )

    result.processing_time_seconds = round(time.time() - start_time, 2)

    trace_dict = final_state.get("trace", {})
    if isinstance(trace_dict, dict):
        trace_dict["total_duration_ms"] = result.processing_time_seconds * 1000

    logger.info(
        "Agent analysis complete: score=%d, level=%s, rec=%s, flags=%d, time=%.2fs",
        result.overall_risk_score, result.overall_risk_level,
        result.recommendation, len(result.risk_flags),
        result.processing_time_seconds,
    )

    return result, trace_dict
