"""
LangGraph Agent Orchestrator — multi-step underwriting analysis.

This module defines the agent as a directed graph (StateGraph) where each node
performs a focused task (retrieve, extract, analyze, check compliance, etc.)
and edges control the flow based on state conditions.

Flow:
  START → extract_facts → retrieve_context → llm_analyze → check_compliance
        → calculate_risk → generate_output → END
"""

import json
import logging
import time

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState, AgentConfig, AgentTrace, StepStatus
from app.agent.tools import (
    retrieve_guidelines,
    extract_document_facts,
    calculate_risk_score,
    check_compliance,
    determine_recommendation,
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
    Node 2: Retrieve relevant guidelines from the vector database.
    Uses the document text as the search query.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("retrieve_context")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        from app.services.vector_service import get_retriever
        retriever = get_retriever()

        document_text = state["document_text"]
        # Use a focused query: combine key facts with document start
        facts = state.get("extracted_facts", {})
        query_parts = []
        if facts.get("credit_score"):
            query_parts.append(f"credit score {facts['credit_score']}")
        if facts.get("dti_ratio"):
            query_parts.append(f"DTI ratio {facts['dti_ratio']}%")
        if facts.get("loan_amount"):
            query_parts.append(f"loan amount ${facts['loan_amount']:,.0f}")
        if facts.get("bankruptcy_mentioned"):
            query_parts.append("bankruptcy history")

        query = " ".join(query_parts) if query_parts else document_text[:2000]

        chunks = retrieve_guidelines(query, retriever)

        step.input_summary = f"Query: {query[:200]}..."
        step.output_summary = f"Retrieved {len(chunks)} guideline chunks"
        step.tool_calls = [{"tool": "retrieve_guidelines", "chunks": len(chunks)}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        return {
            "retrieved_guidelines": chunks,
            "retrieval_query": query,
            "retrieval_count": len(chunks),
            "reasoning": state.get("reasoning", []) + [
                f"Retrieved {len(chunks)} relevant guideline sections"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("retrieve_context node failed: %s", e)
        return {
            "retrieved_guidelines": [],
            "retrieval_count": 0,
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_llm_analyze(state: AgentState) -> dict:
    """
    Node 3: Use the LLM to perform deep analysis of the document
    against the retrieved guidelines. This is the core AI reasoning step.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("llm_analyze")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        from app.services.ai_service import get_rag_chain

        document_text = state["document_text"]
        chain = get_rag_chain()

        step.input_summary = f"Document ({len(document_text)} chars) + guidelines"

        # LCEL chain takes the input text directly as a string
        raw_answer = chain.invoke(document_text)

        step.output_summary = f"LLM response: {len(raw_answer)} chars"
        step.tool_calls = [{"tool": "rag_chain.invoke", "response_length": len(raw_answer)}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        # Try to parse JSON from LLM response

        text = raw_answer.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

        try:
            data = json.loads(text)
            llm_risk_flags = data.get("risk_flags", [])
            llm_summary = data.get("summary", "")
            llm_detailed = data.get("detailed_analysis", raw_answer)
        except json.JSONDecodeError:
            logger.warning("LLM did not return JSON, using raw text")
            llm_summary = raw_answer[:500]

        return {
            "summary": llm_summary,
            "detailed_analysis": llm_detailed,
            "risk_flags": [
                {
                    "category": f.get("category", "Unknown"),
                    "description": f.get("description", ""),
                    "severity": f.get("severity", "moderate"),
                    "guideline_reference": f.get("guideline_reference", ""),
                    "confidence": min(1.0, max(0.0, float(f.get("confidence", 0.5)))),
                }
                for f in llm_risk_flags
            ],
            "reasoning": state.get("reasoning", []) + [
                f"LLM analysis complete — found {len(llm_risk_flags)} risk flags"
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
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_check_compliance(state: AgentState) -> dict:
    """
    Node 4: Run deterministic compliance checks against extracted facts.
    This produces a structured list of pass/fail results per rule.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("check_compliance")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        facts = state.get("extracted_facts", {})
        step.input_summary = f"Checking {len(facts)} extracted facts"

        results = check_compliance(facts)

        violations = [r for r in results if r["violated"]]
        step.output_summary = f"{len(violations)}/{len(results)} rules violated"
        step.tool_calls = [{"tool": "check_compliance", "violations": len(violations)}]
        step.status = StepStatus.COMPLETE
        step.duration_ms = (time.time() - start) * 1000

        # Merge compliance violations into risk flags
        existing_flags = state.get("risk_flags", [])
        compliance_flags = [
            {
                "category": r["rule_name"],
                "description": r["description"],
                "severity": r["severity"],
                "guideline_reference": r["guideline_reference"],
                "confidence": 1.0,  # Deterministic = full confidence
            }
            for r in results if r["violated"]
        ]

        # Deduplicate by category
        seen = {f["category"] for f in existing_flags}
        new_flags = [f for f in compliance_flags if f["category"] not in seen]

        return {
            "compliance_results": results,
            "risk_flags": existing_flags + new_flags,
            "guidelines_checked": len(results),
            "reasoning": state.get("reasoning", []) + [
                f"Compliance check: {len(violations)} violations found across {len(results)} rules"
            ],
            "trace": trace.to_dict(),
        }
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = str(e)
        step.duration_ms = (time.time() - start) * 1000
        logger.exception("check_compliance node failed: %s", e)
        return {
            "compliance_results": [],
            "trace": trace.to_dict(),
            "error": str(e),
        }


def node_calculate_risk(state: AgentState) -> dict:
    """
    Node 5: Calculate the final risk score and recommendation
    based on facts, compliance results, and LLM findings.
    """
    trace = AgentTrace.from_dict(state.get("trace", {}))
    step = trace.add_step("calculate_risk")
    step.status = StepStatus.RUNNING
    start = time.time()

    try:
        facts = state.get("extracted_facts", {})
        compliance_results = state.get("compliance_results", [])

        risk_score, risk_level = calculate_risk_score(facts, compliance_results)
        recommendation = determine_recommendation(risk_score, compliance_results)

        step.input_summary = f"Facts + {len(compliance_results)} compliance results"
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
              → check_compliance → calculate_risk → generate_output → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("extract_facts", node_extract_facts)
    graph.add_node("retrieve_context", node_retrieve_context)
    graph.add_node("llm_analyze", node_llm_analyze)
    graph.add_node("check_compliance", node_check_compliance)
    graph.add_node("calculate_risk", node_calculate_risk)
    graph.add_node("generate_output", node_generate_output)

    # Define edges (linear pipeline for v1; conditional edges in future)
    graph.add_edge(START, "extract_facts")
    graph.add_edge("extract_facts", "retrieve_context")
    graph.add_edge("retrieve_context", "llm_analyze")
    graph.add_edge("llm_analyze", "check_compliance")
    graph.add_edge("check_compliance", "calculate_risk")
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
        "retrieval_query": "",
        "retrieval_count": 0,
        "risk_flags": [],
        "compliance_results": [],
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
