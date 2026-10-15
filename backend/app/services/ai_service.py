"""
AI analysis service: LLM orchestration with agentic pipeline.

This service now delegates to the LangGraph agent for multi-step
analysis, while keeping the RAG chain as a fallback.

Compatible with LangChain v1.0+ (uses LCEL — no langchain.chains).
"""

import json
import time
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.config import get_settings, LLMProvider
from app.models.schemas import AnalysisResult, RiskFlag, RiskLevel
from app.exceptions import AIServiceError

logger = logging.getLogger(__name__)
settings = get_settings()

_rag_chain = None
_analysis_chain = None


# ── Prompt Template ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert underwriting risk analyst at a major financial institution.
Your task is to perform a thorough risk analysis of a loan application against the provided
underwriting guidelines. You must be precise, objective, and thorough.

UNDERWRITING GUIDELINE EXCERPTS (each headed by its citation label):
{context}

LOAN APPLICATION:
{input}

Analyze the application carefully and respond with ONLY a valid JSON object (no other text) in this exact format:

{{
    "summary": "Brief 2-3 sentence summary of the loan application",
    "overall_risk_score": <integer 0-100 where 0=no risk, 100=maximum risk>,
    "overall_risk_level": "<one of: low, moderate, high, critical>",
    "risk_flags": [
        {{
            "category": "<risk category name>",
            "description": "<specific description of the risk>",
            "severity": "<one of: low, moderate, high, critical>",
            "guideline_reference": "<copy the citation label of the excerpt that supports this flag, e.g. HUD Handbook 4000.1, p.192>",
            "confidence": <float 0.0 to 1.0>
        }}
    ],
    "recommendation": "<one of: APPROVE, DENY, MANUAL_REVIEW>",
    "detailed_analysis": "Comprehensive paragraph explaining the full analysis and reasoning",
    "guidelines_checked": <integer count of guidelines evaluated>
}}

IMPORTANT RULES:
- Respond ONLY with the JSON object, no markdown, no explanation before or after
- Base every flag ONLY on the excerpts above; cite one excerpt's page per flag using its exact label
- If no excerpt supports a concern, still flag it but leave guideline_reference empty
- If information is missing from the application, flag it as a risk
- Be conservative — when in doubt, flag for manual review
- Risk score guidelines: 0-25=low, 26-50=moderate, 51-75=high, 76-100=critical
"""


def _get_llm():
    """Get the LLM based on the configured provider."""
    provider = settings.LLM_PROVIDER

    if provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Using OpenAI provider: model=%s", settings.OPENAI_MODEL)
            return ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                temperature=0.1,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, falling back to Ollama")

    elif provider == LLMProvider.AZURE_OPENAI:
        try:
            from langchain_openai import AzureChatOpenAI
            logger.info("Using Azure OpenAI provider: deployment=%s", settings.AZURE_OPENAI_DEPLOYMENT)
            return AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                temperature=0.1,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, falling back to Ollama")

    elif provider == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
            logger.info("Using Anthropic provider: model=%s", settings.ANTHROPIC_MODEL)
            return ChatAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.ANTHROPIC_MODEL,
                temperature=0.1,
                max_tokens=4096,
            )
        except ImportError:
            logger.warning("langchain-anthropic not installed, falling back to Ollama")

    elif provider == LLMProvider.OPENROUTER:
        try:
            from langchain_openai import ChatOpenAI
            model_name = settings.OPENROUTER_MODEL or "google/gemini-2.5-flash"
            logger.info("Using OpenRouter provider: model=%s", model_name)
            return ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=model_name,
                temperature=0.1,
                max_tokens=2048,
                default_headers={
                    "HTTP-Referer": "https://underwriting-assistant.app",
                    "X-Title": settings.APP_NAME,
                },
            )
        except Exception as e:
            logger.warning("OpenRouter LLM initialization failed (%s), falling back to Ollama", e)

    # Default: Ollama
    logger.info("Using Ollama provider: model=%s, host=%s", settings.OLLAMA_MODEL, settings.OLLAMA_HOST)
    from langchain_ollama import OllamaLLM
    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_HOST,
        temperature=0.1,
    )


def _format_docs(docs) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """
    Build the RAG chain using LCEL (LangChain Expression Language).
    Compatible with LangChain v1.0+ (no langchain.chains dependency).
    """
    global _rag_chain

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT)
    ])

    # LCEL pipeline: retrieve → format docs → prompt → llm → parse string
    _rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    model_name = _get_active_model_name()
    logger.info("RAG chain built successfully (provider=%s, model=%s)",
                settings.LLM_PROVIDER.value, model_name)
    return _rag_chain


def get_analysis_chain():
    """Prompt | LLM | parser chain that takes {context, input}; context is built by the agent."""
    global _analysis_chain
    if _analysis_chain is None:
        _analysis_chain = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT)]) | _get_llm() | StrOutputParser()
    return _analysis_chain


def get_rag_chain():
    """Get the current RAG chain."""
    if _rag_chain is None:
        raise AIServiceError("RAG chain not initialized. Server may still be starting up.")
    return _rag_chain


def _get_active_model_name() -> str:
    """Return the active model name for logging."""
    provider = settings.LLM_PROVIDER
    if provider == LLMProvider.OPENAI:
        return settings.OPENAI_MODEL
    elif provider == LLMProvider.AZURE_OPENAI:
        return settings.AZURE_OPENAI_DEPLOYMENT
    elif provider == LLMProvider.ANTHROPIC:
        return settings.ANTHROPIC_MODEL
    elif provider == LLMProvider.OPENROUTER:
        return settings.OPENROUTER_MODEL
    return settings.OLLAMA_MODEL


def _parse_llm_response(raw_response: str) -> AnalysisResult:
    """Parse the LLM's JSON response into a structured AnalysisResult."""
    text = raw_response.strip()

    # Remove markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON. Falling back to unstructured result.")
        return AnalysisResult(
            summary="Analysis completed (unstructured output)",
            overall_risk_score=50,
            overall_risk_level=RiskLevel.MODERATE,
            risk_flags=[],
            recommendation="MANUAL_REVIEW",
            detailed_analysis=raw_response,
            guidelines_checked=0,
        )

    risk_flags = []
    for flag_data in data.get("risk_flags", []):
        try:
            risk_flags.append(RiskFlag(
                category=flag_data.get("category", "Unknown"),
                description=flag_data.get("description", ""),
                severity=flag_data.get("severity", "moderate"),
                guideline_reference=flag_data.get("guideline_reference", ""),
                confidence=min(1.0, max(0.0, float(flag_data.get("confidence", 0.5)))),
            ))
        except Exception as e:
            logger.warning("Failed to parse risk flag: %s", e)

    return AnalysisResult(
        summary=data.get("summary", ""),
        overall_risk_score=min(100, max(0, int(data.get("overall_risk_score", 50)))),
        overall_risk_level=data.get("overall_risk_level", "moderate"),
        risk_flags=risk_flags,
        recommendation=data.get("recommendation", "MANUAL_REVIEW"),
        detailed_analysis=data.get("detailed_analysis", ""),
        guidelines_checked=int(data.get("guidelines_checked", 0)),
    )


async def analyze_document(application_text: str, filename: str = "unknown") -> AnalysisResult:
    """
    Run the full analysis pipeline on the application text.
    Delegates to the agentic pipeline, with RAG-only fallback.

    Returns a structured AnalysisResult.
    """
    start_time = time.time()

    # ── Try agentic pipeline first ──
    if settings.AGENT_ENABLE_TOOLS:
        try:
            from app.agent.orchestrator import run_agent_analysis
            from app.agent.state import AgentConfig

            agent_config = AgentConfig(
                max_iterations=settings.AGENT_MAX_ITERATIONS,
                enable_tools=settings.AGENT_ENABLE_TOOLS,
                verbose=settings.AGENT_VERBOSE,
            )

            logger.info("Starting agentic analysis for '%s' (%d chars)...",
                        filename, len(application_text))

            result, trace = await run_agent_analysis(
                document_text=application_text,
                filename=filename,
                config=agent_config,
            )

            logger.info(
                "Agentic analysis complete: score=%d, level=%s, rec=%s, steps=%d, time=%.2fs",
                result.overall_risk_score, result.overall_risk_level,
                result.recommendation,
                len(trace.get("steps", [])),
                result.processing_time_seconds,
            )
            return result

        except Exception as e:
            logger.warning("Agentic pipeline failed, falling back to direct RAG: %s", e)

    # ── Fallback: direct RAG chain ──
    chain = get_rag_chain()

    try:
        logger.info("Starting direct RAG analysis (text length=%d chars)...", len(application_text))
        # LCEL chain now returns a string directly (StrOutputParser)
        raw_answer = chain.invoke(application_text)

        result = _parse_llm_response(raw_answer)
        result.processing_time_seconds = round(time.time() - start_time, 2)

        logger.info(
            "RAG analysis complete: risk_score=%d, risk_level=%s, recommendation=%s, time=%.2fs",
            result.overall_risk_score, result.overall_risk_level,
            result.recommendation, result.processing_time_seconds
        )
        return result

    except AIServiceError:
        raise
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        logger.exception("AI analysis failed after %.2fs: %s", elapsed, e)
        raise AIServiceError(f"Analysis failed: {str(e)}")
