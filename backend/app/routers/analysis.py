"""
Analysis router: upload and analyze loan applications, view history, dashboard.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, Query, Request
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.dependencies import get_current_user
from app.models.database import get_db, User, AnalysisRecord
from app.models.schemas import (
    AnalysisResponse,
    AnalysisHistoryItem,
    AnalysisHistoryResponse,
    AnalysisResult,
    DashboardMetrics,
    RiskLevel,
)
from app.services.document_service import extract_text
from app.services.ai_service import analyze_document
from app.services.auth_service import log_audit

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/analysis", tags=["Analysis"])


def _get_active_llm_model() -> str:
    """Return the active model name string for the current LLM provider."""
    from app.config import LLMProvider
    p = settings.LLM_PROVIDER
    if p == LLMProvider.OPENAI:
        return settings.OPENAI_MODEL
    elif p == LLMProvider.AZURE_OPENAI:
        return settings.AZURE_OPENAI_DEPLOYMENT
    elif p == LLMProvider.ANTHROPIC:
        return settings.ANTHROPIC_MODEL
    elif p == LLMProvider.OPENROUTER:
        return settings.OPENROUTER_MODEL
    return settings.OLLAMA_MODEL



@router.post("", response_model=AnalysisResponse, status_code=201)
async def create_analysis(
    request: Request,
    file: UploadFile = File(..., description="Loan application document (PDF, DOCX, or TXT)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and analyze a loan application document.
    Returns a structured risk analysis with scores, flags, and recommendations.
    """
    # 1. Read file
    contents = await file.read()
    filename = file.filename or "unknown"

    logger.info("User %s uploading file: %s (%d bytes)", current_user.email, filename, len(contents))

    # 2. Extract text
    application_text = extract_text(filename, contents)

    # 3. Run AI analysis
    result: AnalysisResult = await analyze_document(application_text, filename=filename)

    # 4. Persist the analysis
    record = AnalysisRecord(
        user_id=current_user.id,
        filename=filename,
        file_size_bytes=len(contents),
        summary=result.summary,
        overall_risk_score=result.overall_risk_score,
        overall_risk_level=result.overall_risk_level.value if isinstance(result.overall_risk_level, RiskLevel) else result.overall_risk_level,
        recommendation=result.recommendation,
        risk_flags=[rf.model_dump() for rf in result.risk_flags],
        detailed_analysis=result.detailed_analysis,
        guidelines_checked=result.guidelines_checked,
        processing_time_seconds=result.processing_time_seconds,
        llm_provider=settings.LLM_PROVIDER.value,
        llm_model=_get_active_llm_model(),
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)

    # 5. Audit log
    await log_audit(
        db=db,
        action="analyze",
        user_id=current_user.id,
        resource=filename,
        details=f"risk_score={result.overall_risk_score}, recommendation={result.recommendation}",
        ip_address=request.client.host if request.client else None,
    )

    logger.info(
        "Analysis saved: id=%d, user=%s, file=%s, score=%d",
        record.id, current_user.email, filename, result.overall_risk_score,
    )

    return AnalysisResponse(
        id=record.id,
        filename=filename,
        analysis=result,
        created_at=record.created_at,
    )


@router.post("/task", status_code=202)
async def create_analysis_task(
    request: Request,
    file: UploadFile = File(..., description="Loan application document (PDF, DOCX, or TXT)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start an analysis as a background task.
    Returns a task_id that can be used to stream Server-Sent Events (SSE).
    """
    from app.services.task_service import create_task, run_analysis_task
    import asyncio

    contents = await file.read()
    filename = file.filename or "unknown"
    application_text = extract_text(filename, contents)

    # Capture values needed in the background closure before request context closes
    user_id = current_user.id
    client_ip = request.client.host if request.client else None

    async def wrapped_analysis(text: str, fname: str):
        result = await analyze_document(text, filename=fname)

        # Use a fresh DB session — the request-scoped session is closed by the time this runs
        from app.models.database import get_session_factory
        session_factory = get_session_factory()
        async with session_factory() as bg_db:
            record = AnalysisRecord(
                user_id=user_id,
                filename=fname,
                file_size_bytes=len(contents),
                summary=result.summary,
                overall_risk_score=result.overall_risk_score,
                overall_risk_level=result.overall_risk_level.value if isinstance(result.overall_risk_level, RiskLevel) else result.overall_risk_level,
                recommendation=result.recommendation,
                risk_flags=[rf.model_dump() for rf in result.risk_flags],
                detailed_analysis=result.detailed_analysis,
                guidelines_checked=result.guidelines_checked,
                processing_time_seconds=result.processing_time_seconds,
                llm_provider=settings.LLM_PROVIDER.value,
                llm_model=_get_active_llm_model(),
            )
            bg_db.add(record)
            await bg_db.commit()
            await bg_db.refresh(record)

            await log_audit(
                db=bg_db,
                action="analyze_task",
                user_id=user_id,
                resource=fname,
                details=f"task_completed_score={result.overall_risk_score}",
                ip_address=client_ip,
            )

            final_data = result.model_dump()
            final_data['analysis_id'] = record.id

        # ResultMock wraps the dict — model_dump must accept 'self' as it's a bound method
        result_mock = type("ResultMock", (), {"model_dump": lambda self, **kw: final_data})()
        return result_mock, {"step": "complete"}

    # 1. Create task in DB
    task_id = await create_task(db, current_user.id, "analysis")

    # 2. Fire and forget — use a fresh session for the background worker
    from app.models.database import get_session_factory
    bg_session_factory = get_session_factory()
    async def _run_with_fresh_session():
        async with bg_session_factory() as bg_db:
            await run_analysis_task(bg_db, task_id, application_text, filename, wrapped_analysis)
    asyncio.create_task(_run_with_fresh_session())

    return {"task_id": task_id, "status": "pending"}


@router.get("/history", response_model=AnalysisHistoryResponse)
async def get_analysis_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated analysis history for the current user."""
    offset = (page - 1) * page_size

    # Total count
    count_result = await db.execute(
        select(func.count(AnalysisRecord.id)).where(AnalysisRecord.user_id == current_user.id)
    )
    total = count_result.scalar() or 0

    # Paginated results
    result = await db.execute(
        select(AnalysisRecord)
        .where(AnalysisRecord.user_id == current_user.id)
        .order_by(desc(AnalysisRecord.created_at))
        .offset(offset)
        .limit(page_size)
    )
    records = result.scalars().all()

    items = [
        AnalysisHistoryItem(
            id=r.id,
            filename=r.filename,
            overall_risk_score=r.overall_risk_score or 0,
            overall_risk_level=r.overall_risk_level or "moderate",
            recommendation=r.recommendation or "MANUAL_REVIEW",
            created_at=r.created_at,
        )
        for r in records
    ]

    return AnalysisHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific analysis by ID."""
    from app.exceptions import ResourceNotFoundError

    result = await db.execute(
        select(AnalysisRecord).where(
            AnalysisRecord.id == analysis_id,
            AnalysisRecord.user_id == current_user.id,
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise ResourceNotFoundError("Analysis")

    from app.models.schemas import RiskFlag
    risk_flags = [RiskFlag(**rf) for rf in (record.risk_flags or [])]

    analysis_result = AnalysisResult(
        summary=record.summary or "",
        overall_risk_score=record.overall_risk_score or 0,
        overall_risk_level=record.overall_risk_level or "moderate",
        risk_flags=risk_flags,
        recommendation=record.recommendation or "MANUAL_REVIEW",
        detailed_analysis=record.detailed_analysis or "",
        guidelines_checked=record.guidelines_checked or 0,
        processing_time_seconds=record.processing_time_seconds or 0.0,
    )

    return AnalysisResponse(
        id=record.id,
        filename=record.filename,
        analysis=analysis_result,
        created_at=record.created_at,
    )


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get dashboard metrics for the current user."""
    # Total analyses
    total_result = await db.execute(
        select(func.count(AnalysisRecord.id)).where(AnalysisRecord.user_id == current_user.id)
    )
    total = total_result.scalar() or 0

    # Analyses today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_result = await db.execute(
        select(func.count(AnalysisRecord.id)).where(
            AnalysisRecord.user_id == current_user.id,
            AnalysisRecord.created_at >= today_start,
        )
    )
    today_count = today_result.scalar() or 0

    # Average risk score
    avg_result = await db.execute(
        select(func.avg(AnalysisRecord.overall_risk_score)).where(
            AnalysisRecord.user_id == current_user.id
        )
    )
    avg_score = round(avg_result.scalar() or 0, 1)

    # Risk distribution
    distribution = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    dist_result = await db.execute(
        select(AnalysisRecord.overall_risk_level, func.count(AnalysisRecord.id))
        .where(AnalysisRecord.user_id == current_user.id)
        .group_by(AnalysisRecord.overall_risk_level)
    )
    for level, count in dist_result.all():
        if level in distribution:
            distribution[level] = count

    # Recent analyses (last 5)
    recent_result = await db.execute(
        select(AnalysisRecord)
        .where(AnalysisRecord.user_id == current_user.id)
        .order_by(desc(AnalysisRecord.created_at))
        .limit(5)
    )
    recent = [
        AnalysisHistoryItem(
            id=r.id,
            filename=r.filename,
            overall_risk_score=r.overall_risk_score or 0,
            overall_risk_level=r.overall_risk_level or "moderate",
            recommendation=r.recommendation or "MANUAL_REVIEW",
            created_at=r.created_at,
        )
        for r in recent_result.scalars().all()
    ]

    return DashboardMetrics(
        total_analyses=total,
        analyses_today=today_count,
        avg_risk_score=avg_score,
        risk_distribution=distribution,
        recent_analyses=recent,
    )
