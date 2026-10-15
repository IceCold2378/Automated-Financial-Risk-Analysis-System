"""
Intelligent Underwriting Assistant — FastAPI Application
Commercial-grade entry point with structured logging, CORS, and API versioning.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.exceptions import AppException, app_exception_handler, generic_exception_handler
from app.middleware.logging import RequestLoggingMiddleware
from app.services.vector_service import build_vector_db
from app.services.ai_service import build_rag_chain

settings = get_settings()


# ── Logging Configuration ─────────────────────────────────────────

def setup_logging():
    """Configure structured logging for the application."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("  %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("  Environment: %s", settings.ENVIRONMENT.value)
    logger.info("  LLM Provider: %s", settings.LLM_PROVIDER.value)
    logger.info("=" * 60)

    # 1. Database schema is now managed by Alembic
    logger.info("Initializing database (schema via Alembic)...")
    logger.info("Database ready.")

    # 2. Build vector database
    logger.info("Building vector database from guidelines...")
    try:
        retriever = build_vector_db()
        logger.info("Vector database ready.")
    except FileNotFoundError as e:
        logger.error("Guidelines file not found: %s", e)
        logger.warning("Starting without RAG — analysis will not work until guidelines are loaded.")
        retriever = None
    except Exception as e:
        logger.error("Failed to build vector DB: %s", e)
        retriever = None

    # 3. Build RAG chain
    if retriever:
        try:
            build_rag_chain(retriever)
            logger.info("RAG chain ready.")
        except Exception as e:
            logger.error("Failed to build RAG chain: %s", e)

    # 4. Register enterprise integrations
    logger.info("Registering enterprise integrations...")
    _register_integrations()
    logger.info("Integrations registered.")

    logger.info("Startup complete. API available at %s/", settings.API_PREFIX)

    yield  # App is running

    # Shutdown: disconnect integrations
    from app.integrations.base import integration_registry
    await integration_registry.disconnect_all()
    logger.info("Shutting down...")


# ── Create App ─────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered loan application risk analysis platform.",
    version=settings.APP_VERSION,
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)


from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from app.middleware.rate_limit import get_limiter

# ── Exception Handlers ─────────────────────────────────────────────

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ── Rate Limiting ──────────────────────────────────────────────────

app.state.limiter = get_limiter()
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routers ────────────────────────────────────────────────────────

from app.routers import health, auth, analysis, integrations, admin, streaming

app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(analysis.router, prefix=settings.API_PREFIX)
app.include_router(integrations.router, prefix=settings.API_PREFIX)
app.include_router(admin.router, prefix=settings.API_PREFIX)
app.include_router(streaming.router, prefix=settings.API_PREFIX)


# ── Integration Registration ───────────────────────────────────────

def _register_integrations():
    """Register all enterprise integration connectors."""
    from app.integrations.base import integration_registry
    from app.integrations.salesforce import SalesforceConnector
    from app.integrations.snowflake_connector import SnowflakeConnector
    from app.integrations.azure_services import AzureServicesConnector

    # Register connectors (they start disconnected)
    integration_registry.register(SalesforceConnector())
    integration_registry.register(SnowflakeConnector())
    integration_registry.register(AzureServicesConnector())


# ── Root ───────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
    }


# ── Entry Point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
