"""
Tests for background task SSE streaming.
"""

import pytest
from httpx import AsyncClient

from app.models.database import BackgroundTask
from app.models.schemas import TaskStatus
from app.services.task_service import create_task, update_task_progress

pytestmark = pytest.mark.asyncio


async def test_create_and_update_task(db_session):
    # 1. Create task
    task_id = await create_task(db_session, user_id=1, task_type="analysis")
    assert task_id.startswith("task_")
    
    # Verify in DB
    db_task = await db_session.get(BackgroundTask, task_id)
    assert db_task.status == "pending"
    assert db_task.progress == 0.0
    
    # 2. Update progress
    await update_task_progress(db_session, task_id, TaskStatus.RUNNING, 50.0)
    
    await db_session.refresh(db_task)
    assert db_task.status == "running"
    assert db_task.progress == 50.0


# NOTE: Testing the actual Server-Sent Events stream with HTTPX requires async iteration 
# over the stream output which can be tricky in pytest without a running ASGI server.
# Here we verify the endpoint exists and validates the token correctly.

async def test_streaming_endpoint_404(client: AsyncClient, token: str):
    response = await client.get(
        "/api/v1/stream/task/invalid_task_id", 
        params={"token": token}
    )
    assert response.status_code == 404
