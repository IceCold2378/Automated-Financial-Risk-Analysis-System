"""
Tests for Admin endpoints and API Keys feature.
"""

import pytest
from httpx import AsyncClient

from app.models.database import ApiKey

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def admin_token(db_session, create_test_user, create_jwt_token):
    # Demote default user initially to insert an admin
    user = await create_test_user(db_session, {"email": "admin@test.com", "role": "admin"})
    return create_jwt_token(user.id, "admin@test.com", "admin")


async def test_get_system_metrics(client: AsyncClient, admin_token: str):
    response = await client.get("/api/v1/admin/system/metrics", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total_users"] > 0
    assert "db_size_mb" in metrics


async def test_list_users(client: AsyncClient, admin_token: str):
    response = await client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    users = response.json()
    assert len(users) > 0
    assert any(u["email"] == "admin@test.com" for u in users)


async def test_update_user_role(client: AsyncClient, admin_token: str, db_session, create_test_user):
    user = await create_test_user(db_session, {"email": "analyst2@test.com", "role": "analyst"})
    
    response = await client.put(
        f"/api/v1/admin/users/{user.id}",
        json={"role": "reviewer"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    
    # Verify in DB
    await db_session.refresh(user)
    assert user.role == "reviewer"


async def test_generate_api_key(client: AsyncClient, admin_token: str):
    response = await client.post(
        "/api/v1/admin/api-keys",
        json={"name": "Test Key", "scopes": ["read"]},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Key"
    assert "api_key" in data  # Must contain raw key
    assert data["api_key"].startswith("sk_live_")


async def test_delete_api_key(client: AsyncClient, admin_token: str, db_session):
    # First create a key
    gen_response = await client.post(
        "/api/v1/admin/api-keys",
        json={"name": "To Delete"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    key_id = gen_response.json()["id"]
    
    # Act
    del_response = await client.delete(
        f"/api/v1/admin/api-keys/{key_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert del_response.status_code == 200
    
    # Assert DB is inactive
    db_key = await db_session.get(ApiKey, key_id)
    assert db_key.is_active == 0


async def test_admin_access_denied_for_analyst(client: AsyncClient, token: str):
    # Token from default fixture is an analyst
    response = await client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Access requires one of: ['admin']"
