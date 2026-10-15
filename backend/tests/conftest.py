"""
Pytest fixtures for backend tests.
"""

import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.config import Settings, get_settings
from app.models.database import Base, get_engine, get_session_factory


def get_test_settings():
    """Override settings for testing."""
    return Settings(
        DATABASE_URL="sqlite+aiosqlite:///./test_underwriting.db",
        DEBUG=False,
        SECRET_KEY="test-secret-key-do-not-use-in-production",
        CHROMA_PERSIST_DIR="",  # In-memory for tests
    )


@pytest_asyncio.fixture(scope="function")
async def app():
    """Create a test app with a fresh database."""
    from app.main import app as fastapi_app

    # Override settings
    fastapi_app.dependency_overrides[get_settings] = get_test_settings

    # Clear the global engine so it gets recreated with test settings
    import app.models.database as db_mod
    db_mod._engine = None
    db_mod._async_session_factory = None

    # Create tables
    settings = get_test_settings()
    engine = get_engine()
    
    # Needs to be inside the test loop to avoid connection issues
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Note: We don't override get_db here because the db_session fixture 
    # will do it to ensure tests and endpoints share the SAME connection line.
    
    yield fastapi_app

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(app):
    """HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_token(client):
    """Register a test user and return the JWT token."""
    response = await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
        "organization": "Test Org",
    })
    assert response.status_code == 201
    return response.json()["access_token"]


@pytest_asyncio.fixture
async def create_test_user():
    """Factory fixture to create test users."""
    from app.services.auth_service import hash_password
    from app.models.database import User

    async def _create(db_session, overrides=None):
        data = {
            "email": "test@example.com",
            "hashed_password": hash_password("testpassword123"),
            "full_name": "Test User",
            "role": "analyst",
            "is_active": 1
        }
        if overrides:
            data.update(overrides)
            
        user = User(**data)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        return user
        
    return _create


@pytest_asyncio.fixture
def create_jwt_token():
    """Factory fixture to create JWT tokens."""
    from app.services.auth_service import create_access_token
    
    def _create(user_id: int, email: str, role: str):
        return create_access_token(
            user_id=user_id,
            email=email,
            role=role
        )
        
    return _create


@pytest_asyncio.fixture
def auth_headers(auth_token):
    """Authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture
async def db_session(app):
    """Database session fixture."""
    from app.models.database import get_db
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        # Override get_db to return this specific session
        async def override_get_db():
            yield session
            
        app.dependency_overrides[get_db] = override_get_db
        yield session
        # Clear override after test
        app.dependency_overrides.pop(get_db, None)

@pytest_asyncio.fixture
async def token(client, db_session, create_test_user, create_jwt_token):
    # Demote default user initially to insert an admin
    user = await create_test_user(db_session, {"email": "analyst@test.com", "role": "analyst"})
    return create_jwt_token(user.id, "analyst@test.com", "analyst")
