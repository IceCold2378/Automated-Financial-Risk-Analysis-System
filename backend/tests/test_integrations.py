"""
Tests for the integration layer.

Tests the integration registry, connector base classes, webhook manager,
and the integrations API endpoints. All external services are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.integrations.base import (
    IntegrationBase,
    IntegrationHealth,
    IntegrationRegistry,
    IntegrationStatus,
)
from app.integrations.webhooks import (
    WebhookEndpoint,
    WebhookEvent,
    WebhookManager,
)


# ══════════════════════════════════════════════════════════════════
#  Mock Connector for Testing
# ══════════════════════════════════════════════════════════════════

class MockConnector(IntegrationBase):
    """A simple in-memory connector for testing."""
    name = "mock"
    display_name = "Mock Integration"
    description = "Test-only connector"
    icon = "🧪"

    def __init__(self, should_fail: bool = False):
        super().__init__()
        self._should_fail = should_fail

    async def connect(self) -> bool:
        if self._should_fail:
            raise ConnectionError("Mock connection failure")
        return True

    async def disconnect(self) -> None:
        pass

    async def health_check(self) -> IntegrationHealth:
        return IntegrationHealth(
            status=IntegrationStatus.CONNECTED,
            latency_ms=5.0,
            message="Mock healthy",
        )

    async def sync_data(self, direction: str = "push", **kwargs) -> dict:
        return {"records_synced": 1, "direction": direction}


# ══════════════════════════════════════════════════════════════════
#  Registry Tests
# ══════════════════════════════════════════════════════════════════

class TestIntegrationRegistry:
    """Test the integration registry lifecycle."""

    def test_register_and_list(self):
        registry = IntegrationRegistry()
        connector = MockConnector()
        registry.register(connector)

        integrations = registry.list_all()
        assert len(integrations) == 1
        assert integrations[0]["name"] == "mock"
        assert integrations[0]["status"] == "disconnected"

    def test_get_by_name(self):
        registry = IntegrationRegistry()
        connector = MockConnector()
        registry.register(connector)

        result = registry.get("mock")
        assert result is connector
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_connect_success(self):
        registry = IntegrationRegistry()
        connector = MockConnector()
        registry.register(connector)

        result = await registry.connect("mock")
        assert result is True
        assert connector.status == IntegrationStatus.CONNECTED
        assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        registry = IntegrationRegistry()
        connector = MockConnector(should_fail=True)
        registry.register(connector)

        result = await registry.connect("mock")
        assert result is False
        assert connector.status == IntegrationStatus.ERROR
        assert connector.last_error is not None

    @pytest.mark.asyncio
    async def test_connect_nonexistent(self):
        registry = IntegrationRegistry()
        result = await registry.connect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        registry = IntegrationRegistry()
        connector = MockConnector()
        registry.register(connector)

        await registry.connect("mock")
        assert connector.is_connected is True

        await registry.disconnect("mock")
        assert connector.status == IntegrationStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        registry = IntegrationRegistry()
        connector = MockConnector()
        registry.register(connector)
        await registry.connect("mock")

        results = await registry.health_check_all()
        assert "mock" in results
        assert results["mock"]["status"] == "connected"

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        registry = IntegrationRegistry()
        connector = MockConnector()
        registry.register(connector)

        results = await registry.health_check_all()
        assert results["mock"]["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        registry = IntegrationRegistry()
        registry.register(MockConnector())
        await registry.connect("mock")
        await registry.disconnect_all()

        connector = registry.get("mock")
        assert connector.status == IntegrationStatus.DISCONNECTED


# ══════════════════════════════════════════════════════════════════
#  Connector Info Tests
# ══════════════════════════════════════════════════════════════════

class TestConnectorInfo:
    """Test connector metadata and info serialization."""

    def test_get_info(self):
        connector = MockConnector()
        info = connector.get_info()
        assert info["name"] == "mock"
        assert info["display_name"] == "Mock Integration"
        assert info["status"] == "disconnected"
        assert info["configured"] is False

    def test_configure(self):
        connector = MockConnector()
        connector.configure({"key": "value"})
        info = connector.get_info()
        assert info["configured"] is True


# ══════════════════════════════════════════════════════════════════
#  Webhook Manager Tests
# ══════════════════════════════════════════════════════════════════

class TestWebhookManager:
    """Test webhook registration, delivery, and lifecycle."""

    def _make_endpoint(self, **overrides) -> WebhookEndpoint:
        defaults = {
            "id": "wh_test_001",
            "url": "https://example.com/webhook",
            "events": [WebhookEvent.ANALYSIS_COMPLETE],
            "secret": "test-secret-key",
            "user_id": 1,
        }
        defaults.update(overrides)
        return WebhookEndpoint(**defaults)

    def test_register_and_list(self):
        manager = WebhookManager()
        ep = self._make_endpoint()
        manager.register(ep)

        webhooks = manager.list_all()
        assert len(webhooks) == 1
        assert webhooks[0]["id"] == "wh_test_001"

    def test_list_by_user(self):
        manager = WebhookManager()
        manager.register(self._make_endpoint(id="wh_1", user_id=1))
        manager.register(self._make_endpoint(id="wh_2", user_id=2))

        user1 = manager.list_all(user_id=1)
        assert len(user1) == 1
        assert user1[0]["id"] == "wh_1"

    def test_unregister(self):
        manager = WebhookManager()
        ep = self._make_endpoint()
        manager.register(ep)
        assert manager.unregister("wh_test_001") is True
        assert manager.unregister("nonexistent") is False
        assert len(manager.list_all()) == 0

    def test_update(self):
        manager = WebhookManager()
        ep = self._make_endpoint()
        manager.register(ep)
        assert manager.update("wh_test_001", active=False) is True
        assert ep.active is False

    @pytest.mark.asyncio
    async def test_trigger_no_matching(self):
        manager = WebhookManager()
        deliveries = await manager.trigger("no.such.event", {})
        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_trigger_with_matching(self):
        manager = WebhookManager()
        ep = self._make_endpoint()
        manager.register(ep)

        # Mock the HTTP call
        with patch("app.integrations.webhooks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            deliveries = await manager.trigger(
                WebhookEvent.ANALYSIS_COMPLETE,
                {"risk_score": 42},
            )

        assert len(deliveries) == 1
        assert deliveries[0].success is True
        assert deliveries[0].status_code == 200

    @pytest.mark.asyncio
    async def test_trigger_inactive_skipped(self):
        manager = WebhookManager()
        ep = self._make_endpoint(active=False)
        manager.register(ep)

        deliveries = await manager.trigger(WebhookEvent.ANALYSIS_COMPLETE, {})
        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_delivery_retry_on_failure(self):
        manager = WebhookManager()
        ep = self._make_endpoint(max_retries=2)
        manager.register(ep)

        with patch("app.integrations.webhooks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=500)
            deliveries = await manager.trigger(
                WebhookEvent.ANALYSIS_COMPLETE,
                {"test": True},
            )

        assert len(deliveries) == 1
        assert deliveries[0].success is False
        # Should have retried (mock was called max_retries times)
        assert mock_post.call_count == 2


# ══════════════════════════════════════════════════════════════════
#  Connector Import Tests (verify they exist and are importable)
# ══════════════════════════════════════════════════════════════════

class TestConnectorImports:
    """Verify all connectors can be imported without errors."""

    def test_salesforce_importable(self):
        from app.integrations.salesforce import SalesforceConnector
        c = SalesforceConnector()
        assert c.name == "salesforce"
        assert c.status == IntegrationStatus.DISCONNECTED

    def test_snowflake_importable(self):
        from app.integrations.snowflake_connector import SnowflakeConnector
        c = SnowflakeConnector()
        assert c.name == "snowflake"

    def test_azure_importable(self):
        from app.integrations.azure_services import AzureServicesConnector
        c = AzureServicesConnector()
        assert c.name == "azure"
        info = c.get_info()
        assert "sub_services" in info
