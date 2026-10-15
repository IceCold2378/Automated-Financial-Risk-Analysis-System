"""
Azure cloud services integration.

Provides connectors for:
  - Azure Blob Storage  — document upload/download
  - Azure Key Vault     — secrets management
  - Azure Service Bus   — event publishing for async workflows

Each sub-connector is independently usable. The AzureServicesConnector
wraps all three under a single IntegrationBase interface.
"""

from __future__ import annotations

import logging

from app.integrations.base import IntegrationBase, IntegrationHealth, IntegrationStatus

logger = logging.getLogger(__name__)


# ── Blob Storage ──────────────────────────────────────────────────

class AzureBlobClient:
    """Azure Blob Storage operations for document management."""

    def __init__(self, connection_string: str, container_name: str = "underwriting-docs"):
        self._connection_string = connection_string
        self._container_name = container_name
        self._client = None

    def connect(self) -> bool:
        try:
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient.from_connection_string(self._connection_string)
            # Ensure container exists
            try:
                self._client.create_container(self._container_name)
            except Exception:
                pass  # Container may already exist
            logger.info("Azure Blob: connected (container=%s)", self._container_name)
            return True
        except ImportError:
            logger.warning("azure-storage-blob not installed")
            return False
        except Exception as e:
            logger.exception("Azure Blob connection failed: %s", e)
            return False

    async def upload_document(self, filename: str, data: bytes) -> dict:
        """Upload a document to blob storage."""
        if not self._client:
            return {"error": "Blob client not connected"}
        try:
            container = self._client.get_container_client(self._container_name)
            blob = container.get_blob_client(filename)
            blob.upload_blob(data, overwrite=True)
            url = blob.url
            logger.info("Azure Blob: uploaded %s (%d bytes)", filename, len(data))
            return {"url": url, "filename": filename, "size_bytes": len(data)}
        except Exception as e:
            logger.exception("Azure Blob upload failed: %s", e)
            return {"error": str(e)}

    async def download_document(self, filename: str) -> bytes | None:
        """Download a document from blob storage."""
        if not self._client:
            return None
        try:
            container = self._client.get_container_client(self._container_name)
            blob = container.get_blob_client(filename)
            data = blob.download_blob().readall()
            logger.info("Azure Blob: downloaded %s (%d bytes)", filename, len(data))
            return data
        except Exception as e:
            logger.exception("Azure Blob download failed: %s", e)
            return None

    async def list_documents(self, prefix: str = "") -> list[dict]:
        """List documents in the container."""
        if not self._client:
            return []
        try:
            container = self._client.get_container_client(self._container_name)
            blobs = container.list_blobs(name_starts_with=prefix)
            return [
                {"name": b.name, "size": b.size, "last_modified": str(b.last_modified)}
                for b in blobs
            ]
        except Exception as e:
            logger.exception("Azure Blob list failed: %s", e)
            return []


# ── Key Vault ─────────────────────────────────────────────────────

class AzureKeyVaultClient:
    """Azure Key Vault for secrets management."""

    def __init__(self, vault_url: str):
        self._vault_url = vault_url
        self._client = None

    def connect(self) -> bool:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            credential = DefaultAzureCredential()
            self._client = SecretClient(vault_url=self._vault_url, credential=credential)
            logger.info("Azure Key Vault: connected (%s)", self._vault_url)
            return True
        except ImportError:
            logger.warning("azure-identity or azure-keyvault-secrets not installed")
            return False
        except Exception as e:
            logger.exception("Azure Key Vault connection failed: %s", e)
            return False

    async def get_secret(self, name: str) -> str | None:
        """Retrieve a secret value."""
        if not self._client:
            return None
        try:
            secret = self._client.get_secret(name)
            return secret.value
        except Exception as e:
            logger.warning("Key Vault: failed to get secret '%s': %s", name, e)
            return None

    async def set_secret(self, name: str, value: str) -> bool:
        """Store a secret."""
        if not self._client:
            return False
        try:
            self._client.set_secret(name, value)
            logger.info("Key Vault: stored secret '%s'", name)
            return True
        except Exception as e:
            logger.exception("Key Vault: failed to set secret '%s': %s", name, e)
            return False


# ── Service Bus ───────────────────────────────────────────────────

class AzureServiceBusClient:
    """Azure Service Bus for event-driven communication."""

    def __init__(self, connection_string: str, queue_name: str = "underwriting-events"):
        self._connection_string = connection_string
        self._queue_name = queue_name
        self._client = None

    def connect(self) -> bool:
        try:
            from azure.servicebus import ServiceBusClient as SBClient
            self._client = SBClient.from_connection_string(self._connection_string)
            logger.info("Azure Service Bus: connected (queue=%s)", self._queue_name)
            return True
        except ImportError:
            logger.warning("azure-servicebus not installed")
            return False
        except Exception as e:
            logger.exception("Azure Service Bus connection failed: %s", e)
            return False

    async def publish_event(self, event_type: str, payload: dict) -> bool:
        """Publish an event to the Service Bus queue."""
        if not self._client:
            return False
        try:
            import json
            from azure.servicebus import ServiceBusMessage
            sender = self._client.get_queue_sender(queue_name=self._queue_name)
            message = ServiceBusMessage(
                json.dumps({"event_type": event_type, "payload": payload}),
                content_type="application/json",
            )
            with sender:
                sender.send_messages(message)
            logger.info("Service Bus: published '%s' event", event_type)
            return True
        except Exception as e:
            logger.exception("Service Bus publish failed: %s", e)
            return False

    def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None


# ── Unified Connector ─────────────────────────────────────────────

class AzureServicesConnector(IntegrationBase):
    """
    Unified Azure services connector wrapping Blob Storage,
    Key Vault, and Service Bus under a single IntegrationBase.
    """

    name = "azure"
    display_name = "Microsoft Azure"
    description = "Azure Blob Storage, Key Vault, and Service Bus"
    icon = "🔷"

    def __init__(self):
        super().__init__()
        self.blob: AzureBlobClient | None = None
        self.keyvault: AzureKeyVaultClient | None = None
        self.servicebus: AzureServiceBusClient | None = None
        self._sub_status: dict[str, bool] = {}

    async def connect(self) -> bool:
        """Connect all configured Azure sub-services."""
        connected_any = False

        # Blob Storage
        conn_str = self._config.get("storage_connection_string", "")
        container = self._config.get("storage_container", "underwriting-docs")
        if conn_str:
            self.blob = AzureBlobClient(conn_str, container)
            self._sub_status["blob"] = self.blob.connect()
            connected_any = connected_any or self._sub_status["blob"]

        # Key Vault
        vault_url = self._config.get("keyvault_url", "")
        if vault_url:
            self.keyvault = AzureKeyVaultClient(vault_url)
            self._sub_status["keyvault"] = self.keyvault.connect()
            connected_any = connected_any or self._sub_status["keyvault"]

        # Service Bus
        sb_conn_str = self._config.get("servicebus_connection_string", "")
        queue = self._config.get("servicebus_queue", "underwriting-events")
        if sb_conn_str:
            self.servicebus = AzureServiceBusClient(sb_conn_str, queue)
            self._sub_status["servicebus"] = self.servicebus.connect()
            connected_any = connected_any or self._sub_status["servicebus"]

        if not connected_any:
            self._last_error = "No Azure services configured or all failed to connect"
            logger.warning("Azure: %s", self._last_error)

        return connected_any

    async def disconnect(self) -> None:
        """Disconnect all Azure sub-services."""
        if self.servicebus:
            self.servicebus.disconnect()
        self.blob = None
        self.keyvault = None
        self.servicebus = None
        self._sub_status = {}
        logger.info("Azure: all sub-services disconnected")

    async def health_check(self) -> IntegrationHealth:
        """Check health of all Azure sub-services."""
        if not self._sub_status:
            return IntegrationHealth(
                status=IntegrationStatus.DISCONNECTED,
                message="No sub-services connected",
            )

        healthy = sum(1 for v in self._sub_status.values() if v)
        total = len(self._sub_status)
        status = (
            IntegrationStatus.CONNECTED if healthy == total
            else IntegrationStatus.ERROR if healthy == 0
            else IntegrationStatus.CONNECTED  # partially connected
        )
        return IntegrationHealth(
            status=status,
            message=f"{healthy}/{total} sub-services connected: {self._sub_status}",
        )

    async def sync_data(self, direction: str = "push", **kwargs) -> dict:
        """Sync via Blob Storage (documents) or Service Bus (events)."""
        if direction == "push" and self.servicebus:
            success = await self.servicebus.publish_event(
                event_type=kwargs.get("event_type", "analysis_complete"),
                payload=kwargs.get("payload", {}),
            )
            return {"published": success}
        return {"error": "No suitable Azure service for this operation"}

    def get_info(self) -> dict:
        """Extended info including sub-service status."""
        info = super().get_info()
        info["sub_services"] = self._sub_status
        return info
