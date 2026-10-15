"""
Salesforce CRM integration connector.

Supports pushing analysis results, pulling loan applications,
and syncing contacts. Uses simple-salesforce under the hood,
but falls back gracefully if the library isn't installed.
"""

from __future__ import annotations

import logging
import time

from app.integrations.base import IntegrationBase, IntegrationHealth, IntegrationStatus

logger = logging.getLogger(__name__)


class SalesforceConnector(IntegrationBase):
    """Enterprise Salesforce CRM connector."""

    name = "salesforce"
    display_name = "Salesforce CRM"
    description = "Sync loan applications and analysis results with Salesforce"
    icon = "☁️"

    def __init__(self):
        super().__init__()
        self._client = None

    async def connect(self) -> bool:
        """Connect to Salesforce using stored credentials."""
        username = self._config.get("username")
        password = self._config.get("password")
        security_token = self._config.get("security_token", "")
        domain = self._config.get("domain", "login")  # "login" or "test"

        if not username or not password:
            self._last_error = "Missing Salesforce credentials (username/password)"
            logger.warning("Salesforce: %s", self._last_error)
            return False

        try:
            from simple_salesforce import Salesforce as SFClient

            self._client = SFClient(
                username=username,
                password=password,
                security_token=security_token,
                domain=domain,
            )
            # Verify connection by querying org info
            self._client.query("SELECT Id FROM Organization LIMIT 1")
            logger.info("Salesforce: connected as %s", username)
            return True

        except ImportError:
            self._last_error = (
                "simple-salesforce not installed. "
                "Run: pip install simple-salesforce"
            )
            logger.warning("Salesforce: %s", self._last_error)
            return False
        except Exception as e:
            self._last_error = f"Salesforce connection failed: {e}"
            logger.exception("Salesforce: %s", self._last_error)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Salesforce."""
        self._client = None
        logger.info("Salesforce: disconnected")

    async def health_check(self) -> IntegrationHealth:
        """Verify the Salesforce connection is alive."""
        if not self._client:
            return IntegrationHealth(
                status=IntegrationStatus.DISCONNECTED,
                message="Not connected",
            )
        start = time.time()
        try:
            self._client.query("SELECT Id FROM Organization LIMIT 1")
            latency = (time.time() - start) * 1000
            return IntegrationHealth(
                status=IntegrationStatus.CONNECTED,
                latency_ms=latency,
                message="Healthy",
            )
        except Exception as e:
            return IntegrationHealth(
                status=IntegrationStatus.ERROR,
                message=str(e),
            )

    async def sync_data(self, direction: str = "push", **kwargs) -> dict:
        """
        Sync data with Salesforce.

        Push: Export analysis results to Salesforce custom objects.
        Pull: Import loan application records from Salesforce.
        """
        if not self._client:
            return {"error": "Not connected to Salesforce", "records_synced": 0}

        if direction == "push":
            return await self._push_analysis(kwargs.get("analysis_data", {}))
        elif direction == "pull":
            return await self._pull_applications(kwargs.get("query", ""))
        else:
            return {"error": f"Unknown direction: {direction}", "records_synced": 0}

    async def _push_analysis(self, analysis_data: dict) -> dict:
        """Push an analysis result to Salesforce as a custom object."""
        try:
            # Map AnalysisResult to Salesforce fields
            sf_record = {
                "Name": analysis_data.get("filename", "Unknown"),
                "Risk_Score__c": analysis_data.get("overall_risk_score", 0),
                "Risk_Level__c": analysis_data.get("overall_risk_level", "moderate"),
                "Recommendation__c": analysis_data.get("recommendation", "MANUAL_REVIEW"),
                "Summary__c": (analysis_data.get("summary", ""))[:255],
                "Risk_Flags_Count__c": len(analysis_data.get("risk_flags", [])),
                "Processing_Time__c": analysis_data.get("processing_time_seconds", 0),
            }

            result = self._client.Underwriting_Analysis__c.create(sf_record)  # type: ignore
            logger.info("Salesforce: pushed analysis → ID=%s", result.get("id"))
            return {"records_synced": 1, "salesforce_id": result.get("id")}

        except Exception as e:
            logger.exception("Salesforce push failed: %s", e)
            return {"error": str(e), "records_synced": 0}

    async def _pull_applications(self, query: str = "") -> dict:
        """Pull loan application records from Salesforce."""
        try:
            soql = query or (
                "SELECT Id, Name, Loan_Amount__c, Applicant_Credit_Score__c, "
                "Status__c, CreatedDate "
                "FROM Loan_Application__c "
                "WHERE Status__c = 'Pending Review' "
                "ORDER BY CreatedDate DESC LIMIT 50"
            )
            result = self._client.query(soql)  # type: ignore
            records = result.get("records", [])
            logger.info("Salesforce: pulled %d application records", len(records))
            return {"records_synced": len(records), "records": records}

        except Exception as e:
            logger.exception("Salesforce pull failed: %s", e)
            return {"error": str(e), "records_synced": 0}

    async def push_analysis_result(self, analysis_data: dict) -> dict:
        """Convenience method to push a single analysis result."""
        return await self._push_analysis(analysis_data)

    async def create_case(self, analysis_data: dict) -> dict:
        """Create a Salesforce Case for manual-review items."""
        if not self._client:
            return {"error": "Not connected to Salesforce"}

        try:
            case = {
                "Subject": f"Underwriting Review: {analysis_data.get('filename', 'Unknown')}",
                "Description": analysis_data.get("summary", ""),
                "Priority": (
                    "High" if analysis_data.get("overall_risk_score", 0) > 75
                    else "Medium" if analysis_data.get("overall_risk_score", 0) > 50
                    else "Low"
                ),
                "Status": "New",
                "Type": "Underwriting Review",
            }
            result = self._client.Case.create(case)  # type: ignore
            logger.info("Salesforce: created case → ID=%s", result.get("id"))
            return {"case_id": result.get("id")}

        except Exception as e:
            logger.exception("Salesforce create_case failed: %s", e)
            return {"error": str(e)}
