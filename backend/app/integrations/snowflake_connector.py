"""
Snowflake Data Warehouse integration connector.

Supports exporting analysis results, querying historical data,
and bulk data operations. Falls back gracefully if the Snowflake
connector library isn't installed.
"""

from __future__ import annotations

import json
import logging
import time

from app.integrations.base import IntegrationBase, IntegrationHealth, IntegrationStatus

logger = logging.getLogger(__name__)


class SnowflakeConnector(IntegrationBase):
    """Enterprise Snowflake data warehouse connector."""

    name = "snowflake"
    display_name = "Snowflake Data Warehouse"
    description = "Export analyses and query historical underwriting data"
    icon = "❄️"

    def __init__(self):
        super().__init__()
        self._connection = None

    async def connect(self) -> bool:
        """Connect to Snowflake using stored credentials."""
        account = self._config.get("account")
        user = self._config.get("user")
        password = self._config.get("password")
        database = self._config.get("database", "UNDERWRITING")
        warehouse = self._config.get("warehouse", "COMPUTE_WH")
        schema = self._config.get("schema", "PUBLIC")

        if not all([account, user, password]):
            self._last_error = "Missing Snowflake credentials (account/user/password)"
            logger.warning("Snowflake: %s", self._last_error)
            return False

        try:
            import snowflake.connector

            self._connection = snowflake.connector.connect(
                account=account,
                user=user,
                password=password,
                database=database,
                warehouse=warehouse,
                schema=schema,
            )
            # Verify connection
            cursor = self._connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()[0]
            cursor.close()
            logger.info("Snowflake: connected (version=%s)", version)
            return True

        except ImportError:
            self._last_error = (
                "snowflake-connector-python not installed. "
                "Run: pip install snowflake-connector-python"
            )
            logger.warning("Snowflake: %s", self._last_error)
            return False
        except Exception as e:
            self._last_error = f"Snowflake connection failed: {e}"
            logger.exception("Snowflake: %s", self._last_error)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Snowflake."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None
        logger.info("Snowflake: disconnected")

    async def health_check(self) -> IntegrationHealth:
        """Verify the Snowflake connection is alive."""
        if not self._connection:
            return IntegrationHealth(
                status=IntegrationStatus.DISCONNECTED,
                message="Not connected",
            )
        start = time.time()
        try:
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
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
        Sync data with Snowflake.

        Push: Export analysis results to the data warehouse.
        Pull: Query historical underwriting data.
        """
        if not self._connection:
            return {"error": "Not connected to Snowflake", "records_synced": 0}

        if direction == "push":
            return await self._export_analysis(kwargs.get("analysis_data", {}))
        elif direction == "pull":
            return await self._query_historical(kwargs.get("query", ""))
        else:
            return {"error": f"Unknown direction: {direction}", "records_synced": 0}

    async def _export_analysis(self, analysis_data: dict) -> dict:
        """Export an analysis result to Snowflake."""
        try:
            cursor = self._connection.cursor()

            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS underwriting_analyses (
                    id INTEGER AUTOINCREMENT PRIMARY KEY,
                    filename VARCHAR(500),
                    risk_score INTEGER,
                    risk_level VARCHAR(20),
                    recommendation VARCHAR(30),
                    summary TEXT,
                    risk_flags_json TEXT,
                    processing_time FLOAT,
                    exported_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """)

            cursor.execute(
                """
                INSERT INTO underwriting_analyses
                    (filename, risk_score, risk_level, recommendation,
                     summary, risk_flags_json, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    analysis_data.get("filename", "unknown"),
                    analysis_data.get("overall_risk_score", 0),
                    analysis_data.get("overall_risk_level", "moderate"),
                    analysis_data.get("recommendation", "MANUAL_REVIEW"),
                    analysis_data.get("summary", ""),
                    json.dumps(analysis_data.get("risk_flags", [])),
                    analysis_data.get("processing_time_seconds", 0),
                ),
            )
            cursor.close()
            logger.info("Snowflake: exported analysis for '%s'",
                        analysis_data.get("filename"))
            return {"records_synced": 1}

        except Exception as e:
            logger.exception("Snowflake export failed: %s", e)
            return {"error": str(e), "records_synced": 0}

    async def _query_historical(self, query: str = "") -> dict:
        """Query historical underwriting data from Snowflake."""
        try:
            cursor = self._connection.cursor()
            sql = query or (
                "SELECT filename, risk_score, risk_level, recommendation, "
                "exported_at FROM underwriting_analyses "
                "ORDER BY exported_at DESC LIMIT 100"
            )
            cursor.execute(sql)
            columns = [c[0].lower() for c in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            logger.info("Snowflake: queried %d historical records", len(rows))
            return {"records_synced": len(rows), "records": rows}

        except Exception as e:
            logger.exception("Snowflake query failed: %s", e)
            return {"error": str(e), "records_synced": 0}

    async def bulk_export(self, analyses: list[dict]) -> dict:
        """Bulk export multiple analysis results."""
        results = {"total": len(analyses), "success": 0, "failed": 0}
        for analysis in analyses:
            r = await self._export_analysis(analysis)
            if r.get("error"):
                results["failed"] += 1
            else:
                results["success"] += r.get("records_synced", 0)
        return results
