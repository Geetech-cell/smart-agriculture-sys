"""Optional persistence for alerts and health scores."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests


class AlertStoreError(RuntimeError):
    """Raised when alert persistence fails."""


@dataclass
class AlertStoreConfig:
    path: Optional[Path] = None
    webhook_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 8.0

    @classmethod
    def from_env(cls) -> Optional["AlertStoreConfig"]:
        path = os.getenv("ALERT_STORE_PATH")
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        api_key = os.getenv("ALERT_WEBHOOK_API_KEY")
        timeout = float(os.getenv("ALERT_WEBHOOK_TIMEOUT", "8.0"))
        if not path and not webhook_url:
            return None
        return cls(
            path=Path(path) if path else None,
            webhook_url=webhook_url,
            api_key=api_key,
            timeout=timeout,
        )


class AlertStore:
    """Persist alert & health events to file and/or webhook."""

    def __init__(self, config: AlertStoreConfig):
        if not config.path and not config.webhook_url:
            raise AlertStoreError("No alert persistence destination configured.")
        self._config = config
        if self._config.path:
            self._config.path.parent.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        *,
        alerts: Iterable[Dict[str, Any]],
        health_label: str,
        health_score: float,
        context: Dict[str, Any],
        sensor_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        payloads = []
        has_alerts = False
        for alert in alerts:
            has_alerts = True
            payloads.append(
                {
                    "timestamp": timestamp,
                    "health_label": health_label,
                    "health_score": health_score,
                    "context": context,
                    "severity": alert.get("severity"),
                    "title": alert.get("title"),
                    "message": alert.get("message"),
                    "sensor_snapshot": sensor_snapshot,
                }
            )

        if not has_alerts:
            payloads.append(
                {
                    "timestamp": timestamp,
                    "health_label": health_label,
                    "health_score": health_score,
                    "context": context,
                    "severity": "info",
                    "title": "Health snapshot",
                    "message": "No agronomic alerts triggered.",
                    "sensor_snapshot": sensor_snapshot,
                }
            )

        for payload in payloads:
            self._write(payload)

    def _write(self, payload: Dict[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False)
        if self._config.path:
            try:
                with self._config.path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized + "\n")
            except OSError as exc:
                raise AlertStoreError(f"Failed to append alert log: {exc}") from exc

        if self._config.webhook_url:
            headers = {"Content-Type": "application/json"}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"  # nosec B113
            try:
                response = requests.post(
                    self._config.webhook_url,
                    data=serialized.encode("utf-8"),
                    headers=headers,
                    timeout=self._config.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise AlertStoreError(f"Webhook persistence failed: {exc}") from exc


def build_alert_store(
    *,
    path: Optional[str] = None,
    webhook_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Optional[AlertStore]:
    resolved_path = Path(path) if path else None
    resolved_timeout = timeout or float(os.getenv("ALERT_WEBHOOK_TIMEOUT", "8.0"))
    config = AlertStoreConfig(
        path=resolved_path
        if resolved_path
        else (Path(os.getenv("ALERT_STORE_PATH")) if os.getenv("ALERT_STORE_PATH") else None),
        webhook_url=webhook_url or os.getenv("ALERT_WEBHOOK_URL"),
        api_key=api_key or os.getenv("ALERT_WEBHOOK_API_KEY"),
        timeout=resolved_timeout,
    )
    if not config.path and not config.webhook_url:
        return None
    return AlertStore(config)


