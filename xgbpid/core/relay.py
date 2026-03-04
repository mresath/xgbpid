"""
Telemetry Relay

Mirrors the stable rolling Parquet snapshot to a Supabase storage bucket in a
background thread, so remote dashboards and outreach audiences can follow the
run without touching the CERN DAQ network.

Credentials are read from a .env file (SUPABASE_URL and SUPABASE_KEY). If the
file is missing or either variable is absent, the relay logs a warning and
silently disables itself — the main acquisition loop is never affected.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_BUCKET = "data"


def _load_env(env_path: Path) -> None:
    """Pull variables from a .env file into os.environ without requiring dotenv."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class TelemetryRelay:
    """Background uploader of the Parquet snapshot to Supabase Storage.

    Call :meth:`tick` from the main event loop on every accepted event.
    The method returns immediately and spawns an upload thread only when the
    configured interval has elapsed, so it adds negligible overhead to the
    < 1 ms hot path.
    """

    def __init__(self, cfg: dict, stable_path: Path) -> None:
        relay_cfg = cfg.get("relay", {})
        self._enabled: bool = bool(relay_cfg.get("enabled", False))
        self._interval_s: float = float(relay_cfg.get("upload_interval_seconds", 30))
        self._stable_path: Path = stable_path
        self._last_upload: float = 0.0
        self._upload_thread: Optional[threading.Thread] = None
        self._client = None
        self._upload_count: int = 0
        self._last_error: Optional[str] = None
        self._status_path: Path = stable_path.parent / "relay_status.json"

        if not self._enabled:
            return

        _load_env(Path(".env"))
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            log.warning(
                "Relay is enabled but SUPABASE_URL / SUPABASE_KEY are not set. "
                "Create a .env file with both variables to activate cloud publishing. "
                "Continuing without relay."
            )
            self._enabled = False
            return

        try:
            from supabase import create_client
            self._client = create_client(url, key)
            log.info("Telemetry relay initialised → Supabase bucket '%s'.", _BUCKET)
        except ImportError:
            log.warning(
                "supabase-py is not installed ('pip install supabase'). Relay disabled."
            )
            self._enabled = False

    def tick(self) -> None:
        """Non-blocking poll — call once per flush cycle in the acquisition loop.

        Spawns an upload thread when the interval has elapsed and no upload is
        already in flight. Returns immediately so DAQ throughput is unaffected.
        """
        if not self._enabled:
            return

        now = time.monotonic()
        if now - self._last_upload < self._interval_s:
            return

        if self._upload_thread and self._upload_thread.is_alive():
            return  # previous upload still running; skip this cycle

        if not self._stable_path.exists():
            return

        self._last_upload = now
        self._upload_thread = threading.Thread(
            target=self._upload_to_cloud,
            args=(self._stable_path,),
            daemon=True,
            name="telemetry-relay",
        )
        self._upload_thread.start()
        log.debug("Relay: upload thread started for '%s'.", self._stable_path)

    def _write_status(self, error: Optional[str] = None) -> None:
        payload = {
            "enabled": self._enabled,
            "configured": self._client is not None,
            "upload_count": self._upload_count,
            "last_upload": time.time() if self._last_upload else 0,
            "interval_s": self._interval_s,
            "last_error": error,
        }
        try:
            tmp = self._status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._status_path)
        except Exception:
            pass

    def _upload_to_cloud(self, file_path: Path) -> None:
        """Upload *file_path* to the Supabase 'data' bucket.

        Always receives the atomically renamed stable snapshot — never the
        .parquet.tmp write buffer — so remote readers get a complete file.
        The upload uses upsert so the remote copy is overwritten each cycle.
        """
        try:
            data = file_path.read_bytes()
            self._client.storage.from_(_BUCKET).upload(
                path="live_telemetry.parquet",
                file=data,
                file_options={"upsert": "true", "content-type": "application/octet-stream"},
            )
            self._upload_count += 1
            self._last_error = None
            log.debug("Relay: uploaded %d bytes to bucket '%s'.", len(data), _BUCKET)
            self._write_status()
        except Exception as exc:
            self._last_error = str(exc)
            log.warning("Relay: upload failed.", exc_info=True)
            self._write_status(error=str(exc))
