"""Alert manager: alert state, de-duplication, and non-blocking dispatch.

Notifiers (Telegram, webhook) run on a single background thread so a slow or
unreachable endpoint can never delay the inspection pipeline.  Both notifiers
are inert unless configured through environment variables / config (no real
side effects in tests or the demo).
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Notifier = Callable[["Alert"], None]

_HISTORY_LIMIT = 500


@dataclass
class Alert:
    timestamp: str
    billet_id: str
    status: str
    reasons: List[str]
    image_path: Optional[str] = None
    kind: str = "billet"      # billet | camera | system | warning
    sound: bool = True         # whether the dashboard should beep
    alert_id: int = 0
    latency_ms: Optional[float] = None  # decision -> alert raised

    def message(self) -> str:
        """One-line human-readable summary used by Telegram / webhook."""
        head = f"[{self.status}] {self.billet_id}" if self.kind == "billet" else f"[{self.kind.upper()}] {self.status}"
        return head + (": " + "; ".join(self.reasons) if self.reasons else "")

    def to_payload(self) -> Dict[str, Any]:
        """JSON-serialisable dict for webhook delivery."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "billet_id": self.billet_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "image_path": self.image_path,
            "message": self.message(),
        }


class AlertManager:
    """Manages active and historical alerts and fans them out to notifiers."""

    def __init__(
        self,
        notifiers: Optional[List[Notifier]] = None,
        debounce_s: float = 10.0,
    ) -> None:
        self.history: List[Alert] = []
        self.active_alert: Optional[Alert] = None
        self.notifiers: List[Notifier] = list(notifiers or [])
        self.debounce_s = debounce_s

        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._recent: Dict[tuple, float] = {}   # (kind, status, reasons) -> monotonic time
        self._queue: "queue.Queue[Optional[Alert]]" = queue.Queue(maxsize=200)
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------------

    def trigger(
        self,
        billet_id: str,
        status: str,
        reasons: List[str],
        image_path: Optional[str] = None,
        *,
        kind: str = "billet",
        sound: bool = True,
        latency_ms: Optional[float] = None,
    ) -> Alert:
        """Record an alert, make it active, and dispatch it to notifiers.

        Per-billet alerts are never suppressed.  System-level alerts (camera
        lost, etc.) with identical content inside ``debounce_s`` are collapsed
        into the existing alert so a flapping source cannot flood operators.
        """
        key = (kind, status, tuple(reasons))
        now = time.monotonic()
        with self._lock:
            if kind != "billet":
                last = self._recent.get(key)
                if last is not None and now - last < self.debounce_s and self.history:
                    for existing in reversed(self.history):
                        if (existing.kind, existing.status, tuple(existing.reasons)) == key:
                            return existing
                self._recent[key] = now
            alert = Alert(
                timestamp=datetime.now(timezone.utc).isoformat(),
                billet_id=billet_id,
                status=status,
                reasons=list(reasons),
                image_path=image_path,
                kind=kind,
                sound=sound,
                alert_id=next(self._ids),
                latency_ms=latency_ms,
            )
            self.history.append(alert)
            if len(self.history) > _HISTORY_LIMIT:
                del self.history[: len(self.history) - _HISTORY_LIMIT]
            self.active_alert = alert
        self._dispatch(alert)
        return alert

    def clear_active(self) -> None:
        self.active_alert = None

    def clear_history(self) -> None:
        """Dismiss the active alert and drop the history (debounce state is kept)."""
        with self._lock:
            self.history.clear()
            self.active_alert = None

    def reset(self) -> None:
        """Forget all alerts (a new input source starts with a clean slate)."""
        with self._lock:
            self.history.clear()
            self._recent.clear()
            self.active_alert = None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, alert: Alert) -> None:
        if not self.notifiers:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run_notifiers, name="AlertNotifier", daemon=True
            )
            self._worker.start()
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            logger.warning("Alert notifier queue full — dropping notification for alert %d", alert.alert_id)

    def _run_notifiers(self) -> None:
        while True:
            alert = self._queue.get()
            if alert is None:
                return
            for notify in list(self.notifiers):
                try:
                    notify(alert)
                except Exception as exc:  # never let a notifier kill the thread
                    logger.error("Alert notifier %r failed: %s", notify, exc)

    def close(self, timeout: float = 2.0) -> None:
        """Stop the notifier thread after it drains queued alerts."""
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=timeout)
