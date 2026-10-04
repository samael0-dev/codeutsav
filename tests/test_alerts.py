"""Tests for alerts manager and notifications.

Verifies PRD §7.4 (FR-17) & Tier 2 #9:
- Alert triggering on out-of-tolerance and review items
- Alert history tracking and clearing active alerts
- Telegram dispatcher resilience (safe return when no env vars configured)
"""

import os
from unittest.mock import patch
import pytest
from billetvision.alerts.manager import AlertManager, Alert
from billetvision.alerts.telegram import send_telegram_alert


class TestAlertManager:
    def test_trigger_alert(self):
        manager = AlertManager()
        alert = manager.trigger(
            billet_id="H123459",
            status="FAIL",
            reasons=["width 131.8 mm out of range (130.0 ± 1.0 mm)"],
            image_path="data/outputs/snapshots/snap_4.jpg"
        )
        assert isinstance(alert, Alert)
        assert alert.billet_id == "H123459"
        assert alert.status == "FAIL"
        assert len(alert.reasons) == 1
        assert manager.active_alert == alert
        assert len(manager.history) == 1

    def test_clear_active_alert(self):
        manager = AlertManager()
        manager.trigger("H123459", "FAIL", ["width out of range"])
        assert manager.active_alert is not None
        manager.clear_active()
        assert manager.active_alert is None
        assert len(manager.history) == 1

    def test_clear_history(self):
        manager = AlertManager()
        manager.trigger("H123459", "FAIL", ["width out of range"])
        manager.clear_history()
        assert manager.active_alert is None
        assert manager.history == []


class TestTelegramAlerts:
    def test_telegram_disabled_without_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            success = send_telegram_alert("Out of tolerance test alert")
            assert success is False

    @patch("requests.post")
    def test_telegram_send_success_with_env_vars(self, mock_post):
        mock_post.return_value.status_code = 200
        env = {
            "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            "TELEGRAM_CHAT_ID": "-1001234567890"
        }
        with patch.dict(os.environ, env):
            success = send_telegram_alert("Inspection FAIL: Billet H123459")
            assert success is True
            assert mock_post.called
