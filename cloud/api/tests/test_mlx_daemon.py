"""Tests for the MlxReasoningProvider daemon architecture.

Validates:
- Lightweight health checks (no model loading, valid JSON)
- Daemon lifecycle (startup, shutdown, crash recovery)
- Health caching with 30s TTL
- Tiered health probes (Tier 1 subprocess, Tier 2 daemon ping)
- stdin/stdout JSON-line protocol
- _format_prompt with has_image kwarg
- Error handling and edge cases
"""
from __future__ import annotations

import json
import select as _select_mod
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from cloud.runtime.models import (
    Answer,
    Observation,
    ProviderConfig,
    ProviderHealth,
)
from cloud.runtime.providers import MlxReasoningProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mlx_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for the MLX provider with sensible defaults."""
    defaults = dict(
        name="mlx",
        enabled=True,
        model_path="/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit",
        model="gemma-4-e4b",
        timeout_s=60,
        metadata={"command": "python3 scripts/mlx_reasoner.py"},
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _fake_observation(**overrides) -> Observation:
    """Create a minimal Observation for prompt tests."""
    from datetime import datetime, timezone
    defaults = dict(
        id="obs-test-1",
        session_id="s",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        image_path="/tmp/fake.png",
        thumbnail_path="/tmp/fake_thumb.png",
        width=32,
        height=32,
        embedding=[0.0] * 128,
        provider="basic",
        confidence=0.5,
        summary="red mug on desk",
    )
    defaults.update(overrides)
    return Observation(**defaults)


@pytest.fixture
def provider():
    """Create a fresh MlxReasoningProvider and shut it down after the test."""
    p = MlxReasoningProvider()
    yield p
    p.shutdown()


@pytest.fixture
def config():
    return _mlx_config()


# ---------------------------------------------------------------------------
# 1. _format_prompt
# ---------------------------------------------------------------------------

class TestFormatPrompt:
    def test_prompt_without_image(self, provider):
        result = provider._format_prompt("What is this?", [])
        assert result == "What is this?"

    def test_prompt_with_image(self, provider):
        result = provider._format_prompt("What is this?", [], has_image=True)
        assert "live camera frame" in result
        assert "What is this?" in result

    def test_prompt_with_context(self, provider):
        obs = _fake_observation(summary="red mug")
        result = provider._format_prompt("Describe", [obs], has_image=False)
        assert "red mug" in result
        assert "secondary context" in result.lower()

    def test_default_has_image_is_false(self, provider):
        """The fixed _format_prompt should default has_image=False without TypeError."""
        result = provider._format_prompt("Hello", [])
        assert "Hello" in result
        # Should NOT contain camera frame text when has_image defaults to False
        assert "camera frame" not in result

    def test_prompt_with_image_and_context(self, provider):
        obs = _fake_observation(summary="blue sky")
        result = provider._format_prompt("Describe", [obs], has_image=True)
        assert "live camera frame" in result
        assert "blue sky" in result

    def test_context_limited_to_5(self, provider):
        observations = [_fake_observation(id=f"obs-{i}", summary=f"item {i}") for i in range(10)]
        result = provider._format_prompt("Test", observations)
        # Only first 5 should appear
        assert "item 4" in result
        assert "item 5" not in result


# ---------------------------------------------------------------------------
# 2. Health — disabled / missing config
# ---------------------------------------------------------------------------

class TestHealthBasicChecks:
    def test_disabled_provider(self, provider):
        config = _mlx_config(enabled=False)
        health = provider.health(config)
        assert health.enabled is False
        assert health.healthy is False

    def test_missing_command(self, provider):
        config = _mlx_config(metadata={})
        health = provider.health(config)
        assert health.healthy is False
        assert "command" in health.message.lower()

    def test_missing_model_path(self, provider):
        config = _mlx_config(model_path="", model="")
        health = provider.health(config)
        assert health.healthy is False
        assert "model_path" in health.message.lower()


# ---------------------------------------------------------------------------
# 3. Health caching
# ---------------------------------------------------------------------------

class TestHealthCaching:
    def test_health_cache_returns_cached_result(self, provider, config):
        """After caching a health result, subsequent calls should return the cached entry."""
        cached_health = ProviderHealth(
            name="mlx", role="reasoning", enabled=True, healthy=True,
            message="cached result",
        )
        provider._health_cache = (cached_health, time.time())
        result = provider.health(config)
        assert result.message == "cached result"

    def test_health_cache_expires_after_ttl(self, provider, config):
        """An expired cache entry should trigger a fresh probe."""
        stale_health = ProviderHealth(
            name="mlx", role="reasoning", enabled=True, healthy=True,
            message="stale",
        )
        provider._health_cache = (stale_health, time.time() - 60)  # 60s ago

        # The fresh probe will fail since no real daemon/subprocess is available,
        # but the point is it should NOT return "stale"
        with patch.object(provider, '_run_healthcheck_subprocess', side_effect=RuntimeError("no subprocess")):
            result = provider.health(config)
        assert result.message != "stale"
        assert result.healthy is False


# ---------------------------------------------------------------------------
# 4. Tiered health checks
# ---------------------------------------------------------------------------

class TestTieredHealthChecks:
    def test_tier2_daemon_ping_when_daemon_alive(self, provider, config):
        """If daemon is alive, health() should ping it for Tier 2 check."""
        with patch.object(provider, '_daemon_alive', return_value=True):
            with patch.object(provider, '_send_receive', return_value={
                "success": True, "message": "daemon alive (gemma-4-e4b)"
            }) as mock_send:
                result = provider.health(config)
        assert result.healthy is True
        assert "daemon alive" in result.message
        mock_send.assert_called_once()

    def test_tier1_fallback_when_daemon_not_running(self, provider, config):
        """Without a running daemon, health() falls back to lightweight subprocess probe."""
        provider._daemon = None
        with patch.object(provider, '_run_healthcheck_subprocess', return_value={
            "success": True, "message": "READY (/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit)"
        }) as mock_sub:
            result = provider.health(config)
        assert result.healthy is True
        assert "model loads on first use" in result.message
        mock_sub.assert_called_once()

    def test_tier2_failure_falls_back_to_tier1(self, provider, config):
        """If daemon ping fails, health() should fall through to Tier 1."""
        with patch.object(provider, '_daemon_alive', return_value=True):
            with patch.object(provider, '_send_receive', side_effect=RuntimeError("daemon broken")):
                with patch.object(provider, '_run_healthcheck_subprocess', return_value={
                    "success": True, "message": "READY (fallback)"
                }) as mock_sub:
                    result = provider.health(config)
        assert result.healthy is True
        mock_sub.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Daemon lifecycle
# ---------------------------------------------------------------------------

class TestDaemonLifecycle:
    def test_daemon_alive_when_none(self, provider):
        provider._daemon = None
        assert provider._daemon_alive() is False

    def test_daemon_alive_when_exited(self, provider):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited
        provider._daemon = mock_proc
        assert provider._daemon_alive() is False

    def test_daemon_alive_when_running(self, provider):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        provider._daemon = mock_proc
        assert provider._daemon_alive() is True

    def test_shutdown_terminates_daemon(self, provider):
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.poll.return_value = None
        provider._daemon = mock_proc

        provider.shutdown()

        mock_proc.stdin.close.assert_called_once()
        mock_proc.terminate.assert_called_once()
        assert provider._daemon is None

    def test_shutdown_safe_when_no_daemon(self, provider):
        """shutdown() should be safe to call when daemon is None."""
        provider._daemon = None
        provider.shutdown()  # should not raise
        assert provider._daemon is None

    def test_shutdown_kills_on_terminate_failure(self, provider):
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = OSError("terminate failed")
        mock_proc.wait.side_effect = OSError("wait failed")
        provider._daemon = mock_proc

        provider.shutdown()  # should not raise
        mock_proc.kill.assert_called_once()
        assert provider._daemon is None


# ---------------------------------------------------------------------------
# 6. _ensure_daemon
# ---------------------------------------------------------------------------

class TestEnsureDaemon:
    def test_reuses_existing_daemon(self, provider, config):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        provider._daemon = mock_proc

        result = provider._ensure_daemon(config)
        assert result is mock_proc

    def test_starts_new_daemon_when_none(self, provider, config):
        provider._daemon = None
        mock_popen = MagicMock()
        mock_popen.poll.return_value = None

        with patch('cloud.runtime.providers.subprocess.Popen', return_value=mock_popen) as mock_cls:
            with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
                result = provider._ensure_daemon(config)

        assert result is mock_popen
        mock_cls.assert_called_once()

    def test_restarts_daemon_after_crash(self, provider, config):
        crashed_proc = MagicMock()
        crashed_proc.poll.return_value = 1  # exited
        provider._daemon = crashed_proc

        new_proc = MagicMock()
        new_proc.poll.return_value = None

        with patch('cloud.runtime.providers.subprocess.Popen', return_value=new_proc):
            with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
                result = provider._ensure_daemon(config)

        assert result is new_proc
        crashed_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# 7. _send_receive
# ---------------------------------------------------------------------------

class TestSendReceive:
    def _mock_daemon(self):
        """Create a mock daemon process with stdin/stdout/stderr."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stdout.fileno.return_value = 3
        return mock_proc

    def test_successful_roundtrip(self, provider, config):
        mock_proc = self._mock_daemon()
        mock_proc.stdout.readline.return_value = '{"text":"hello","tokens_generated":4}\n'
        provider._daemon = mock_proc

        with patch.object(provider, '_ensure_daemon', return_value=mock_proc):
            with patch('select.select', return_value=([3], [], [])):
                result = provider._send_receive(config, {"prompt": "test"}, timeout_s=5.0)

        assert result["text"] == "hello"
        assert result["tokens_generated"] == 4

    def test_broken_pipe_resets_daemon(self, provider, config):
        mock_proc = self._mock_daemon()
        mock_proc.stdin.write.side_effect = BrokenPipeError("broken")
        provider._daemon = mock_proc

        with patch.object(provider, '_ensure_daemon', return_value=mock_proc):
            with pytest.raises(RuntimeError, match="daemon stdin broken"):
                provider._send_receive(config, {"prompt": "test"})
        assert provider._daemon is None

    def test_timeout_raises(self, provider, config):
        mock_proc = self._mock_daemon()
        provider._daemon = mock_proc

        with patch.object(provider, '_ensure_daemon', return_value=mock_proc):
            with patch('select.select', return_value=([], [], [])):
                with pytest.raises(RuntimeError, match="timed out"):
                    provider._send_receive(config, {"prompt": "test"}, timeout_s=1.0)

    def test_daemon_exit_during_read(self, provider, config):
        mock_proc = self._mock_daemon()
        # After readline returns empty, _daemon_alive() calls poll().
        # Returning 1 means daemon exited → triggers "exited unexpectedly".
        mock_proc.poll.return_value = 1
        mock_proc.stdout.readline.return_value = ""
        provider._daemon = mock_proc

        with patch.object(provider, '_ensure_daemon', return_value=mock_proc):
            with patch('select.select', return_value=([3], [], [])):
                with pytest.raises(RuntimeError, match="exited unexpectedly"):
                    provider._send_receive(config, {"prompt": "test"})

    def test_invalid_json_response(self, provider, config):
        mock_proc = self._mock_daemon()
        mock_proc.stdout.readline.return_value = "not json\n"
        provider._daemon = mock_proc

        with patch.object(provider, '_ensure_daemon', return_value=mock_proc):
            with patch('select.select', return_value=([3], [], [])):
                with pytest.raises(RuntimeError, match="invalid JSON"):
                    provider._send_receive(config, {"prompt": "test"})


# ---------------------------------------------------------------------------
# 8. reason() via daemon
# ---------------------------------------------------------------------------

class TestReason:
    def test_reason_with_image_bytes(self, provider, config):
        with patch.object(provider, '_send_receive', return_value={
            "text": "a red coffee mug", "tokens_generated": 8,
        }) as mock_sr:
            answer = provider.reason(
                config=config, prompt="What is this?",
                image_bytes=b"\xff\xd8\xff\xe0",  # fake JPEG header
                context=[], image_path=None,
            )
        assert answer.text == "a red coffee mug"
        assert answer.provider == "mlx"
        call_payload = mock_sr.call_args[0][1]
        assert call_payload["image_base64"] is not None
        assert call_payload["prompt"]

    def test_reason_with_image_path(self, provider, config, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n fake png")
        with patch.object(provider, '_send_receive', return_value={
            "text": "a blue sky", "tokens_generated": 6,
        }):
            answer = provider.reason(
                config=config, prompt="Describe",
                image_bytes=None, context=[],
                image_path=str(img),
            )
        assert answer.text == "a blue sky"

    def test_reason_error_in_result(self, provider, config):
        with patch.object(provider, '_send_receive', return_value={
            "text": "", "error": "out of memory",
        }):
            with pytest.raises(RuntimeError, match="out of memory"):
                provider.reason(
                    config=config, prompt="test",
                    image_bytes=None, context=[], image_path=None,
                )

    def test_reason_empty_text(self, provider, config):
        with patch.object(provider, '_send_receive', return_value={
            "text": "", "tokens_generated": 0,
        }):
            with pytest.raises(RuntimeError, match="empty output"):
                provider.reason(
                    config=config, prompt="test",
                    image_bytes=None, context=[], image_path=None,
                )

    def test_reason_passes_max_tokens(self, provider):
        config = _mlx_config(metadata={"command": "python3 scripts/mlx_reasoner.py", "max_tokens": 256})
        with patch.object(provider, '_send_receive', return_value={
            "text": "result", "tokens_generated": 10,
        }) as mock_sr:
            provider.reason(config=config, prompt="test", image_bytes=None, context=[], image_path=None)
        call_payload = mock_sr.call_args[0][1]
        assert call_payload["max_tokens"] == 256


# ---------------------------------------------------------------------------
# 9. _run_healthcheck_subprocess
# ---------------------------------------------------------------------------

class TestRunHealthcheckSubprocess:
    def test_successful_healthcheck(self, provider, config):
        result_json = json.dumps({"success": True, "status": "READY", "message": "READY", "model_path": "/test"})
        with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
            with patch('cloud.runtime.providers.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout=result_json + "\n", stderr=""
                )
                result = provider._run_healthcheck_subprocess(config)
        assert result["success"] is True
        assert result["status"] == "READY"

    def test_parses_last_json_line(self, provider, config):
        """When mlx prints warnings above the JSON, parser should find the last valid JSON line."""
        stdout = "WARNING: something\n" + json.dumps({"success": True, "status": "READY", "message": "ok", "model_path": ""})
        with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
            with patch('cloud.runtime.providers.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
                result = provider._run_healthcheck_subprocess(config)
        assert result["success"] is True

    def test_failure_with_json_payload(self, provider, config):
        """Non-zero exit but valid JSON should return the parsed result."""
        result_json = json.dumps({"success": False, "message": "mlx_vlm not installed"})
        with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
            with patch('cloud.runtime.providers.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout=result_json, stderr="")
                result = provider._run_healthcheck_subprocess(config)
        assert result["success"] is False

    def test_failure_no_json_raises(self, provider, config):
        """Non-zero exit with no valid JSON should raise RuntimeError."""
        with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
            with patch('cloud.runtime.providers.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="segfault")
                with pytest.raises(RuntimeError):
                    provider._run_healthcheck_subprocess(config)

    def test_no_json_in_stdout_raises(self, provider, config):
        """Zero exit but no valid JSON should raise RuntimeError."""
        with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
            with patch('cloud.runtime.providers.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="plain text only\n", stderr="")
                with pytest.raises(RuntimeError, match="malformed json"):
                    provider._run_healthcheck_subprocess(config)

    def test_healthcheck_appends_flag(self, provider, config):
        """_run_healthcheck_subprocess should invoke the script with --healthcheck."""
        with patch.object(provider, '_resolve_wrapper_prefix', return_value=['python3', 'scripts/mlx_reasoner.py']):
            with patch('cloud.runtime.providers.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps({"success": True, "message": "ok"}) + "\n",
                    stderr="",
                )
                provider._run_healthcheck_subprocess(config)
        call_argv = mock_run.call_args[0][0]
        assert "--healthcheck" in call_argv


# ---------------------------------------------------------------------------
# 10. _resolve_wrapper_prefix
# ---------------------------------------------------------------------------

class TestResolveWrapperPrefix:
    def test_valid_command(self, provider):
        config = _mlx_config(metadata={"command": "python3 scripts/mlx_reasoner.py"})
        with patch('cloud.runtime.providers.shutil.which', return_value='/usr/bin/python3'):
            result = provider._resolve_wrapper_prefix(config)
        assert result == ['/usr/bin/python3', 'scripts/mlx_reasoner.py']

    def test_missing_command(self, provider):
        config = _mlx_config(metadata={})
        with pytest.raises(RuntimeError, match="command missing"):
            provider._resolve_wrapper_prefix(config)

    def test_command_without_mlx_reasoner(self, provider):
        config = _mlx_config(metadata={"command": "python3 some_other_script.py"})
        with pytest.raises(RuntimeError, match="mlx_reasoner.py"):
            provider._resolve_wrapper_prefix(config)

    def test_executable_not_found(self, provider):
        config = _mlx_config(metadata={"command": "nonexistent_python scripts/mlx_reasoner.py"})
        with patch('cloud.runtime.providers.Path.exists', return_value=False):
            with patch('cloud.runtime.providers.shutil.which', return_value=None):
                with pytest.raises(RuntimeError, match="not found"):
                    provider._resolve_wrapper_prefix(config)


# ---------------------------------------------------------------------------
# 11. Integration: health() caching prevents overlapping probes
# ---------------------------------------------------------------------------

class TestHealthCachePreventsOverlap:
    def test_rapid_health_calls_use_cache(self, provider, config):
        """Simulates the UI polling health rapidly; only the first call should probe."""
        call_count = 0

        def counting_healthcheck(cfg):
            nonlocal call_count
            call_count += 1
            return {"success": True, "message": "READY", "model_path": "/test"}

        with patch.object(provider, '_run_healthcheck_subprocess', side_effect=counting_healthcheck):
            for _ in range(10):
                provider.health(config)

        # Only the first call should have triggered the subprocess probe
        assert call_count == 1, f"Expected 1 probe but got {call_count}"

    def test_health_cache_respects_ttl(self, provider, config):
        """After TTL expires, health should re-probe."""
        call_count = 0

        def counting_healthcheck(cfg):
            nonlocal call_count
            call_count += 1
            return {"success": True, "message": "READY", "model_path": "/test"}

        with patch.object(provider, '_run_healthcheck_subprocess', side_effect=counting_healthcheck):
            # First call
            provider.health(config)
            assert call_count == 1

            # Expire the cache
            provider._health_cache = (provider._health_cache[0], time.time() - 60)

            # Second call should re-probe
            provider.health(config)
            assert call_count == 2
