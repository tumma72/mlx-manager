"""Tests for LogFire configuration module."""

import os
from unittest.mock import MagicMock, patch

import mlx_manager.mlx_server.observability.logfire_config as logfire_mod


class TestConfigureLogfire:
    """Tests for configure_logfire function."""

    def setup_method(self):
        """Reset _configured flag before each test."""
        logfire_mod._configured = False

    def teardown_method(self):
        """Reset _configured flag after each test."""
        logfire_mod._configured = False

    def test_configure_logfire_calls_logfire(self):
        """configure_logfire calls logfire.configure with correct args."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLX_MANAGER_DISABLE_TELEMETRY", None)
            with patch.object(logfire_mod, "logfire") as mock_logfire:
                logfire_mod.configure_logfire(
                    service_name="test-service",
                    service_version="1.0.0",
                )

                mock_logfire.configure.assert_called_once_with(
                    service_name="test-service",
                    service_version="1.0.0",
                    send_to_logfire="if-token-present",
                )

    def test_configure_logfire_sets_configured_flag(self):
        """configure_logfire sets _configured to True."""
        with patch.object(logfire_mod, "logfire"):
            assert logfire_mod._configured is False
            logfire_mod.configure_logfire()
            assert logfire_mod._configured is True

    def test_configure_logfire_skips_if_already_configured(self):
        """configure_logfire is idempotent - skips if already configured."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            logfire_mod.configure_logfire()
            logfire_mod.configure_logfire()  # Second call

            # Should only be called once
            assert mock_logfire.configure.call_count == 1

    def test_configure_logfire_default_service_name(self):
        """configure_logfire uses 'mlx-server' as default service name."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLX_MANAGER_DISABLE_TELEMETRY", None)
            with patch.object(logfire_mod, "logfire") as mock_logfire:
                logfire_mod.configure_logfire()

                mock_logfire.configure.assert_called_once_with(
                    service_name="mlx-server",
                    service_version=None,
                    send_to_logfire="if-token-present",
                )

    def test_configure_logfire_disables_telemetry_when_env_set(self):
        """configure_logfire disables sending when DISABLE_TELEMETRY is set."""
        with patch.dict(os.environ, {"MLX_MANAGER_DISABLE_TELEMETRY": "true"}):
            with patch.object(logfire_mod, "logfire") as mock_logfire:
                logfire_mod.configure_logfire()

                mock_logfire.configure.assert_called_once_with(
                    service_name="mlx-server",
                    service_version=None,
                    send_to_logfire=False,
                )


class TestInstrumentFastapi:
    """Tests for instrument_fastapi function."""

    def test_instruments_app(self):
        """instrument_fastapi calls logfire.instrument_fastapi."""
        mock_app = MagicMock()
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            logfire_mod.instrument_fastapi(mock_app)

            mock_logfire.instrument_fastapi.assert_called_once_with(mock_app)


class TestInstrumentHttpx:
    """Tests for instrument_httpx function."""

    def test_instruments_httpx(self):
        """instrument_httpx calls logfire.instrument_httpx."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            logfire_mod.instrument_httpx()

            mock_logfire.instrument_httpx.assert_called_once()


class TestInstrumentSqlalchemy:
    """Tests for instrument_sqlalchemy function."""

    def test_instruments_engine(self):
        """instrument_sqlalchemy calls logfire.instrument_sqlalchemy."""
        mock_engine = MagicMock()
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            logfire_mod.instrument_sqlalchemy(mock_engine)

            mock_logfire.instrument_sqlalchemy.assert_called_once_with(
                engine=mock_engine,
            )


class TestInstrumentLlmClients:
    """Tests for instrument_llm_clients function."""

    def test_instruments_both_clients(self):
        """instrument_llm_clients instruments OpenAI and Anthropic."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            mock_logfire.instrument_openai = MagicMock()
            mock_logfire.instrument_anthropic = MagicMock()

            logfire_mod.instrument_llm_clients()

            mock_logfire.instrument_openai.assert_called_once()
            mock_logfire.instrument_anthropic.assert_called_once()

    def test_handles_openai_unavailable(self):
        """instrument_llm_clients handles missing OpenAI gracefully."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            mock_logfire.instrument_openai.side_effect = Exception("No openai")
            mock_logfire.instrument_anthropic = MagicMock()

            # Should not raise
            logfire_mod.instrument_llm_clients()

            mock_logfire.instrument_anthropic.assert_called_once()

    def test_handles_anthropic_unavailable(self):
        """instrument_llm_clients handles missing Anthropic gracefully."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            mock_logfire.instrument_openai = MagicMock()
            mock_logfire.instrument_anthropic.side_effect = Exception("No anthropic")

            # Should not raise
            logfire_mod.instrument_llm_clients()

            mock_logfire.instrument_openai.assert_called_once()

    def test_handles_both_unavailable(self):
        """instrument_llm_clients handles both clients missing."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            mock_logfire.instrument_openai.side_effect = Exception("No openai")
            mock_logfire.instrument_anthropic.side_effect = Exception("No anthropic")

            # Should not raise
            logfire_mod.instrument_llm_clients()

    def test_logs_when_clients_instrumented(self):
        """instrument_llm_clients logs which clients were instrumented."""
        with patch.object(logfire_mod, "logfire") as mock_logfire:
            mock_logfire.instrument_openai = MagicMock()
            mock_logfire.instrument_anthropic = MagicMock()

            logfire_mod.instrument_llm_clients()

            # Both should have been instrumented successfully
            mock_logfire.instrument_openai.assert_called_once()
            mock_logfire.instrument_anthropic.assert_called_once()
