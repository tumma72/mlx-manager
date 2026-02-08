"""Tests for logging_config module."""

import logging
from types import FrameType
from unittest.mock import MagicMock, patch

from loguru import logger


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that setup_logging creates log directory."""
        from mlx_manager import logging_config

        # Reset the configured flag
        logging_config._logging_configured = False

        with patch.dict("os.environ", {"MLX_MANAGER_LOG_DIR": str(tmp_path / "logs")}):
            # Reload to pick up new env var
            import importlib

            importlib.reload(logging_config)

            logging_config.setup_logging()

            assert (tmp_path / "logs").exists()

        # Reset for other tests
        logging_config._logging_configured = False

    def test_setup_logging_idempotent(self):
        """Test that setup_logging can be called multiple times safely."""
        from mlx_manager import logging_config

        # Reset the configured flag
        logging_config._logging_configured = False

        # First call
        logging_config.setup_logging()
        assert logging_config._logging_configured is True

        # Second call should be a no-op
        logging_config.setup_logging()
        assert logging_config._logging_configured is True


class TestInterceptHandler:
    """Tests for InterceptHandler class."""

    def test_emit_with_valid_level(self):
        """Test emit with a valid log level."""
        from mlx_manager.logging_config import InterceptHandler

        handler = InterceptHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Should not raise
        handler.emit(record)

    def test_emit_with_invalid_level_uses_levelno(self):
        """Test emit with invalid level name falls back to levelno (lines 97-98)."""
        from mlx_manager.logging_config import InterceptHandler

        handler = InterceptHandler()

        # Create a record with a custom/invalid level name
        record = logging.LogRecord(
            name="test",
            level=99,  # Custom level number
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.levelname = "CUSTOM_LEVEL_NOT_IN_LOGURU"

        # Mock logger.level to raise ValueError for unknown level
        with patch.object(logger, "level", side_effect=ValueError("Unknown level")):
            with patch.object(logger, "opt") as mock_opt:
                mock_log = MagicMock()
                mock_opt.return_value = mock_log

                # Should handle ValueError and use levelno instead
                handler.emit(record)

                # Verify logger.opt was called
                mock_opt.assert_called_once()
                # Verify log was called with str(levelno)
                mock_log.log.assert_called_once()
                call_args = mock_log.log.call_args
                assert call_args[0][0] == "99"  # levelno as string

    def test_emit_traverses_frames_to_find_caller(self):
        """Test emit traverses frames to find caller (lines 104-105)."""
        from mlx_manager.logging_config import InterceptHandler

        handler = InterceptHandler()

        # Create a record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Mock logging.currentframe to return a chain of frames
        # Create mock frames that simulate the logging module's call stack
        mock_frame_3 = MagicMock(spec=FrameType)
        mock_frame_3.f_code.co_filename = "user_code.py"
        mock_frame_3.f_back = None

        mock_frame_2 = MagicMock(spec=FrameType)
        mock_frame_2.f_code.co_filename = logging.__file__
        mock_frame_2.f_back = mock_frame_3

        mock_frame_1 = MagicMock(spec=FrameType)
        mock_frame_1.f_code.co_filename = logging.__file__
        mock_frame_1.f_back = mock_frame_2

        with patch("logging.currentframe", return_value=mock_frame_1):
            with patch.object(logger, "opt") as mock_opt:
                mock_log = MagicMock()
                mock_opt.return_value = mock_log

                handler.emit(record)

                # Verify logger.opt was called with depth > 2 (traversed the frames)
                mock_opt.assert_called_once()
                call_kwargs = mock_opt.call_args[1]
                # Initial depth is 2, plus 2 traversals = 4
                assert call_kwargs["depth"] == 4

    def test_emit_handles_none_frame(self):
        """Test emit handles case when frame is None."""
        from mlx_manager.logging_config import InterceptHandler

        handler = InterceptHandler()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Mock currentframe to return None
        with patch("logging.currentframe", return_value=None):
            with patch.object(logger, "opt") as mock_opt:
                mock_log = MagicMock()
                mock_opt.return_value = mock_log

                # Should handle None frame gracefully
                handler.emit(record)

                # Verify logger.opt was called with initial depth of 2
                mock_opt.assert_called_once()
                call_kwargs = mock_opt.call_args[1]
                assert call_kwargs["depth"] == 2


class TestInterceptStandardLogging:
    """Tests for intercept_standard_logging function."""

    def test_intercept_standard_logging_sets_handler(self):
        """Test that intercept_standard_logging sets InterceptHandler."""
        from mlx_manager.logging_config import InterceptHandler, intercept_standard_logging

        intercept_standard_logging()

        # Check that root logger has InterceptHandler
        root_logger = logging.getLogger()
        has_intercept = any(isinstance(h, InterceptHandler) for h in root_logger.handlers)
        assert has_intercept

    def test_intercept_standard_logging_sets_noisy_loggers_to_warning(self):
        """Test that noisy loggers are set to WARNING level."""
        from mlx_manager.logging_config import intercept_standard_logging

        intercept_standard_logging()

        # Check that noisy loggers have WARNING level
        noisy_loggers = ["httpx", "aiosqlite", "sqlalchemy", "asyncio"]
        for name in noisy_loggers:
            logger_obj = logging.getLogger(name)
            assert logger_obj.level == logging.WARNING
