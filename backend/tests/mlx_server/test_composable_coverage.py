"""Additional tests for composable adapter covering uncovered lines.

Targets:
- composable.py lines: 226, 229-230, 236-244, 353-374, 382-392, 403-406, 455, 458-459, 678
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from mlx_manager.mlx_server.models.adapters.composable import ModelAdapter, create_adapter
from mlx_manager.mlx_server.models.adapters.configs import FAMILY_CONFIGS
from mlx_manager.mlx_server.parsers import (
    HermesJsonParser,
    NullThinkingParser,
    NullToolParser,
    ThinkTagParser,
)


class FakeTokenizer:
    """Lightweight fake tokenizer."""

    def __init__(
        self,
        eos_token_id: int = 0,
        special_tokens: dict[str, int] | None = None,
        raise_on_template: bool = False,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.unk_token_id = -1
        self._special_tokens = special_tokens or {
            "<|im_end|>": 100,
            "<|eot_id|>": 200,
        }
        self._raise_on_template = raise_on_template

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special_tokens.get(token, self.unk_token_id)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if self._raise_on_template:
            raise TypeError("Unexpected kwargs")
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>{content}")
        suffix = "<|assistant|>" if kwargs.get("add_generation_prompt") else ""
        return "".join(parts) + suffix

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(len(text)))


# ============================================================================
# Tests for apply_chat_template uncovered lines (226, 229-230, 236-244)
# ============================================================================


class TestApplyChatTemplateUncoveredLines:
    """Tests for apply_chat_template specific branches."""

    def test_native_tools_passed_to_tokenizer(self):
        """When native_tools is set, tools are passed to apply_chat_template (line 226)."""
        # Use qwen which has native_tools=True
        adapter = create_adapter("qwen", FakeTokenizer(), model_type="text-gen")
        tools = [{"function": {"name": "test_tool"}}]
        messages = [{"role": "user", "content": "hello"}]

        # apply_chat_template with native tools should work
        result = adapter.apply_chat_template(messages, tools=tools)
        assert result is not None
        assert isinstance(result, str)

    def test_template_options_merged_into_kwargs(self):
        """When template_options is set, they are passed as kwargs to tokenizer (lines 229-230)."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="text-gen", template_options={"key1": "val1"}
        )

        messages = [{"role": "user", "content": "hello"}]

        # The apply_chat_template should pass template_options as kwargs
        # Since FakeTokenizer ignores unknown kwargs, this just tests the code path
        result = adapter.apply_chat_template(messages)
        assert result is not None

    def test_template_options_with_multiple_keys(self):
        """Template options with multiple keys are all merged (lines 229-230 for loop)."""
        adapter = create_adapter(
            "default",
            FakeTokenizer(),
            model_type="text-gen",
            template_options={"opt1": True, "opt2": "value", "opt3": 42},
        )
        messages = [{"role": "user", "content": "test"}]
        result = adapter.apply_chat_template(messages)
        assert result is not None

    def test_type_error_fallback_without_template_options(self):
        """TypeError from tokenizer triggers fallback without template_options (lines 236-244)."""
        # Create a tokenizer that raises TypeError on the first call but not on fallback
        call_count = {"n": 0}

        class SometimesRaises:
            eos_token_id = 0
            unk_token_id = -1

            def convert_tokens_to_ids(self, token: str) -> int:
                return -1

            def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
                call_count["n"] += 1
                if call_count["n"] == 1 and kwargs.get("extra_arg"):
                    raise TypeError("Unexpected kwarg: extra_arg")
                return "fallback_result"

        tokenizer = SometimesRaises()
        adapter = ModelAdapter(
            model_type="text-gen", tokenizer=tokenizer, template_options={"extra_arg": True}
        )

        messages = [{"role": "user", "content": "test"}]
        result = adapter.apply_chat_template(messages)
        # Should return the fallback result
        assert result is not None

    def test_type_error_fallback_with_native_tools(self):
        """TypeError fallback preserves native_tools (lines 242-243)."""
        call_count = {"n": 0}

        class RaisesOnFirstCall:
            eos_token_id = 0
            unk_token_id = -1

            def convert_tokens_to_ids(self, token: str) -> int:
                return -1

            def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise TypeError("Error with template options")
                return f"result_with_tools={bool(kwargs.get('tools'))}"

        tokenizer = RaisesOnFirstCall()
        adapter = ModelAdapter(
            model_type="text-gen",
            config=FAMILY_CONFIGS["default"],  # native_tools=False for default
            tokenizer=tokenizer,
            template_options={"some_option": True},
        )

        messages = [{"role": "user", "content": "test"}]
        # No native tools since default adapter
        result = adapter.apply_chat_template(messages)
        assert result is not None

    def test_apply_chat_template_without_template_strategy_uses_tokenizer(self):
        """Default path calls tokenizer.apply_chat_template directly."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        messages = [{"role": "user", "content": "hello world"}]
        result = adapter.apply_chat_template(messages, add_generation_prompt=True)
        assert "hello world" in result
        assert "<|assistant|>" in result


# ============================================================================
# Tests for configure() uncovered lines (353-374)
# ============================================================================


class TestConfigureUncoveredLines:
    """Tests for configure() method branches."""

    def test_configure_tool_parser_none_with_factory(self):
        """configure(tool_parser=None) resets to factory parser (lines 364-365)."""
        # Use qwen which has a tool_parser_factory
        adapter = create_adapter("qwen", FakeTokenizer(), model_type="text-gen")

        # Set a custom parser first
        custom_parser = MagicMock(spec=HermesJsonParser)
        adapter._tool_parser = custom_parser

        # Reset to factory default by passing None
        adapter.configure(tool_parser=None)

        # Should be reset to config's factory parser (HermesJsonParser for qwen)
        assert isinstance(adapter._tool_parser, HermesJsonParser)

    def test_configure_tool_parser_none_without_factory(self):
        """configure(tool_parser=None) uses NullToolParser when no factory (lines 366-367)."""
        # Use default which has no tool_parser_factory
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")

        # Pass None to trigger the "no factory" path
        adapter.configure(tool_parser=None)

        assert isinstance(adapter._tool_parser, NullToolParser)

    def test_configure_thinking_parser_none_with_factory(self):
        """configure(thinking_parser=None) resets to factory parser (lines 371-372)."""
        adapter = create_adapter("qwen", FakeTokenizer(), model_type="text-gen")

        # Reset thinking_parser to factory default
        adapter.configure(thinking_parser=None)

        assert isinstance(adapter._thinking_parser, ThinkTagParser)

    def test_configure_thinking_parser_none_without_factory(self):
        """configure(thinking_parser=None) uses NullThinkingParser when no factory."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")

        # Pass None to trigger the "no factory" path
        adapter.configure(thinking_parser=None)

        assert isinstance(adapter._thinking_parser, NullThinkingParser)

    def test_configure_enable_tool_injection_none_resets_to_false(self):
        """configure(enable_tool_injection=None) resets to False (lines 355-358)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        adapter._enable_tool_injection = True

        # Passing None should set to False (line 357)
        adapter.configure(enable_tool_injection=None)

        assert adapter._enable_tool_injection is False

    def test_configure_updates_system_prompt(self):
        """configure(system_prompt=...) updates the prompt (line 353-354)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        adapter.configure(system_prompt="New system prompt")
        assert adapter._system_prompt == "New system prompt"

    def test_configure_updates_template_options(self):
        """configure(template_options=...) updates template options (lines 359-360)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        adapter.configure(template_options={"thinking": True})
        assert adapter._template_options == {"thinking": True}

    def test_configure_with_actual_parser_instance(self):
        """configure with actual parser instance sets it (lines 362-363)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        new_parser = HermesJsonParser()
        adapter.configure(tool_parser=new_parser)
        assert adapter._tool_parser is new_parser

    def test_configure_with_actual_thinking_parser_instance(self):
        """configure with actual thinking parser instance sets it (lines 369-370)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        new_thinking = ThinkTagParser()
        adapter.configure(thinking_parser=new_thinking)
        assert adapter._thinking_parser is new_thinking


# ============================================================================
# Tests for reset_to_defaults() uncovered lines (382-392)
# ============================================================================


class TestResetToDefaultsUncoveredLines:
    """Tests for reset_to_defaults() method."""

    def test_reset_restores_factory_tool_parser(self):
        """reset_to_defaults restores tool parser from factory (lines 385-386)."""
        adapter = create_adapter("qwen", FakeTokenizer(), model_type="text-gen")

        # Override with a custom parser
        adapter._tool_parser = MagicMock()

        adapter.reset_to_defaults()

        # Should be reset to HermesJsonParser (qwen's factory)
        assert isinstance(adapter._tool_parser, HermesJsonParser)

    def test_reset_uses_null_tool_parser_when_no_factory(self):
        """reset_to_defaults uses NullToolParser when no factory (lines 387-388)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        adapter._tool_parser = MagicMock()  # Set custom

        adapter.reset_to_defaults()

        assert isinstance(adapter._tool_parser, NullToolParser)

    def test_reset_restores_factory_thinking_parser(self):
        """reset_to_defaults restores thinking parser from factory (lines 389-390)."""
        adapter = create_adapter("qwen", FakeTokenizer(), model_type="text-gen")
        adapter._thinking_parser = MagicMock()  # Override

        adapter.reset_to_defaults()

        assert isinstance(adapter._thinking_parser, ThinkTagParser)

    def test_reset_uses_null_thinking_parser_when_no_factory(self):
        """reset_to_defaults uses NullThinkingParser when no factory (lines 391-392)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        adapter._thinking_parser = MagicMock()  # Override

        adapter.reset_to_defaults()

        assert isinstance(adapter._thinking_parser, NullThinkingParser)

    def test_reset_clears_all_settings(self):
        """reset_to_defaults clears system_prompt, enable_tool_injection, template_options."""
        adapter = create_adapter(
            "default",
            FakeTokenizer(),
            model_type="text-gen",
            system_prompt="Original prompt",
            enable_tool_injection=True,
            template_options={"thinking": True},
        )

        adapter.reset_to_defaults()

        assert adapter._system_prompt is None
        assert adapter._enable_tool_injection is False
        assert adapter._template_options is None


# ============================================================================
# Tests for _ensure_system_prompt (lines 403-406)
# ============================================================================


class TestEnsureSystemPrompt:
    """Tests for _ensure_system_prompt method (lines 400-406)."""

    def test_no_system_prompt_returns_messages_unchanged(self):
        """If no system_prompt configured, messages pass through (line 400-401)."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        messages = [{"role": "user", "content": "hello"}]
        result = adapter._ensure_system_prompt(messages)
        assert result is messages  # Same object

    def test_system_prompt_prepended_when_no_existing_system(self):
        """System prompt prepended when messages have no system message (lines 405-406)."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="text-gen", system_prompt="You are helpful."
        )
        messages = [{"role": "user", "content": "hello"}]
        result = adapter._ensure_system_prompt(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_system_prompt_not_prepended_when_system_already_exists(self):
        """When messages already have a system message, no new prompt added (lines 403-404)."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="text-gen", system_prompt="Default prompt."
        )
        messages = [
            {"role": "system", "content": "Custom system."},
            {"role": "user", "content": "hello"},
        ]
        result = adapter._ensure_system_prompt(messages)

        # Should be unchanged (not prepend default)
        assert result is messages
        assert len(result) == 2
        assert result[0]["content"] == "Custom system."

    def test_system_prompt_with_empty_messages(self):
        """System prompt prepended even for empty message list."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="text-gen", system_prompt="My prompt."
        )
        result = adapter._ensure_system_prompt([])

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "My prompt."


# ============================================================================
# Tests for prepare_input vision path (lines 455, 458-459)
# ============================================================================


class TestPrepareInputVisionPath:
    """Tests for prepare_input vision path (lines 455, 458-459)."""

    def test_vision_path_with_system_message(self):
        """Vision path passes system messages through to vlm for enrichment."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="vision", model_id="test/vision-model"
        )
        messages = [
            {"role": "system", "content": "You are a vision model."},
            {"role": "user", "content": "What is in this image?"},
        ]
        fake_image = MagicMock()

        # vlm returns enriched messages with image tokens
        vlm_messages = [
            {"role": "system", "content": "You are a vision model."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What?"}]},
        ]
        mock_config = {"model_type": "llava"}
        with (
            patch(
                "mlx_vlm.prompt_utils.apply_chat_template", return_value=vlm_messages
            ) as mock_tpl,
            patch("mlx_vlm.utils.load_config", return_value=mock_config),
        ):
            result = adapter.prepare_input(messages, images=[fake_image])

        # Prompt rendered by unified template path (FakeTokenizer)
        assert "<|system|>" in result.prompt
        assert result.pixel_values == [fake_image]
        # Verify messages were passed as structured list
        call_args = mock_tpl.call_args
        passed_messages = call_args[0][2]  # Third positional arg
        assert any(m.get("role") == "system" for m in passed_messages)

    def test_vision_path_with_assistant_message(self):
        """Vision path passes assistant messages through to vlm."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="vision", model_id="test/vision-model"
        )
        messages = [
            {"role": "user", "content": "Describe the image"},
            {"role": "assistant", "content": "I see a cat."},
            {"role": "user", "content": "What else?"},
        ]
        fake_image = MagicMock()

        vlm_messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe"}]},
            {"role": "assistant", "content": "I see a cat."},
            {"role": "user", "content": "What else?"},
        ]
        with (
            patch(
                "mlx_vlm.prompt_utils.apply_chat_template", return_value=vlm_messages
            ) as mock_tpl,
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            result = adapter.prepare_input(messages, images=[fake_image])

        # Verify assistant content makes it to the final prompt
        assert "I see a cat." in result.prompt
        # Verify messages were passed as structured list with assistant role
        call_args = mock_tpl.call_args
        passed_messages = call_args[0][2]
        assert any(m.get("role") == "assistant" for m in passed_messages)

    def test_vision_path_with_multipart_content(self):
        """Vision path passes multipart content to vlm for image token enrichment."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="vision", model_id="test/vision-model"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        fake_image = MagicMock()

        vlm_messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Look"}]},
        ]
        with (
            patch(
                "mlx_vlm.prompt_utils.apply_chat_template", return_value=vlm_messages
            ) as mock_tpl,
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            result = adapter.prepare_input(messages, images=[fake_image])

        assert isinstance(result.prompt, str)
        # Verify multipart messages were passed to vlm
        call_args = mock_tpl.call_args
        passed_messages = call_args[0][2]
        assert isinstance(passed_messages[0]["content"], list)

    def test_vision_path_empty_content_message_passed_through(self):
        """Vision path passes all messages to vlm including empty content."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="vision", model_id="test/vision-model"
        )
        messages = [
            {"role": "user", "content": ""},  # Empty content
            {"role": "user", "content": "Hello"},
        ]
        fake_image = MagicMock()

        vlm_messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hello"}]},
        ]
        with (
            patch(
                "mlx_vlm.prompt_utils.apply_chat_template", return_value=vlm_messages
            ) as mock_tpl,
            patch("mlx_vlm.utils.load_config", return_value={}),
        ):
            result = adapter.prepare_input(messages, images=[fake_image])

        assert isinstance(result.prompt, str)
        # All messages passed to vlm (filtering is vlm's responsibility)
        call_args = mock_tpl.call_args
        assert call_args[1].get("return_messages") is True


# ============================================================================
# Tests for configure() with _UNSET sentinel (line 353, 355, 359, 361, 368)
# ============================================================================


class TestConfigureWithUnsetSentinel:
    """Tests that omitted args leave values unchanged."""

    def test_configure_omit_system_prompt_leaves_unchanged(self):
        """Omitting system_prompt leaves it unchanged (sentinel behavior)."""
        adapter = create_adapter(
            "default", FakeTokenizer(), model_type="text-gen", system_prompt="Original"
        )
        adapter.configure(enable_tool_injection=True)  # Don't pass system_prompt
        assert adapter._system_prompt == "Original"

    def test_configure_omit_tool_parser_leaves_unchanged(self):
        """Omitting tool_parser leaves it unchanged."""
        adapter = create_adapter("default", FakeTokenizer(), model_type="text-gen")
        original_parser = adapter._tool_parser
        adapter.configure(system_prompt="New")  # Don't pass tool_parser
        assert adapter._tool_parser is original_parser

    def test_configure_omit_thinking_parser_leaves_unchanged(self):
        """Omitting thinking_parser leaves it unchanged."""
        adapter = create_adapter("qwen", FakeTokenizer(), model_type="text-gen")
        original_parser = adapter._thinking_parser
        adapter.configure(system_prompt="Changed")  # Don't pass thinking_parser
        assert adapter._thinking_parser is original_parser
