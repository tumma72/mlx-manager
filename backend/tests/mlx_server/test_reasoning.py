"""Tests for reasoning content extraction.

Tests the ReasoningExtractor for extracting chain-of-thought content
from model output tags like <think>, <thinking>, etc.
"""

from mlx_manager.mlx_server.services.reasoning import (
    THINKING_PATTERNS,
    ReasoningExtractor,
)


class TestReasoningExtractor:
    """Tests for ReasoningExtractor."""

    def test_extract_think_tags(self):
        """Extract content from <think> tags."""
        extractor = ReasoningExtractor()
        text = "<think>Let me analyze this problem step by step.</think>The answer is 42."

        reasoning, content = extractor.extract(text)

        assert reasoning == "Let me analyze this problem step by step."
        assert content == "The answer is 42."

    def test_extract_thinking_tags(self):
        """Extract content from <thinking> tags."""
        extractor = ReasoningExtractor()
        text = "<thinking>First, I'll consider the options.</thinking>Option B is best."

        reasoning, content = extractor.extract(text)

        assert reasoning == "First, I'll consider the options."
        assert content == "Option B is best."

    def test_extract_reasoning_tags(self):
        """Extract content from <reasoning> tags."""
        extractor = ReasoningExtractor()
        text = "<reasoning>Given the constraints...</reasoning>The solution is X."

        reasoning, content = extractor.extract(text)

        assert reasoning == "Given the constraints..."
        assert content == "The solution is X."

    def test_extract_reflection_tags(self):
        """Extract content from <reflection> tags."""
        extractor = ReasoningExtractor()
        text = "<reflection>Upon reflection, I see that...</reflection>My final answer is Y."

        reasoning, content = extractor.extract(text)

        assert reasoning == "Upon reflection, I see that..."
        assert content == "My final answer is Y."

    def test_extract_multiple_tags(self):
        """Extract and combine content from multiple thinking tags."""
        extractor = ReasoningExtractor()
        text = (
            "<think>First thought.</think>"
            "Middle content."
            "<think>Second thought.</think>"
            "Final answer."
        )

        reasoning, content = extractor.extract(text)

        assert "First thought." in reasoning
        assert "Second thought." in reasoning
        assert reasoning == "First thought.\nSecond thought."
        assert content == "Middle content.Final answer."

    def test_extract_no_reasoning_tags(self):
        """Return None for text without reasoning tags."""
        extractor = ReasoningExtractor()
        text = "This is just a regular response without any thinking."

        reasoning, content = extractor.extract(text)

        assert reasoning is None
        assert content == "This is just a regular response without any thinking."

    def test_extract_empty_tags(self):
        """Handle empty thinking tags."""
        extractor = ReasoningExtractor()
        text = "<think></think>The answer is here."

        reasoning, content = extractor.extract(text)

        # Empty tags result in empty reasoning
        assert reasoning == ""
        assert content == "The answer is here."

    def test_extract_multiline_reasoning(self):
        """Extract multiline content from thinking tags."""
        extractor = ReasoningExtractor()
        text = """<think>
Step 1: Consider the input.
Step 2: Apply the formula.
Step 3: Check the result.
</think>The result is 100."""

        reasoning, content = extractor.extract(text)

        assert "Step 1:" in reasoning
        assert "Step 2:" in reasoning
        assert "Step 3:" in reasoning
        assert content == "The result is 100."

    def test_extract_preserves_content_order(self):
        """Content before and after tags is preserved in order."""
        extractor = ReasoningExtractor()
        text = "Prefix text.<think>Reasoning here.</think>Suffix text."

        reasoning, content = extractor.extract(text)

        assert reasoning == "Reasoning here."
        assert content == "Prefix text.Suffix text."

    def test_extract_strips_whitespace(self):
        """Extracted content is stripped of leading/trailing whitespace."""
        extractor = ReasoningExtractor()
        text = "<think>  Some thinking with spaces  </think>   The answer.   "

        reasoning, content = extractor.extract(text)

        assert reasoning == "Some thinking with spaces"
        assert content == "The answer."

    def test_has_reasoning_tags_true(self):
        """has_reasoning_tags returns True when tags present."""
        extractor = ReasoningExtractor()

        assert extractor.has_reasoning_tags("<think>test</think>") is True
        assert extractor.has_reasoning_tags("<thinking>test</thinking>") is True
        assert extractor.has_reasoning_tags("<reasoning>test</reasoning>") is True
        assert extractor.has_reasoning_tags("<reflection>test</reflection>") is True

    def test_has_reasoning_tags_false(self):
        """has_reasoning_tags returns False when no tags."""
        extractor = ReasoningExtractor()

        assert extractor.has_reasoning_tags("Just regular text.") is False
        assert extractor.has_reasoning_tags("<other>tag</other>") is False
        assert extractor.has_reasoning_tags("<think>unclosed") is False


class TestAdapterReasoningSupport:
    """Tests for adapter reasoning mode support."""

    def test_llama_adapter_supports_reasoning_mode(self):
        """Llama adapter supports reasoning mode."""
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()

        assert adapter.supports_reasoning_mode() is True

    def test_qwen_adapter_supports_reasoning_mode(self):
        """Qwen adapter supports reasoning mode."""
        from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

        adapter = QwenAdapter()

        assert adapter.supports_reasoning_mode() is True

    def test_default_adapter_does_not_support_reasoning_mode(self):
        """Default adapter does not support reasoning mode."""
        from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

        adapter = DefaultAdapter()

        assert adapter.supports_reasoning_mode() is False

    def test_llama_adapter_extract_reasoning(self):
        """Llama adapter delegates to ReasoningExtractor."""
        from mlx_manager.mlx_server.models.adapters.llama import LlamaAdapter

        adapter = LlamaAdapter()
        text = "<think>My reasoning.</think>My answer."

        reasoning, content = adapter.extract_reasoning(text)

        assert reasoning == "My reasoning."
        assert content == "My answer."

    def test_qwen_adapter_extract_reasoning(self):
        """Qwen adapter delegates to ReasoningExtractor."""
        from mlx_manager.mlx_server.models.adapters.qwen import QwenAdapter

        adapter = QwenAdapter()
        text = "<thinking>Let me think...</thinking>Here's my response."

        reasoning, content = adapter.extract_reasoning(text)

        assert reasoning == "Let me think..."
        assert content == "Here's my response."

    def test_default_adapter_extract_reasoning_passthrough(self):
        """Default adapter returns text unchanged."""
        from mlx_manager.mlx_server.models.adapters.base import DefaultAdapter

        adapter = DefaultAdapter()
        text = "<think>test</think>content"

        reasoning, content = adapter.extract_reasoning(text)

        assert reasoning is None
        assert content == text


class TestThinkingPatterns:
    """Tests for thinking pattern definitions."""

    def test_patterns_are_compiled(self):
        """All thinking patterns are compiled regex patterns."""
        import re

        for pattern, name in THINKING_PATTERNS:
            assert isinstance(pattern, re.Pattern)
            assert isinstance(name, str)

    def test_patterns_match_expected_tags(self):
        """Each pattern matches its intended tag format."""
        test_cases = [
            ("<think>content</think>", "think"),
            ("<thinking>content</thinking>", "thinking"),
            ("<reasoning>content</reasoning>", "reasoning"),
            ("<reflection>content</reflection>", "reflection"),
        ]

        for text, expected_name in test_cases:
            matched = False
            for pattern, name in THINKING_PATTERNS:
                if pattern.search(text):
                    matched = True
                    assert name == expected_name
                    break
            assert matched, f"No pattern matched {text}"

    def test_patterns_use_dotall_flag(self):
        """Patterns can match multiline content."""
        for pattern, name in THINKING_PATTERNS:
            multiline_text = f"<{name}>\nline1\nline2\n</{name}>"
            match = pattern.search(multiline_text)
            assert match is not None, f"Pattern {name} should match multiline"
