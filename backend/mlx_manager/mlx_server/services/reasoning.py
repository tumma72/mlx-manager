"""Reasoning content extraction service.

Extracts chain-of-thought content from model output tags like <think>, <thinking>,
<reasoning>, and <reflection>. Modern reasoning models (DeepSeek-R1, Qwen3-thinking,
Llama-thinking) output thinking content in these special tags that users may want
to access separately from the final answer.

Pattern reference: mlx-omni-server ThinkingDecoder pattern
"""

import re

# List of (pattern, tag_name) tuples for thinking/reasoning tags
# Pattern must use re.DOTALL flag for multiline content
THINKING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<think>(.*?)</think>", re.DOTALL), "think"),
    (re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL), "thinking"),
    (re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL), "reasoning"),
    (re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL), "reflection"),
]


class ReasoningExtractor:
    """Extract reasoning/thinking content from model output.

    Modern reasoning models like DeepSeek-R1, Qwen3-thinking, and Llama-thinking
    output chain-of-thought content in special XML-like tags. This extractor
    finds and extracts that content, returning it separately from the final answer.

    Supported tag formats:
    - <think>...</think> - DeepSeek-R1 style
    - <thinking>...</thinking> - Alternative thinking tag
    - <reasoning>...</reasoning> - Explicit reasoning tag
    - <reflection>...</reflection> - Reflection tag (some models)

    Example:
        >>> extractor = ReasoningExtractor()
        >>> text = "<think>Let me analyze this...</think>The answer is 42."
        >>> reasoning, content = extractor.extract(text)
        >>> print(reasoning)
        "Let me analyze this..."
        >>> print(content)
        "The answer is 42."
    """

    def extract(self, text: str) -> tuple[str | None, str]:
        """Extract reasoning content from response.

        Searches for all supported thinking/reasoning tag patterns,
        collects the reasoning content from matched tags, and removes
        the tags from the final content.

        Args:
            text: Model output text that may contain reasoning tags

        Returns:
            Tuple of (reasoning_content, final_content).
            reasoning_content is None if no reasoning tags found.
            final_content has all reasoning tags removed and is stripped.
        """
        reasoning_parts: list[str] = []
        content = text

        # Process each pattern
        for pattern, _tag_name in THINKING_PATTERNS:
            # Find all matches for this pattern
            matches = pattern.findall(content)
            for match in matches:
                # Collect reasoning content
                reasoning_parts.append(match.strip())

            # Remove matched tags from content
            content = pattern.sub("", content)

        # Combine reasoning parts with newlines if multiple found
        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

        return reasoning, content.strip()

    def has_reasoning_tags(self, text: str) -> bool:
        """Quick check if text contains any reasoning tags.

        Useful for early exit in adapters when reasoning extraction
        is not needed.

        Args:
            text: Text to check for reasoning tags

        Returns:
            True if text contains any supported reasoning tag patterns
        """
        for pattern, _tag_name in THINKING_PATTERNS:
            if pattern.search(text):
                return True
        return False
