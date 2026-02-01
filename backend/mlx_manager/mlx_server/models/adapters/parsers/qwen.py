"""Qwen tool call parser.

Parses Qwen (including Qwen2, Qwen2.5, Qwen3) Hermes-style tool calls:
- Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>

Qwen models use Hermes template for function calling.
Reference: Qwen Function Calling docs
"""

import json
import logging
import re
import uuid
from typing import Any

from mlx_manager.mlx_server.models.adapters.parsers.base import ToolCallParser

logger = logging.getLogger(__name__)

# Pattern for Hermes-style tool calls
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


class QwenToolParser(ToolCallParser):
    """Parser for Qwen Hermes-style tool calls.

    Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>

    Multiple tool calls can appear in a single response.
    """

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse Qwen tool calls from output text.

        Args:
            text: Model output text

        Returns:
            List of tool calls in OpenAI format
        """
        calls: list[dict[str, Any]] = []

        for match in TOOL_CALL_PATTERN.finditer(text):
            try:
                json_str = match.group(1).strip()
                data = json.loads(json_str)

                # Hermes format: {"name": "...", "arguments": {...}}
                name = data.get("name", "")
                arguments = data.get("arguments", {})

                # Arguments should be a JSON string in OpenAI format
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments)
                else:
                    arguments_str = str(arguments)

                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments_str,
                        },
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in Qwen tool call: %s", e)
                continue

        return calls

    def format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for Qwen prompt (Hermes style).

        Args:
            tools: List of tool definitions in OpenAI format

        Returns:
            Formatted string for system prompt
        """
        if not tools:
            return ""

        tool_docs: list[str] = []
        for tool in tools:
            func = tool.get("function", {})
            tool_docs.append(json.dumps(func, indent=2))

        # Build the prompt with proper line breaks for readability
        intro = (
            "You are a function calling AI model. You are provided with function "
            "signatures within <tools></tools> XML tags. You may call one or more "
            "functions to assist with the user query. Don't make assumptions about "
            "what values to plug into functions."
        )
        usage = (
            "For each function call, return a JSON object with function name and "
            "arguments within <tool_call></tool_call> XML tags as follows:"
        )

        return f"""{intro}

<tools>
{chr(10).join(tool_docs)}
</tools>

{usage}
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
</tool_call>"""
