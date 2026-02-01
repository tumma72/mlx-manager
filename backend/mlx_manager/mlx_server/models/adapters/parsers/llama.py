"""Llama tool call parser.

Parses Llama 3.x tool calling formats:
- XML style: <function=name>{"param": "value"}</function>
- Python style: <|python_tag|>tool.call(query="...")<|eom_id|>
- Array style: [func_name(param1='value1')]<|eot_id|>

Reference: Meta Llama Models prompt_format.md
"""

import json
import logging
import re
import uuid
from typing import Any

from mlx_manager.mlx_server.models.adapters.parsers.base import ToolCallParser

logger = logging.getLogger(__name__)

# Pattern for XML-style function calls: <function=name>{...}</function>
FUNCTION_PATTERN = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)

# Pattern for Python-style tool calls: tool.call(arg="value")
PYTHON_TOOL_PATTERN = re.compile(
    r"<\|python_tag\|>\s*(\w+)\.(\w+)\((.*?)\)\s*<\|eom_id\|>",
    re.DOTALL,
)


class LlamaToolParser(ToolCallParser):
    """Parser for Llama 3.x tool calling formats.

    Supports:
    - XML style: <function=get_weather>{"city": "SF"}</function>
    - Python style: <|python_tag|>weather.get(city="SF")<|eom_id|>
    """

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse Llama tool calls from output text.

        Args:
            text: Model output text

        Returns:
            List of tool calls in OpenAI format
        """
        calls: list[dict[str, Any]] = []

        # Try XML-style format first (most common)
        for match in FUNCTION_PATTERN.finditer(text):
            name = match.group(1)
            args_str = match.group(2).strip()

            try:
                # Validate JSON
                json.loads(args_str)
                calls.append(self._make_tool_call(name, args_str))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in Llama tool call %s: %s", name, e)
                # Still include with raw arguments
                calls.append(self._make_tool_call(name, args_str))

        # Try Python-style if no XML calls found
        if not calls:
            for match in PYTHON_TOOL_PATTERN.finditer(text):
                module = match.group(1)
                method = match.group(2)
                args_str = match.group(3).strip()

                # Convert Python-style args to JSON
                args_dict = self._parse_python_args(args_str)
                name = f"{module}.{method}"
                calls.append(self._make_tool_call(name, json.dumps(args_dict)))

        return calls

    def format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for Llama prompt.

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
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            doc = f"""{name}:
  description: {description}
  parameters: {json.dumps(parameters, indent=2)}"""
            tool_docs.append(doc)

        return f"""You have access to the following functions:

{chr(10).join(tool_docs)}

To call a function, respond with:
<function=function_name>{{"param": "value"}}</function>

Only call functions when necessary. If no function call is needed, respond normally."""

    def _make_tool_call(self, name: str, arguments: str) -> dict[str, Any]:
        """Create OpenAI-compatible tool call dict."""
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

    def _parse_python_args(self, args_str: str) -> dict[str, Any]:
        """Parse Python-style function arguments.

        Example: 'query="hello", limit=5' -> {"query": "hello", "limit": 5}
        """
        result: dict[str, Any] = {}
        if not args_str:
            return result

        # Simple parsing for key=value pairs
        # This handles basic cases; complex nested structures would need AST parsing
        pattern = re.compile(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\d+(?:\.\d+)?|\w+)')
        for match in pattern.finditer(args_str):
            key = match.group(1)
            value_str = match.group(2)

            # Parse value
            if value_str.startswith('"') or value_str.startswith("'"):
                value: Any = value_str[1:-1]  # Remove quotes
            elif value_str.isdigit():
                value = int(value_str)
            elif "." in value_str:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
            elif value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                value = value_str

            result[key] = value

        return result
