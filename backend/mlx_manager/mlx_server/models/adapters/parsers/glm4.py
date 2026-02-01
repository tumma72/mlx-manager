"""GLM4 tool call parser.

Parses GLM4 XML-style tool calls:
- Format: <tool_call><name>func</name><arguments>{...}</arguments></tool_call>

GLM4 has a known issue where it sometimes outputs duplicate tool call markers.
This parser handles deduplication.

Reference: 14-RESEARCH.md Pitfall 6
"""

import hashlib
import json
import logging
import re
import uuid
from typing import Any

from mlx_manager.mlx_server.models.adapters.parsers.base import ToolCallParser

logger = logging.getLogger(__name__)

# Pattern for GLM4-style tool calls with nested elements
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
    re.DOTALL,
)

# Alternative pattern: some GLM4 versions use simpler JSON format
TOOL_CALL_JSON_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


class GLM4ToolParser(ToolCallParser):
    """Parser for GLM4 XML-style tool calls.

    Formats supported:
    - XML nested: <tool_call><name>func</name><arguments>{...}</arguments></tool_call>
    - JSON: <tool_call>{"name": "func", "arguments": {...}}</tool_call>

    Handles deduplication for GLM4's duplicate tag bug.
    """

    def parse(self, text: str) -> list[dict[str, Any]]:
        """Parse GLM4 tool calls from output text.

        Args:
            text: Model output text

        Returns:
            List of tool calls in OpenAI format (deduplicated)
        """
        calls: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()

        # Try XML nested format first
        for match in TOOL_CALL_PATTERN.finditer(text):
            name = match.group(1).strip()
            args_str = match.group(2).strip()

            # Deduplicate by content hash
            content_hash = hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()
            if content_hash in seen_hashes:
                logger.debug("Skipping duplicate GLM4 tool call: %s", name)
                continue
            seen_hashes.add(content_hash)

            try:
                # Validate JSON arguments
                json.loads(args_str)
                calls.append(self._make_tool_call(name, args_str))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in GLM4 tool call %s: %s", name, e)
                # Still include with raw arguments
                calls.append(self._make_tool_call(name, args_str))

        # If no XML format found, try JSON format
        if not calls:
            for match in TOOL_CALL_JSON_PATTERN.finditer(text):
                try:
                    json_str = match.group(1).strip()
                    data = json.loads(json_str)

                    name = data.get("name", "")
                    arguments = data.get("arguments", {})

                    if isinstance(arguments, dict):
                        args_str = json.dumps(arguments)
                    else:
                        args_str = str(arguments)

                    # Deduplicate by content hash
                    content_hash = hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()
                    if content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)

                    calls.append(self._make_tool_call(name, args_str))
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in GLM4 tool call: %s", e)
                    continue

        return calls

    def format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tools for GLM4 prompt.

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

            doc = f"""<tool>
<name>{name}</name>
<description>{description}</description>
<parameters>{json.dumps(parameters)}</parameters>
</tool>"""
            tool_docs.append(doc)

        return f"""You have access to the following tools:

{chr(10).join(tool_docs)}

When you need to call a tool, use this format:
<tool_call>
<name>tool_name</name>
<arguments>{{"param": "value"}}</arguments>
</tool_call>

Only call tools when necessary."""

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
