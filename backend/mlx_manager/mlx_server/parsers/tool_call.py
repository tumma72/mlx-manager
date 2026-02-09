"""Concrete tool call parser implementations."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from loguru import logger

from mlx_manager.mlx_server.parsers.base import ToolCallParser
from mlx_manager.mlx_server.schemas.openai import FunctionCall, ToolCall


class HermesJsonParser(ToolCallParser):
    """Hermes/Qwen JSON format: <tool_call>{"name": ..., "arguments": ...}</tool_call>

    Also handles unclosed variant where </tool_call> is missing.
    """

    # Closed: <tool_call>{...}</tool_call>
    _PATTERN_CLOSED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # Unclosed: <tool_call>{...}$ (some models like Qwen3 0.6B omit closing tag)
    _PATTERN_UNCLOSED = re.compile(r"<tool_call>\s*(\{.+\})\s*$", re.DOTALL | re.MULTILINE)

    @property
    def parser_id(self) -> str:
        return "hermes_json"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<tool_call>", "</tool_call>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        seen: set[str] = set()

        for pattern in (self._PATTERN_CLOSED, self._PATTERN_UNCLOSED):
            for match in pattern.finditer(text):
                tc = self._parse_match(match)
                if tc:
                    key = f"{tc.function.name}:{tc.function.arguments}"
                    if key not in seen:
                        seen.add(key)
                        results.append(tc)
        return results

    @staticmethod
    def _parse_match(match: re.Match[str]) -> ToolCall | None:
        try:
            json_str = match.group(1).strip()
            data = json.loads(json_str)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments)
            else:
                arguments_str = str(arguments)
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=FunctionCall(name=name, arguments=arguments_str),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Invalid Hermes tool call: {}", e)
            return None


class Glm4NativeParser(ToolCallParser):
    """GLM-4.7 native compact format: <tool_call>func_name<param>value</param>...

    Also handles the malformed attr variant: <tool_call>func_name<param="value"</param>
    """

    # Compact: <tool_call>func_name<param>value</param>...
    _PATTERN_COMPACT = re.compile(
        r"<tool_call>([^{<][^<]*(?:<\w+>[^<]*</\w+>)+)(?:</tool_call>|$)", re.DOTALL
    )
    # Attr: <tool_call>func_name<param="value"</param>...
    _PATTERN_ATTR = re.compile(
        r'<tool_call>(\w+(?:<\w+="[^"]*"</\w+>)+)\s*(?:</tool_call>|$)', re.DOTALL
    )

    @property
    def parser_id(self) -> str:
        return "glm4_native"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<tool_call>", "</tool_call>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        seen: set[str] = set()

        # Try compact format first
        for match in self._PATTERN_COMPACT.finditer(text):
            tc = self._parse_compact(match)
            if tc:
                key = f"{tc.function.name}:{tc.function.arguments}"
                if key not in seen:
                    seen.add(key)
                    results.append(tc)

        # Then try attr format
        for match in self._PATTERN_ATTR.finditer(text):
            tc = self._parse_attr(match)
            if tc:
                key = f"{tc.function.name}:{tc.function.arguments}"
                if key not in seen:
                    seen.add(key)
                    results.append(tc)

        return results

    @staticmethod
    def _parse_compact(match: re.Match[str]) -> ToolCall | None:
        try:
            content = match.group(1).strip()
            name_match = re.match(r"(\w+)", content)
            if not name_match:
                return None
            name = name_match.group(1)
            param_pattern = r"<(\w+)>([^<]*)</\1>"
            params = dict(re.findall(param_pattern, content))
            args_str = json.dumps(params) if params else "{}"
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=FunctionCall(name=name, arguments=args_str),
            )
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid GLM4.7 compact tool call: {}", e)
            return None

    @staticmethod
    def _parse_attr(match: re.Match[str]) -> ToolCall | None:
        try:
            content = match.group(1).strip()
            name_match = re.match(r"(\w+)", content)
            if not name_match:
                return None
            name = name_match.group(1)
            attr_pattern = r'<(\w+)="([^"]*)"(?:</\1>|/>|<)'
            params = dict(re.findall(attr_pattern, content))
            if not params:
                return None
            args_str = json.dumps(params)
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=FunctionCall(name=name, arguments=args_str),
            )
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid GLM4.7 attr tool call: {}", e)
            return None


class Glm4XmlParser(ToolCallParser):
    """GLM4 full XML format: <tool_call><name>fn</name><arguments>{...}</arguments></tool_call>"""

    _PATTERN = re.compile(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        re.DOTALL,
    )

    @property
    def parser_id(self) -> str:
        return "glm4_xml"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<tool_call>", "</tool_call>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        for match in self._PATTERN.finditer(text):
            tc = self._parse_match(match)
            if tc:
                results.append(tc)
        return results

    @staticmethod
    def _parse_match(match: re.Match[str]) -> ToolCall | None:
        try:
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                json.loads(args_str)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in GLM4 tool call {}: {}", name, e)
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=FunctionCall(name=name, arguments=args_str),
            )
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid GLM4 tool call: {}", e)
            return None


class LlamaXmlParser(ToolCallParser):
    """Llama XML format: <function=name>{...}</function>"""

    _PATTERN = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)

    @property
    def parser_id(self) -> str:
        return "llama_xml"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<function=", "</function>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        for match in self._PATTERN.finditer(text):
            tc = self._parse_match(match)
            if tc:
                results.append(tc)
        return results

    @staticmethod
    def _parse_match(match: re.Match[str]) -> ToolCall | None:
        try:
            name = match.group(1)
            args_str = match.group(2).strip()
            try:
                json.loads(args_str)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in Llama tool call {}: {}", name, e)
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=FunctionCall(name=name, arguments=args_str),
            )
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid Llama tool call: {}", e)
            return None


class LlamaPythonParser(ToolCallParser):
    """Llama Python format: <|python_tag|>module.method(args)<|eom_id|>"""

    _PATTERN = re.compile(r"<\|python_tag\|>\s*(\w+)\.(\w+)\((.*?)\)\s*<\|eom_id\|>", re.DOTALL)

    @property
    def parser_id(self) -> str:
        return "llama_python"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<|python_tag|>", "<|eom_id|>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        for match in self._PATTERN.finditer(text):
            tc = self._parse_match(match)
            if tc:
                results.append(tc)
        return results

    @staticmethod
    def _parse_match(match: re.Match[str]) -> ToolCall | None:
        try:
            module = match.group(1)
            method = match.group(2)
            args_str = match.group(3).strip()
            args_dict = _parse_python_args(args_str)
            name = f"{module}.{method}"
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=FunctionCall(name=name, arguments=json.dumps(args_dict)),
            )
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid Llama Python tool call: {}", e)
            return None


class NullToolParser(ToolCallParser):
    """No-op parser that never matches. Used for models without tool support."""

    @property
    def parser_id(self) -> str:
        return "null"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return []

    def extract(self, text: str) -> list[ToolCall]:
        return []


def _parse_python_args(args_str: str) -> dict[str, Any]:
    """Parse Python-style function arguments.

    Example: 'query="hello", limit=5' -> {"query": "hello", "limit": 5}
    """
    result: dict[str, Any] = {}
    if not args_str:
        return result

    pattern = re.compile(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\d+(?:\.\d+)?|\w+)')
    for match in pattern.finditer(args_str):
        key = match.group(1)
        value_str = match.group(2)

        if value_str.startswith('"') or value_str.startswith("'"):
            value: Any = value_str[1:-1]
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
