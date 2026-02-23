"""Concrete tool call parser implementations."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from loguru import logger

from mlx_manager.mlx_server.parsers.base import ToolCallParser
from mlx_manager.mlx_server.schemas.openai import ToolCall


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

        for pattern in (self._PATTERN_CLOSED, self._PATTERN_UNCLOSED):
            for match in pattern.finditer(text):
                tc = self._parse_match(match)
                if tc:
                    results.append(tc)
        return self._deduplicate(results)

    def _parse_match(self, match: re.Match[str]) -> ToolCall | None:
        try:
            json_str = match.group(1).strip()
            data = json.loads(json_str)
            name = data.get("name", "")
            arguments_str = self._coerce_arguments(data.get("arguments", {}))
            return self._make_tool_call(name, arguments_str)
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

        # Try compact format first
        for match in self._PATTERN_COMPACT.finditer(text):
            tc = self._parse_compact(match)
            if tc:
                results.append(tc)

        # Then try attr format
        for match in self._PATTERN_ATTR.finditer(text):
            tc = self._parse_attr(match)
            if tc:
                results.append(tc)

        return self._deduplicate(results)

    def _parse_compact(self, match: re.Match[str]) -> ToolCall | None:
        try:
            content = match.group(1).strip()
            name_match = re.match(r"(\w+)", content)
            if not name_match:
                return None
            name = name_match.group(1)
            param_pattern = r"<(\w+)>([^<]*)</\1>"
            params = dict(re.findall(param_pattern, content))
            args_str = json.dumps(params) if params else "{}"
            return self._make_tool_call(name, args_str)
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid GLM4.7 compact tool call: {}", e)
            return None

    def _parse_attr(self, match: re.Match[str]) -> ToolCall | None:
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
            return self._make_tool_call(name, args_str)
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

    def _parse_match(self, match: re.Match[str]) -> ToolCall | None:
        try:
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            if not args_str:
                logger.debug("Empty arguments in GLM4 tool call {}", name)
                return None
            try:
                json.loads(args_str)
            except json.JSONDecodeError as e:
                logger.debug("Invalid JSON in GLM4 tool call {}: {}", name, e)
                return None
            return self._make_tool_call(name, args_str)
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

    def _parse_match(self, match: re.Match[str]) -> ToolCall | None:
        try:
            name = match.group(1)
            args_str = match.group(2).strip()
            if not args_str:
                logger.debug("Empty arguments in Llama tool call {}", name)
                return None
            try:
                json.loads(args_str)
            except json.JSONDecodeError as e:
                logger.debug("Invalid JSON in Llama tool call {}: {}", name, e)
                return None
            return self._make_tool_call(name, args_str)
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

    def _parse_match(self, match: re.Match[str]) -> ToolCall | None:
        try:
            module = match.group(1)
            method = match.group(2)
            args_str = match.group(3).strip()
            args_dict = _parse_python_args(args_str)
            name = f"{module}.{method}"
            return self._make_tool_call(name, json.dumps(args_dict))
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid Llama Python tool call: {}", e)
            return None


class LiquidPythonParser(ToolCallParser):
    """LiquidAI format: <|tool_call_start|>[func(arg="val")]<|tool_call_end|>"""

    _PATTERN_CLOSED = re.compile(
        r"<\|tool_call_start\|\>\s*(.*?)\s*<\|tool_call_end\|\>", re.DOTALL
    )
    _PATTERN_UNCLOSED = re.compile(r"<\|tool_call_start\|\>\s*(.*?)\s*$", re.DOTALL)

    @property
    def parser_id(self) -> str:
        return "liquid_python"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<|tool_call_start|>", "<|tool_call_end|>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []

        for match in self._PATTERN_CLOSED.finditer(text):
            results.extend(self._parse_match(match))

        if not results:
            for match in self._PATTERN_UNCLOSED.finditer(text):
                results.extend(self._parse_match(match))

        return self._deduplicate(results)

    def _parse_match(self, match: re.Match[str]) -> list[ToolCall]:
        """Parse Pythonic function call syntax using ast.

        Format: [func_name(arg="val", arg2=123)]
        or: func_name(arg="val")
        """
        try:
            content = match.group(1).strip()
            if not content:
                return []

            # Parse as Python expression
            try:
                tree = ast.parse(content, mode="eval")
            except SyntaxError:
                logger.warning("Invalid Python syntax in Liquid tool call: {}", content)
                return []

            results: list[ToolCall] = []
            body = tree.body

            # Handle both list of calls and single call
            calls: list[ast.Call] = []
            if isinstance(body, ast.List):
                for elt in body.elts:
                    if isinstance(elt, ast.Call):
                        calls.append(elt)
            elif isinstance(body, ast.Call):
                calls.append(body)

            for call in calls:
                # Extract function name
                if isinstance(call.func, ast.Name):
                    name = call.func.id
                else:
                    logger.warning("Unsupported function type in Liquid tool call")
                    continue

                # Extract arguments
                args_dict: dict[str, Any] = {}
                for kw in call.keywords:
                    arg_name = kw.arg
                    if arg_name is None:
                        continue
                    try:
                        # Try literal_eval for safe evaluation
                        value = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        # Fallback to unparsing the AST node
                        value = ast.unparse(kw.value)
                    args_dict[arg_name] = value

                results.append(self._make_tool_call(name, json.dumps(args_dict)))

            return results
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid Liquid tool call: {}", e)
            return []


class MistralNativeParser(ToolCallParser):
    """Mistral native tool calling: [TOOL_CALLS] [{...}].

    Used by Mistral v3, Devstral, and other Mistral-family models.
    The model emits a [TOOL_CALLS] marker followed by a JSON array
    of tool calls with name, arguments (object), and optional id.
    """

    _PATTERN = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL)

    @property
    def parser_id(self) -> str:
        return "mistral_native"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("[TOOL_CALLS]", "")]

    def extract(self, text: str) -> list[ToolCall]:
        match = self._PATTERN.search(text)
        if not match:
            return []
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in Mistral tool calls: {}", e)
            return []
        if not isinstance(data, list):
            return []
        results: list[ToolCall] = []
        for item in data:
            tc = self._parse_item(item)
            if tc:
                results.append(tc)
        return results

    def _parse_item(self, item: dict[str, Any]) -> ToolCall | None:
        try:
            name = item.get("name", "")
            if not name:
                return None
            arguments = item.get("arguments", {})
            arguments_str = self._coerce_arguments(arguments)
            # Preserve model-generated ID if present, otherwise generate one
            return self._make_tool_call(name, arguments_str, call_id=item.get("id") or None)
        except (KeyError, TypeError) as e:
            logger.warning("Invalid Mistral tool call item: {}", e)
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


class OpenAIJsonParser(ToolCallParser):
    """Raw JSON tool call format (no wrapper tags).

    Handles two variants produced by models like Qwen3-VL:
    - {"type": "function", "function": {"name": ..., "arguments": ...}}
    - {"name": ..., "arguments": ...}
    """

    @property
    def parser_id(self) -> str:
        return "openai_json"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return []

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            # Find next '{' to attempt JSON decode
            brace_pos = text.find("{", idx)
            if brace_pos == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(text, brace_pos)
                idx = end_idx
            except json.JSONDecodeError:
                idx = brace_pos + 1
                continue

            if not isinstance(obj, dict):
                continue

            tc = self._parse_obj(obj)
            if tc:
                results.append(tc)

        return self._deduplicate(results)

    def _parse_obj(self, obj: dict[str, Any]) -> ToolCall | None:
        """Try to extract a tool call from a JSON object."""
        # Variant 1: {"type": "function", "function": {"name": ..., "arguments": ...}}
        if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
            func = obj["function"]
            name = func.get("name")
            if name:
                arguments = self._coerce_arguments(func.get("arguments", {}))
                return self._make_tool_call(name, arguments)

        # Variant 2: {"name": ..., "arguments": ...}
        if "name" in obj and "arguments" in obj:
            name = obj["name"]
            if isinstance(name, str) and name:
                arguments = self._coerce_arguments(obj["arguments"])
                return self._make_tool_call(name, arguments)

        return None


class ToolCodePythonParser(ToolCallParser):
    """Qwen3-Coder format: ```tool_code\\nfunc(arg="val")\\n```"""

    _PATTERN_CLOSED = re.compile(r"```tool_code\s*\n(.*?)\n\s*```", re.DOTALL)
    _PATTERN_UNCLOSED = re.compile(r"```tool_code\s*\n(.*?)\s*$", re.DOTALL)

    @property
    def parser_id(self) -> str:
        return "tool_code_python"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("```tool_code", "```")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []

        for pattern in (self._PATTERN_CLOSED, self._PATTERN_UNCLOSED):
            for match in pattern.finditer(text):
                results.extend(self._parse_match(match))

        return self._deduplicate(results)

    def _parse_match(self, match: re.Match[str]) -> list[ToolCall]:
        """Parse Python function call(s) from tool_code block using ast.

        Uses ast.parse + ast.literal_eval for safe, sandboxed parsing
        (same approach as LiquidPythonParser). No code execution occurs.
        """
        try:
            content = match.group(1).strip()
            if not content:
                return []

            try:
                tree = ast.parse(content, mode="eval")
            except SyntaxError:
                logger.warning("Invalid Python syntax in tool_code block: {}", content[:200])
                return []

            results: list[ToolCall] = []
            body = tree.body

            calls: list[ast.Call] = []
            if isinstance(body, ast.List):
                for elt in body.elts:
                    if isinstance(elt, ast.Call):
                        calls.append(elt)
            elif isinstance(body, ast.Call):
                calls.append(body)

            for call in calls:
                if isinstance(call.func, ast.Name):
                    name = call.func.id
                else:
                    continue

                args_dict: dict[str, Any] = {}
                for kw in call.keywords:
                    arg_name = kw.arg
                    if arg_name is None:
                        continue
                    try:
                        # ast.literal_eval safely evaluates literals only
                        value = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        # Fallback: unparse the AST node to string
                        value = ast.unparse(kw.value)
                    args_dict[arg_name] = value

                results.append(self._make_tool_call(name, json.dumps(args_dict)))

            return results
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid tool_code block: {}", e)
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


class Qwen3CoderXmlParser(ToolCallParser):
    """Qwen3-Coder / Nemotron XML format.

    Format:
        <tool_call>
        <function=name><parameter=key>value</parameter></function>
        </tool_call>

    Multiple functions can appear in a single <tool_call> block or in separate blocks.
    """

    _BLOCK_CLOSED = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _BLOCK_UNCLOSED = re.compile(r"<tool_call>(.*?)$", re.DOTALL)
    _FUNCTION = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
    _PARAMETER = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

    @property
    def parser_id(self) -> str:
        return "qwen3_coder_xml"

    @property
    def stream_markers(self) -> list[tuple[str, str]]:
        return [("<tool_call>", "</tool_call>")]

    def extract(self, text: str) -> list[ToolCall]:
        results: list[ToolCall] = []

        # Collect all block contents (closed first, then unclosed fallback)
        block_contents: list[str] = []
        for match in self._BLOCK_CLOSED.finditer(text):
            block_contents.append(match.group(1))
        if not block_contents:
            for match in self._BLOCK_UNCLOSED.finditer(text):
                block_contents.append(match.group(1))

        for content in block_contents:
            for func_match in self._FUNCTION.finditer(content):
                tc = self._parse_match(func_match)
                if tc:
                    results.append(tc)

        return self._deduplicate(results)

    def _parse_match(self, match: re.Match[str]) -> ToolCall | None:
        try:
            name = match.group(1)
            inner = match.group(2)

            # Extract parameters as key-value pairs
            params: dict[str, str] = {}
            for param_match in self._PARAMETER.finditer(inner):
                params[param_match.group(1)] = param_match.group(2)

            return self._make_tool_call(name, json.dumps(params))
        except (IndexError, AttributeError) as e:
            logger.warning("Invalid Qwen3-Coder XML tool call: {}", e)
            return None
