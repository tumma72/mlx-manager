"""Structured output validation service.

Validates model JSON output against user-provided JSON Schema definitions.
When users specify response_format with json_schema in their requests,
this service ensures the model output conforms to the specified schema.

Pattern reference: OpenAI Structured Outputs, 14-RESEARCH.md Pattern 4
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator, ValidationError


@dataclass
class ValidationResult:
    """Result of JSON Schema validation.

    Attributes:
        success: True if validation passed, False otherwise
        data: Parsed JSON data if validation succeeded, None otherwise
        error: Error message if validation failed, None otherwise
        error_path: JSON path to the element that caused validation failure,
                    None if validation succeeded or error was at root level
    """

    success: bool
    data: dict[str, Any] | list[Any] | None = None
    error: str | None = None
    error_path: str | None = None


class StructuredOutputValidator:
    """Validate model output against JSON Schema.

    This validator handles three main scenarios:
    1. Valid JSON that passes schema validation -> success with parsed data
    2. Valid JSON that fails schema validation -> error with path details
    3. Invalid JSON (parse error) -> error describing the JSON syntax issue

    Additionally, provides utilities for:
    - Extracting JSON from text that contains surrounding content
    - Type coercion for common LLM output issues (string "5" vs integer 5)

    Example:
        >>> validator = StructuredOutputValidator()
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> result = validator.validate('{"name": "Alice", "age": 30}', schema)
        >>> print(result.success)
        True
        >>> print(result.data)
        {"name": "Alice", "age": 30}
    """

    # Regex patterns for extracting JSON from mixed content
    # Matches outermost { } or [ ] with balanced brackets (simple approach)
    JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
    JSON_ARRAY_PATTERN = re.compile(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", re.DOTALL)

    def validate(self, output: str, schema: dict[str, Any]) -> ValidationResult:
        """Validate output string against JSON Schema.

        Attempts to parse the output as JSON and validate against the provided
        schema. Returns detailed error information including the path to the
        failing element when validation fails.

        Args:
            output: Model output string (should be valid JSON)
            schema: JSON Schema definition to validate against

        Returns:
            ValidationResult with success status and either parsed data or
            detailed error information including path to failing element
        """
        # Step 1: Parse JSON
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return ValidationResult(
                success=False,
                error=f"Invalid JSON: {e.msg} at position {e.pos}",
                error_path=None,
            )

        # Step 2: Validate against schema
        try:
            validator = Draft202012Validator(schema)
            validator.validate(data)
            return ValidationResult(success=True, data=data)
        except ValidationError as e:
            # Build path string from error path (list of keys/indices)
            error_path = "".join(
                f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.absolute_path
            )
            # Remove leading dot if present
            if error_path.startswith("."):
                error_path = error_path[1:]
            # Use "$" for root-level errors
            if not error_path:
                error_path = "$"

            return ValidationResult(
                success=False,
                error=f"Schema validation failed: {e.message}",
                error_path=error_path,
            )

    def extract_json(self, text: str) -> str | None:
        """Extract JSON from text that may have surrounding content.

        Models sometimes output explanatory text before or after JSON.
        This method attempts to find and extract the JSON portion.

        Looks for the first occurrence of content between { } or [ ],
        attempting to find valid JSON within the text.

        Args:
            text: Text that may contain JSON among other content

        Returns:
            Extracted JSON string if found and valid, None otherwise
        """
        # Try to find JSON object first
        object_match = self.JSON_OBJECT_PATTERN.search(text)
        if object_match:
            candidate = object_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        array_match = self.JSON_ARRAY_PATTERN.search(text)
        if array_match:
            candidate = array_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Try more aggressive extraction: find outermost braces/brackets
        # This handles nested structures better
        return self._extract_nested_json(text)

    def _extract_nested_json(self, text: str) -> str | None:
        """Extract nested JSON by tracking bracket depth.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string if valid, None otherwise
        """
        # Find first { or [
        start_obj = text.find("{")
        start_arr = text.find("[")

        # Determine which comes first
        if start_obj == -1 and start_arr == -1:
            return None

        if start_obj == -1:
            start = start_arr
            open_char, close_char = "[", "]"
        elif start_arr == -1:
            start = start_obj
            open_char, close_char = "{", "}"
        else:
            if start_obj < start_arr:
                start = start_obj
                open_char, close_char = "{", "}"
            else:
                start = start_arr
                open_char, close_char = "[", "]"

        # Track depth to find matching close
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        return None

        return None

    def validate_and_coerce(
        self, output: str, schema: dict[str, Any]
    ) -> ValidationResult:
        """Validate with type coercion for common LLM output issues.

        LLMs sometimes output values as strings when integers or booleans
        are expected (e.g., "5" instead of 5, "true" instead of true).
        This method attempts type coercion based on the schema before
        validation.

        Coercion rules:
        - String "5" -> integer 5 (for integer fields)
        - String "3.14" -> number 3.14 (for number fields)
        - String "true"/"false" -> boolean (for boolean fields)
        - String "null" -> None (for nullable fields)

        Args:
            output: Model output string
            schema: JSON Schema definition

        Returns:
            ValidationResult with coerced data if successful
        """
        # Step 1: Parse JSON
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return ValidationResult(
                success=False,
                error=f"Invalid JSON: {e.msg} at position {e.pos}",
                error_path=None,
            )

        # Step 2: Coerce types based on schema
        coerced_data = self._coerce_types(data, schema)

        # Step 3: Validate coerced data
        try:
            validator = Draft202012Validator(schema)
            validator.validate(coerced_data)
            return ValidationResult(success=True, data=coerced_data)
        except ValidationError as e:
            error_path = "".join(
                f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.absolute_path
            )
            if error_path.startswith("."):
                error_path = error_path[1:]
            if not error_path:
                error_path = "$"

            return ValidationResult(
                success=False,
                error=f"Schema validation failed after coercion: {e.message}",
                error_path=error_path,
            )

    def _coerce_types(self, data: Any, schema: dict[str, Any]) -> Any:
        """Recursively coerce types based on schema.

        Args:
            data: Data to coerce
            schema: Schema defining expected types

        Returns:
            Data with types coerced where possible
        """
        if schema is None:
            return data

        schema_type = schema.get("type")

        # Handle object properties
        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            result = {}
            for key, value in data.items():
                if key in properties:
                    result[key] = self._coerce_types(value, properties[key])
                else:
                    result[key] = value
            return result

        # Handle array items
        if schema_type == "array" and isinstance(data, list):
            items_schema = schema.get("items", {})
            return [self._coerce_types(item, items_schema) for item in data]

        # Coerce string to integer
        if schema_type == "integer" and isinstance(data, str):
            try:
                return int(data)
            except ValueError:
                return data

        # Coerce string to number
        if schema_type == "number" and isinstance(data, str):
            try:
                return float(data)
            except ValueError:
                return data

        # Coerce string to boolean
        if schema_type == "boolean" and isinstance(data, str):
            lower = data.lower()
            if lower == "true":
                return True
            elif lower == "false":
                return False
            return data

        # Coerce string "null" to None
        if isinstance(data, str) and data.lower() == "null":
            # Check if null is allowed in the schema
            if "null" in schema.get("type", []) if isinstance(
                schema.get("type"), list
            ) else False:
                return None

        return data
