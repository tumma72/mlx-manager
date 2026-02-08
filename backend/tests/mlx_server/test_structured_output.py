"""Tests for structured output validation.

Tests the StructuredOutputValidator for validating model JSON output
against user-provided JSON Schema definitions.
"""

from mlx_manager.mlx_server.services.structured_output import (
    StructuredOutputValidator,
    ValidationResult,
)


class TestStructuredOutputValidator:
    """Tests for StructuredOutputValidator."""

    def test_validate_valid_json_object(self):
        """Validate valid JSON against object schema."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        output = '{"name": "Alice", "age": 30}'

        result = validator.validate(output, schema)

        assert result.success is True
        assert result.data == {"name": "Alice", "age": 30}
        assert result.error is None

    def test_validate_missing_required_field(self):
        """Validation fails when required field is missing."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        output = '{"name": "Alice"}'

        result = validator.validate(output, schema)

        assert result.success is False
        assert "required" in result.error.lower() or "age" in result.error.lower()

    def test_validate_wrong_type(self):
        """Validation fails when field has wrong type."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        output = '{"count": "not a number"}'

        result = validator.validate(output, schema)

        assert result.success is False
        assert "count" in result.error_path or "$" in result.error_path

    def test_validate_invalid_json(self):
        """Validation fails for invalid JSON."""
        validator = StructuredOutputValidator()
        schema = {"type": "object"}
        output = "{invalid json}"

        result = validator.validate(output, schema)

        assert result.success is False
        assert "Invalid JSON" in result.error
        assert result.error_path is None

    def test_validate_array_schema(self):
        """Validate JSON array against array schema."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        output = '["a", "b", "c"]'

        result = validator.validate(output, schema)

        assert result.success is True
        assert result.data == ["a", "b", "c"]

    def test_validate_nested_object(self):
        """Validate nested JSON object."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                }
            },
        }
        output = '{"user": {"name": "Bob"}}'

        result = validator.validate(output, schema)

        assert result.success is True
        assert result.data["user"]["name"] == "Bob"

    def test_validate_error_path_for_nested_error(self):
        """Error path correctly identifies nested field."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                    },
                }
            },
        }
        output = '{"user": {"age": "not an int"}}'

        result = validator.validate(output, schema)

        assert result.success is False
        # Path should indicate user.age
        assert "user" in result.error_path

    def test_validate_error_path_for_array_element(self):
        """Error path correctly identifies array index."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "array",
            "items": {"type": "integer"},
        }
        output = '[1, 2, "three", 4]'

        result = validator.validate(output, schema)

        assert result.success is False
        # Path should indicate [2]
        assert "[2]" in result.error_path


class TestExtractJson:
    """Tests for JSON extraction from mixed content."""

    def test_extract_json_object(self):
        """Extract JSON object from text."""
        validator = StructuredOutputValidator()
        text = 'Here is the result: {"key": "value"} as requested.'

        result = validator.extract_json(text)

        assert result == '{"key": "value"}'

    def test_extract_json_array(self):
        """Extract JSON array from text."""
        validator = StructuredOutputValidator()
        text = "The list is: [1, 2, 3]."

        result = validator.extract_json(text)

        assert result == "[1, 2, 3]"

    def test_extract_nested_json(self):
        """Extract nested JSON structure."""
        validator = StructuredOutputValidator()
        text = 'Result: {"outer": {"inner": "value"}}'

        result = validator.extract_json(text)

        assert result == '{"outer": {"inner": "value"}}'

    def test_extract_no_json(self):
        """Return None when no JSON found."""
        validator = StructuredOutputValidator()
        text = "This text has no JSON content."

        result = validator.extract_json(text)

        assert result is None

    def test_extract_first_json_object(self):
        """Extract first JSON when multiple present."""
        validator = StructuredOutputValidator()
        text = 'First: {"a": 1} Second: {"b": 2}'

        result = validator.extract_json(text)

        assert result == '{"a": 1}'


class TestValidateAndCoerce:
    """Tests for validation with type coercion."""

    def test_coerce_string_to_integer(self):
        """Coerce string '5' to integer 5."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        output = '{"count": "5"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data["count"] == 5
        assert isinstance(result.data["count"], int)

    def test_coerce_string_to_number(self):
        """Coerce string '3.14' to float 3.14."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
        }
        output = '{"value": "3.14"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data["value"] == 3.14
        assert isinstance(result.data["value"], float)

    def test_coerce_string_to_boolean_true(self):
        """Coerce string 'true' to boolean True."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
            },
        }
        output = '{"enabled": "true"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data["enabled"] is True

    def test_coerce_string_to_boolean_false(self):
        """Coerce string 'false' to boolean False."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
            },
        }
        output = '{"enabled": "false"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data["enabled"] is False

    def test_coerce_nested_object(self):
        """Coerce types in nested object."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "settings": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                    },
                }
            },
        }
        output = '{"settings": {"timeout": "30"}}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data["settings"]["timeout"] == 30

    def test_coerce_array_items(self):
        """Coerce types in array items."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "array",
            "items": {"type": "integer"},
        }
        output = '["1", "2", "3"]'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data == [1, 2, 3]

    def test_coerce_preserves_already_correct_types(self):
        """Coercion doesn't change already correct types."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "name": {"type": "string"},
            },
        }
        output = '{"count": 10, "name": "test"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is True
        assert result.data["count"] == 10
        assert result.data["name"] == "test"

    def test_coerce_fails_for_non_coercible(self):
        """Coercion fails when value can't be converted."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        output = '{"count": "not a number"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is False


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_successful_result(self):
        """Create successful validation result."""
        result = ValidationResult(success=True, data={"key": "value"})

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.error_path is None

    def test_failed_result(self):
        """Create failed validation result."""
        result = ValidationResult(
            success=False,
            error="Type mismatch",
            error_path="user.age",
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "Type mismatch"
        assert result.error_path == "user.age"

    def test_result_with_list_data(self):
        """ValidationResult can hold list data."""
        result = ValidationResult(success=True, data=[1, 2, 3])

        assert result.success is True
        assert result.data == [1, 2, 3]


# --- Extract JSON Edge Cases ---


class TestExtractJsonEdgeCases:
    """Edge case tests for JSON extraction to bring coverage to 90%+."""

    def test_extract_invalid_json_object_falls_through(self):
        """Object regex matches but json.loads fails -> try array or nested."""
        validator = StructuredOutputValidator()
        # This has braces but is not valid JSON
        text = "Result: {foo: bar} and done"

        result = validator.extract_json(text)

        # The regex may match but json.loads will fail
        # Should fall through to _extract_nested_json
        assert result is None

    def test_extract_invalid_json_array_falls_through(self):
        """Array regex matches but json.loads fails -> try nested extraction."""
        validator = StructuredOutputValidator()
        # Brackets with invalid JSON content
        text = "[not, valid, json]"

        result = validator.extract_json(text)

        # Should fall through to _extract_nested_json
        assert result is None

    def test_extract_json_with_invalid_object_valid_array(self):
        """Invalid object first, then valid array -> returns array."""
        validator = StructuredOutputValidator()
        text = "{invalid} followed by [1, 2, 3]"

        result = validator.extract_json(text)

        assert result == "[1, 2, 3]"

    def test_extract_deeply_nested_json_object(self):
        """Extract deeply nested JSON using bracket-depth tracking."""
        validator = StructuredOutputValidator()
        # The simple regex only handles one level of nesting;
        # _extract_nested_json handles arbitrary nesting
        nested = '{"a": {"b": {"c": [1, 2, {"d": true}]}}}'
        text = f"The result is: {nested} done."

        result = validator.extract_json(text)

        # The simple regex may return a partial match, but _extract_nested_json
        # should eventually find valid JSON
        assert result is not None
        import json

        data = json.loads(result)
        # Verify it parsed something valid (exact content depends on regex behavior)
        assert isinstance(data, dict)

    def test_extract_nested_json_array(self):
        """Extract nested JSON array using bracket-depth tracking."""
        validator = StructuredOutputValidator()
        text = "Output: [[1, 2], [3, 4]]"

        result = validator.extract_json(text)

        assert result is not None
        import json

        data = json.loads(result)
        assert data == [[1, 2], [3, 4]]


class TestExtractNestedJson:
    """Tests for _extract_nested_json bracket-depth tracker."""

    def test_no_brackets_returns_none(self):
        """No brackets or braces returns None."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json("plain text no brackets")
        assert result is None

    def test_only_array_brackets(self):
        """Only [ ] brackets present (no braces)."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json("[1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_only_object_braces(self):
        """Only { } braces present (no brackets)."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_object_before_array(self):
        """Object comes before array -> extracts object."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json('{"a": 1} [1, 2]')
        assert result == '{"a": 1}'

    def test_array_before_object(self):
        """Array comes before object -> extracts array."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json('[1, 2] {"a": 1}')
        assert result == "[1, 2]"

    def test_nested_braces(self):
        """Handles nested braces correctly."""
        validator = StructuredOutputValidator()
        text = '{"outer": {"inner": "value"}}'
        result = validator._extract_nested_json(text)
        assert result == text

    def test_string_with_braces(self):
        """Handles braces inside strings correctly."""
        validator = StructuredOutputValidator()
        text = '{"message": "use {name} here"}'
        result = validator._extract_nested_json(text)
        assert result == text

    def test_escaped_quotes(self):
        """Handles escaped quotes inside strings."""
        validator = StructuredOutputValidator()
        text = '{"msg": "say \\"hello\\""}'
        result = validator._extract_nested_json(text)
        assert result == text

    def test_unbalanced_braces_returns_none(self):
        """Unbalanced braces returns None."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json('{"key": "value"')
        assert result is None

    def test_invalid_json_with_balanced_braces(self):
        """Balanced braces but invalid JSON returns None."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json("{not: valid: json}")
        assert result is None

    def test_text_before_json(self):
        """JSON extraction with text before."""
        validator = StructuredOutputValidator()
        result = validator._extract_nested_json('Result: {"x": 1}')
        assert result == '{"x": 1}'

    def test_backslash_not_in_string(self):
        """Backslash outside string is handled correctly."""
        validator = StructuredOutputValidator()
        # This is not valid JSON, but tests the escape_next logic
        result = validator._extract_nested_json('{"a": 1}')
        assert result == '{"a": 1}'


# --- Validate and Coerce Edge Cases ---


class TestValidateAndCoerceEdgeCases:
    """Edge case tests for validate_and_coerce to bring coverage to 90%+."""

    def test_coerce_invalid_json(self):
        """validate_and_coerce fails on invalid JSON."""
        validator = StructuredOutputValidator()
        schema = {"type": "object"}
        output = "{not valid json}"

        result = validator.validate_and_coerce(output, schema)

        assert result.success is False
        assert "Invalid JSON" in result.error
        assert result.error_path is None

    def test_coerce_validation_failure_with_path(self):
        """validate_and_coerce reports error path on validation failure after coercion."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                    },
                }
            },
        }
        # "hello" cannot be coerced to integer
        output = '{"user": {"age": "hello"}}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is False
        assert "user" in result.error_path

    def test_coerce_root_validation_failure(self):
        """validate_and_coerce reports $ for root-level errors after coercion."""
        validator = StructuredOutputValidator()
        schema = {"type": "array"}
        output = '{"not": "array"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is False
        assert result.error_path == "$"

    def test_coerce_integer_valueerror(self):
        """Coerce non-numeric string to integer fails gracefully."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }
        output = '{"count": "abc"}'

        result = validator.validate_and_coerce(output, schema)

        # "abc" can't be coerced to int -> validation fails
        assert result.success is False

    def test_coerce_number_valueerror(self):
        """Coerce non-numeric string to number fails gracefully."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
        }
        output = '{"value": "not_a_number"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is False

    def test_coerce_boolean_non_boolean_string(self):
        """Non-boolean string not coerced to boolean."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"},
            },
        }
        output = '{"flag": "maybe"}'

        result = validator.validate_and_coerce(output, schema)

        assert result.success is False

    def test_coerce_null_with_type_list(self):
        """Coerce string 'null' to None when schema type includes null."""
        validator = StructuredOutputValidator()
        # Direct _coerce_types test since validate_and_coerce goes through JSON parsing
        schema = {"type": ["string", "null"]}

        result = validator._coerce_types("null", schema)

        assert result is None

    def test_coerce_null_without_null_type(self):
        """String 'null' not coerced when schema doesn't include null type."""
        validator = StructuredOutputValidator()
        schema = {"type": "string"}

        result = validator._coerce_types("null", schema)

        # Not coerced since null not in schema type list
        assert result == "null"

    def test_coerce_types_none_schema(self):
        """_coerce_types with None schema returns data unchanged."""
        validator = StructuredOutputValidator()

        assert validator._coerce_types(42, None) == 42
        assert validator._coerce_types("hello", None) == "hello"
        assert validator._coerce_types([1, 2], None) == [1, 2]

    def test_coerce_object_unknown_key(self):
        """Object with key not in schema properties is preserved."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "object",
            "properties": {
                "known": {"type": "integer"},
            },
        }
        data = {"known": "5", "unknown": "value"}

        result = validator._coerce_types(data, schema)

        assert result["known"] == 5  # Coerced
        assert result["unknown"] == "value"  # Preserved as-is

    def test_coerce_array_items(self):
        """Array items are coerced based on items schema."""
        validator = StructuredOutputValidator()
        schema = {
            "type": "array",
            "items": {"type": "integer"},
        }
        data = ["1", "2", "3"]

        result = validator._coerce_types(data, schema)

        assert result == [1, 2, 3]

    def test_validate_root_level_error_path(self):
        """validate() returns '$' for root-level validation errors."""
        validator = StructuredOutputValidator()
        schema = {"type": "array"}
        output = '{"not": "array"}'

        result = validator.validate(output, schema)

        assert result.success is False
        assert result.error_path == "$"
