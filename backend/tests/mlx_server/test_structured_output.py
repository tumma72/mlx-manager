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
