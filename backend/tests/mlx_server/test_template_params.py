"""Tests for template parameter discovery."""

from mlx_manager.mlx_server.utils.template_params import (
    KNOWN_TEMPLATE_PARAMS,
    discover_template_params,
)


class MockTokenizer:
    """Simple mock tokenizer with configurable chat_template."""

    def __init__(self, chat_template: str | None):
        self.chat_template = chat_template


def test_qwen_template_with_enable_thinking():
    """Qwen template with enable_thinking parameter should be detected."""
    template = """
    {%- set enable_thinking = enable_thinking | default(true) -%}
    {% if enable_thinking %}
    <|im_start|>think
    {% endif %}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert "enable_thinking" in result
    assert result["enable_thinking"].name == "enable_thinking"
    assert result["enable_thinking"].param_type == "bool"
    assert result["enable_thinking"].default is True
    assert result["enable_thinking"].label == "Enable Thinking"
    assert "chain-of-thought" in result["enable_thinking"].description


def test_liquid_template_with_keep_past_thinking():
    """Liquid template with keep_past_thinking parameter should be detected."""
    template = """
    {%- set keep_past_thinking = keep_past_thinking | default(false) -%}
    {% if keep_past_thinking %}
    {{ past_thinking }}
    {% endif %}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert "keep_past_thinking" in result
    assert "enable_thinking" not in result
    assert result["keep_past_thinking"].name == "keep_past_thinking"
    assert result["keep_past_thinking"].param_type == "bool"
    assert result["keep_past_thinking"].default is False
    assert result["keep_past_thinking"].label == "Keep Past Thinking"
    assert "previous thinking" in result["keep_past_thinking"].description


def test_plain_template_without_params():
    """Plain template without any known parameters should return empty dict."""
    template = """
    {% for message in messages %}
    <|im_start|>{{ message.role }}
    {{ message.content }}<|im_end|>
    {% endfor %}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert result == {}


def test_no_template():
    """Tokenizer with no chat_template should return empty dict."""
    tokenizer = MockTokenizer(None)
    result = discover_template_params(tokenizer)

    assert result == {}


def test_empty_template():
    """Tokenizer with empty chat_template should return empty dict."""
    tokenizer = MockTokenizer("")
    result = discover_template_params(tokenizer)

    assert result == {}


def test_both_params_in_template():
    """Template with both enable_thinking and keep_past_thinking should detect both."""
    template = """
    {%- set enable_thinking = enable_thinking | default(true) -%}
    {%- set keep_past_thinking = keep_past_thinking | default(false) -%}
    {% if enable_thinking %}
    <think>
    {% if keep_past_thinking %}{{ past_thinking }}{% endif %}
    </think>
    {% endif %}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert len(result) == 2
    assert "enable_thinking" in result
    assert "keep_past_thinking" in result
    assert result["enable_thinking"].default is True
    assert result["keep_past_thinking"].default is False


def test_pattern2_detection_without_set_default():
    """Template using {% if enable_thinking %} without set...default detected via Pattern 2."""
    template = """
    {% for message in messages %}
    {% if enable_thinking %}
    <|im_start|>think
    {% endif %}
    <|im_start|>{{ message.role }}
    {{ message.content }}<|im_end|>
    {% endfor %}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert "enable_thinking" in result
    # Pattern 2 uses default from KNOWN_TEMPLATE_PARAMS registry
    assert result["enable_thinking"].default is True
    assert result["enable_thinking"].name == "enable_thinking"


def test_pattern2_with_variable_reference():
    """Template using {{ enable_thinking }} should be detected via Pattern 2."""
    template = """
    {% for message in messages %}
    <meta thinking="{{ enable_thinking }}">
    <|im_start|>{{ message.role }}
    {{ message.content }}<|im_end|>
    {% endfor %}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert "enable_thinking" in result
    assert result["enable_thinking"].default is True


def test_custom_default_value_overrides_registry():
    """Pattern 1 should override registry default with template's default value."""
    template = """
    {%- set enable_thinking = enable_thinking | default(false) -%}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert "enable_thinking" in result
    # Template specifies false, should override registry default of true
    assert result["enable_thinking"].default is False


def test_unknown_param_not_detected():
    """Unknown parameters not in KNOWN_TEMPLATE_PARAMS should be ignored."""
    template = """
    {%- set unknown_param = unknown_param | default(true) -%}
    {%- set enable_thinking = enable_thinking | default(true) -%}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert "enable_thinking" in result
    assert "unknown_param" not in result
    assert len(result) == 1


def test_tokenizer_wrapper_with_nested_tokenizer():
    """Handle processors/wrappers that have a nested tokenizer attribute."""

    class MockProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    template = "{%- set enable_thinking = enable_thinking | default(true) -%}"
    inner_tokenizer = MockTokenizer(template)
    processor = MockProcessor(inner_tokenizer)

    result = discover_template_params(processor)

    assert "enable_thinking" in result
    assert result["enable_thinking"].default is True


def test_pattern1_with_whitespace_variations():
    """Pattern 1 should handle various whitespace styles in Jinja templates."""
    variations = [
        "{%- set enable_thinking = enable_thinking | default(true) -%}",
        "{% set enable_thinking = enable_thinking | default(true) %}",
        "{%-set enable_thinking=enable_thinking|default(true)-%}",
        "{% set enable_thinking = enable_thinking | default( true ) %}",
    ]

    for template in variations:
        tokenizer = MockTokenizer(template)
        result = discover_template_params(tokenizer)
        assert "enable_thinking" in result, f"Failed for template: {template}"
        assert result["enable_thinking"].default is True


def test_default_value_parsing_booleans():
    """Test parsing of boolean default values in templates."""
    test_cases = [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
    ]

    for jinja_value, expected_python in test_cases:
        template = f"{{% set enable_thinking = enable_thinking | default({jinja_value}) %}}"
        tokenizer = MockTokenizer(template)
        result = discover_template_params(tokenizer)
        assert result["enable_thinking"].default is expected_python


def test_result_is_model_copy():
    """Each result should be an independent copy, not sharing references."""
    template = """
    {%- set enable_thinking = enable_thinking | default(true) -%}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    # Modifying result shouldn't affect the registry
    result["enable_thinking"].default = False
    assert KNOWN_TEMPLATE_PARAMS["enable_thinking"].default is True


def test_duplicate_param_definitions_use_first():
    """If a param appears multiple times, the first occurrence should be used."""
    template = """
    {%- set enable_thinking = enable_thinking | default(true) -%}
    {%- set enable_thinking = enable_thinking | default(false) -%}
    """
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    # Should use first occurrence (true)
    assert result["enable_thinking"].default is True


def test_default_value_parsing_none():
    """Test parsing of None default values in templates."""
    # Create a custom param that allows None default
    template = "{%- set enable_thinking = enable_thinking | default(none) -%}"
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert result["enable_thinking"].default is None


def test_default_value_parsing_integers():
    """Test parsing of integer default values in templates."""
    # Create a custom param that allows integer default
    template = "{%- set enable_thinking = enable_thinking | default(42) -%}"
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert result["enable_thinking"].default == 42


def test_default_value_parsing_strings():
    """Test parsing of quoted string default values in templates."""
    test_cases = [
        ("'hello'", "hello"),
        ('"world"', "world"),
        ("'quoted string'", "quoted string"),
    ]

    for jinja_value, expected_python in test_cases:
        template = f"{{% set enable_thinking = enable_thinking | default({jinja_value}) %}}"
        tokenizer = MockTokenizer(template)
        result = discover_template_params(tokenizer)
        # Note: This will use the parsed string as default, not the bool from registry
        assert result["enable_thinking"].default == expected_python


def test_default_value_parsing_unquoted_string():
    """Test parsing of unquoted string default values (edge case)."""
    # Unquoted strings are returned as-is
    template = "{%- set enable_thinking = enable_thinking | default(somevar) -%}"
    tokenizer = MockTokenizer(template)
    result = discover_template_params(tokenizer)

    assert result["enable_thinking"].default == "somevar"
