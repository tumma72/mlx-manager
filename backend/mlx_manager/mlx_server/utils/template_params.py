"""Template parameter discovery for model-specific options.

Scans a tokenizer's Jinja chat template to discover configurable
parameters (e.g. ``enable_thinking``, ``keep_past_thinking``) and
returns structured metadata about each.

Detection uses regex-based scanning because Jinja2 silently ignores
unknown kwargs, making trial calls unreliable for discovery.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel


class TemplateParamInfo(BaseModel):
    """Metadata for a single discoverable template parameter."""

    name: str
    param_type: str  # "bool" or "string"
    default: Any
    label: str
    description: str


# Registry of known template parameters we look for.
# Only parameters listed here will be reported — this avoids surfacing
# internal variables that aren't meaningful to users.
KNOWN_TEMPLATE_PARAMS: dict[str, TemplateParamInfo] = {
    "enable_thinking": TemplateParamInfo(
        name="enable_thinking",
        param_type="bool",
        default=True,
        label="Enable Thinking",
        description=(
            "Allow the model to use chain-of-thought reasoning in <think> tags before answering."
        ),
    ),
    "keep_past_thinking": TemplateParamInfo(
        name="keep_past_thinking",
        param_type="bool",
        default=False,
        label="Keep Past Thinking",
        description="Include previous thinking content in multi-turn conversations.",
    ),
}

# Pattern 1: Explicit Jinja set declarations with defaults
# Matches: {%- set enable_thinking = enable_thinking | default(true) -%}
# Also:    {% set X = X | default(Y) %}
_SET_DEFAULT_RE = re.compile(
    r"\{%-?\s*set\s+(\w+)\s*=\s*\1\s*\|\s*default\(\s*(.*?)\s*\)\s*-?%\}",
)

# Pattern 2: Simple usage of known params (if X, {{ X }})
_USAGE_RE = re.compile(r"\{[%{]-?\s*(?:if\s+)?(\w+)\s*[%}]-?\}")


def _parse_jinja_default(raw: str) -> Any:
    """Parse a Jinja default() argument into a Python value."""
    lower = raw.strip().lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None
    # Try numeric
    try:
        return int(raw.strip())
    except ValueError:
        pass
    # String (strip quotes)
    stripped = raw.strip().strip("'\"")
    if stripped != raw.strip():
        return stripped
    return raw.strip()


def discover_template_params(tokenizer: Any) -> dict[str, TemplateParamInfo]:
    """Discover configurable template parameters from a tokenizer's chat template.

    Returns a dict mapping parameter name to its metadata. Only parameters
    that appear in both the template AND the ``KNOWN_TEMPLATE_PARAMS``
    registry are returned.

    Args:
        tokenizer: A HuggingFace tokenizer (or Processor wrapping one).

    Returns:
        Dict of discovered parameters with metadata.
    """
    actual = getattr(tokenizer, "tokenizer", tokenizer)
    template: str | None = getattr(actual, "chat_template", None)
    if not template or not isinstance(template, str):
        return {}

    found: dict[str, TemplateParamInfo] = {}

    # Pattern 1: Explicit set-with-default declarations
    for match in _SET_DEFAULT_RE.finditer(template):
        param_name = match.group(1)
        if param_name in KNOWN_TEMPLATE_PARAMS and param_name not in found:
            default_val = _parse_jinja_default(match.group(2))
            info = KNOWN_TEMPLATE_PARAMS[param_name].model_copy()
            info.default = default_val
            found[param_name] = info

    # Pattern 2: Usage references for known params not found by Pattern 1
    for match in _USAGE_RE.finditer(template):
        param_name = match.group(1)
        if param_name in KNOWN_TEMPLATE_PARAMS and param_name not in found:
            found[param_name] = KNOWN_TEMPLATE_PARAMS[param_name].model_copy()

    return found
