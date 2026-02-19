"""Template parameter discovery for model-specific options.

Scans a tokenizer's Jinja chat template to discover configurable
parameters (e.g. ``enable_thinking``, ``keep_past_thinking``) and
returns structured metadata about each.

Detection uses Jinja2's AST analysis (``jinja2.meta.find_undeclared_variables``)
to reliably discover all template parameters regardless of syntax pattern.
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

    Uses Jinja2 AST analysis to find all undeclared variables in the template,
    then filters to only known parameters. This handles all Jinja syntax
    patterns including ``is defined``, ``| default()``, and direct usage.

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

    from jinja2 import Environment
    from jinja2 import meta as jinja_meta

    try:
        ast = Environment().parse(template)
        variables = jinja_meta.find_undeclared_variables(ast)
    except Exception:
        variables = set()

    found: dict[str, TemplateParamInfo] = {}
    for var_name in sorted(variables):
        if var_name in KNOWN_TEMPLATE_PARAMS:
            info = KNOWN_TEMPLATE_PARAMS[var_name].model_copy()
            # Try to extract default from set...default() pattern
            match = re.search(
                rf"\bset\s+{re.escape(var_name)}\s*=\s*{re.escape(var_name)}\s*\|\s*default\(\s*(.*?)\s*\)",
                template,
            )
            if match:
                info.default = _parse_jinja_default(match.group(1))
            found[var_name] = info

    return found
