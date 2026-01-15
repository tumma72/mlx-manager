"""
Fuzzy string matching for model-to-parser option mapping.

This module implements fuzzy matching to dynamically find the best
parser option for a given model name, without requiring static mappings.

The matching strategy is:
1. First try exact substring match (e.g., "qwen3" in "Qwen3-8B-4bit")
2. For substring matches, prefer more specific options (qwen3_coder over qwen3)
3. Require the model name to contain the parser-specific part to match

Two implementations are provided:
- RapidfuzzMatcher: Uses the rapidfuzz library (C++ optimized, ~10x faster)
- DifflibMatcher: Uses Python's built-in difflib (no external dependencies)
"""

from abc import ABC, abstractmethod
from functools import lru_cache

from mlx_manager.services.parser_options import get_parser_options


class FuzzyMatcher(ABC):
    """Abstract base class for fuzzy matchers."""

    # Minimum partial_ratio score threshold (0-100) to accept a match
    # Set low since _contains_option_tokens already validates the match
    # This is just a secondary check to filter very weak matches
    PARTIAL_THRESHOLD = 50

    # For message_converter, we rely primarily on _contains_option_tokens
    # to ensure variant specificity (e.g., qwen3 won't match qwen3_coder)
    MESSAGE_CONVERTER_PARTIAL_THRESHOLD = 50

    @abstractmethod
    def _calculate_partial_score(self, s1: str, s2: str) -> float:
        """Calculate partial match score (0-100) - how well s2 is contained in s1."""

    def _normalize_for_matching(self, s: str) -> str:
        """Normalize string for matching: lowercase, replace underscores with hyphens."""
        return s.lower().replace("_", "-")

    def _extract_base_family(self, option: str) -> str:
        """Extract the base family name from a parser option.

        Examples:
            qwen3 -> qwen
            qwen3_coder -> qwen
            glm4_moe -> glm
            nemotron3_nano -> nemotron
            minimax_m2 -> minimax
            solar_open -> solar
            hermes -> hermes
        """
        option_lower = option.lower()
        # Take first part before underscore
        base = option_lower.split("_")[0]
        # Remove trailing digits (qwen3 -> qwen, glm4 -> glm)
        return "".join(c for c in base if c.isalpha())

    def _extract_variant(self, option: str) -> str | None:
        """Extract the variant suffix from a parser option.

        Examples:
            qwen3_coder -> coder
            qwen3_moe -> moe
            qwen3_vl -> vl
            glm4_moe -> moe
            minimax_m2 -> m2
            qwen3 -> None (no variant)
        """
        parts = option.lower().split("_")
        if len(parts) > 1:
            return parts[-1]  # Return last part as variant
        return None

    def _contains_option_tokens(self, model_lower: str, option: str) -> bool:
        """Check if model name contains the identifying tokens of the option.

        Strategy:
        1. The base family name MUST be present (qwen, glm, minimax, etc.)
        2. If the option has a variant (coder, moe, vl, m2), the variant must
           also be present, OR if no variant in model, match only base options.
        """
        base_family = self._extract_base_family(option)
        variant = self._extract_variant(option)

        # Base family must be present
        if base_family not in model_lower:
            return False

        # If option has a variant
        if variant:
            # Check if variant is in model name
            if variant in model_lower:
                return True
            # For "m2" variant, also check for "m2.1", "m2.x" patterns
            if variant == "m2":
                import re
                if re.search(r"m2[\.\d]?", model_lower):
                    return True
            # Variant not found - this option doesn't match
            return False

        # No variant - this is a base option (like "qwen3", "hermes")
        # Only match if the model doesn't have a more specific variant
        # that another option could match
        known_variants = ["coder", "moe", "vl", "nano", "open"]
        for kv in known_variants:
            if kv in model_lower:
                return False  # Let the more specific option match

        return True

    def _group_options_by_family(self, available: list[str]) -> dict[str, list[str]]:
        """Group options by their base family name."""
        groups: dict[str, list[str]] = {}
        for option in available:
            base = self._extract_base_family(option)
            if base not in groups:
                groups[base] = []
            groups[base].append(option)
        return groups

    def find_best_match(self, model_name: str, parser_type: str) -> str | None:
        """
        Find the best matching parser option for a model name.

        Matching strategy:
        1. Check for substring containment of option tokens in model name
        2. Among matching options, prefer longer/more specific matches
        3. If a family has only one option and base matches, use that option

        Args:
            model_name: Model identifier (e.g., "mlx-community/Qwen3-8B-4bit")
            parser_type: One of "tool_call_parser", "reasoning_parser", "message_converter"

        Returns:
            Best matching parser option, or None if no match above threshold.
        """
        options = get_parser_options()

        # Get available options for this parser type
        available = options.get(f"{parser_type}s", [])  # e.g., "tool_call_parsers"
        if not available:
            return None

        # Normalize model name for matching
        model_lower = self._normalize_for_matching(model_name)

        # Select threshold based on parser type
        threshold = (
            self.MESSAGE_CONVERTER_PARTIAL_THRESHOLD
            if parser_type == "message_converter"
            else self.PARTIAL_THRESHOLD
        )

        # Find all options that have their tokens contained in model name
        candidates = []
        for option in available:
            if self._contains_option_tokens(model_lower, option):
                # Calculate how well the option matches (partial score)
                score = self._calculate_partial_score(model_lower, option.lower())
                if score >= threshold:
                    candidates.append((option, score, len(option)))

        # If no direct matches, check for families with only one option
        # where the base family matches (e.g., GLM-4-9B -> glm4_moe)
        # EXCEPTION: Don't apply this fallback for message_converter since
        # it's critical to only set it when the model actually needs it
        # (e.g., base Qwen3 should NOT get qwen3_coder message_converter)
        if not candidates and parser_type != "message_converter":
            family_groups = self._group_options_by_family(available)
            for base_family, family_options in family_groups.items():
                if base_family in model_lower and len(family_options) == 1:
                    # Single option for this family, use it
                    option = family_options[0]
                    score = self._calculate_partial_score(model_lower, option.lower())
                    candidates.append((option, score, len(option)))

        if not candidates:
            return None

        # Sort by: 1) length descending (prefer more specific), 2) score descending
        # This ensures qwen3_coder is preferred over qwen3 when both match
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

        return candidates[0][0]


class RapidfuzzMatcher(FuzzyMatcher):
    """Fuzzy matcher using rapidfuzz library (C++ optimized).

    Uses partial_ratio which finds the best substring alignment.
    This is the recommended matcher for production use.
    """

    def _calculate_partial_score(self, s1: str, s2: str) -> float:
        from rapidfuzz import fuzz

        # partial_ratio finds the best partial match - ideal for finding
        # if s2 (option) appears as a substring-like pattern in s1 (model name)
        return fuzz.partial_ratio(s1, s2)


class DifflibMatcher(FuzzyMatcher):
    """Fuzzy matcher using Python's built-in difflib.

    Uses SequenceMatcher with partial matching logic.
    This is a fallback when rapidfuzz is not installed.
    """

    def _calculate_partial_score(self, s1: str, s2: str) -> float:
        from difflib import SequenceMatcher

        # Use SequenceMatcher's ratio for overall similarity
        # For partial matching, we check if the shorter string matches well
        matcher = SequenceMatcher(None, s1, s2)

        # Find longest contiguous match
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))

        # Score based on how much of the shorter string is matched
        shorter_len = min(len(s1), len(s2))
        if shorter_len == 0:
            return 0.0

        # Return percentage of shorter string that's matched
        return (match.size / len(s2)) * 100 if len(s2) > 0 else 0.0


@lru_cache(maxsize=1)
def get_matcher() -> FuzzyMatcher:
    """Get the configured fuzzy matcher instance.

    Prefers rapidfuzz for performance, falls back to difflib.
    """
    try:
        import rapidfuzz  # noqa: F401

        return RapidfuzzMatcher()
    except ImportError:
        return DifflibMatcher()


def find_parser_options(model_id: str) -> dict[str, str]:
    """
    Find recommended parser options for a model using fuzzy matching.

    This is the main entry point for the fuzzy matching system.
    It matches the model name against available parser options and
    returns the best matches above the threshold.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/Qwen3-8B-4bit")

    Returns:
        Dictionary with matched parser options (only non-None values).
        Keys are "tool_call_parser", "reasoning_parser", "message_converter".
    """
    matcher = get_matcher()

    result = {}
    for parser_type in ["tool_call_parser", "reasoning_parser", "message_converter"]:
        match = matcher.find_best_match(model_id, parser_type)
        if match:
            result[parser_type] = match

    return result
