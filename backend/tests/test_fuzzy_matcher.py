"""Test-driven development for fuzzy parser matcher.

These tests define expected behavior and will be used to compare
rapidfuzz vs difflib libraries for matching model names to parser options.
"""

import pytest

# Test cases: (model_name, parser_type, expected_match, should_match)
# should_match=False means the best match score should be BELOW threshold

TOOL_PARSER_TEST_CASES = [
    # Exact/near-exact matches (should match)
    ("Qwen3-8B-4bit", "tool_call_parser", "qwen3", True),
    ("Qwen3-Coder-32B", "tool_call_parser", "qwen3_coder", True),
    ("Qwen3-MoE-30B", "tool_call_parser", "qwen3_moe", True),
    ("MiniMax-M2.1-3bit", "tool_call_parser", "minimax_m2", True),
    ("GLM-4-MoE", "tool_call_parser", "glm4_moe", True),
    ("Nemotron-3-8B", "tool_call_parser", "nemotron3_nano", True),
    ("Hermes-3-70B", "tool_call_parser", "hermes", True),
    ("Solar-10.7B", "tool_call_parser", "solar_open", True),
    # Models without specialized parsers (should NOT match above threshold)
    ("Llama-3.1-70B-4bit", "tool_call_parser", None, False),
    ("Mistral-7B-Instruct", "tool_call_parser", None, False),
    ("Phi-3-mini", "tool_call_parser", None, False),
]

REASONING_PARSER_TEST_CASES = [
    # Should match
    ("Qwen3-8B-4bit", "reasoning_parser", "qwen3", True),
    ("Qwen3-MoE-30B", "reasoning_parser", "qwen3_moe", True),
    ("MiniMax-M2.1-3bit", "reasoning_parser", "minimax_m2", True),
    ("GLM-4-MoE", "reasoning_parser", "glm4_moe", True),
    ("Hermes-3-70B", "reasoning_parser", "hermes", True),
    # Qwen3-Coder should NOT match reasoning (not in options)
    ("Qwen3-Coder-32B", "reasoning_parser", None, False),
]

MESSAGE_CONVERTER_TEST_CASES = [
    # Should match - models with explicit variant identifiers
    ("Qwen3-Coder-32B", "message_converter", "qwen3_coder", True),
    ("MiniMax-M2.1-3bit", "message_converter", "minimax_m2", True),
    # CRITICAL: Base models should NOT get message_converter even if there's only one option
    # because message_converter is critical for correctness - wrong converter causes errors
    ("GLM-4-9B", "message_converter", None, False),  # No "moe" in model name
    ("Nemotron-3-8B", "message_converter", None, False),  # No "nano" in model name
    ("Qwen3-8B-4bit", "message_converter", None, False),  # No "coder" in model name
    ("Qwen3-MoE-30B", "message_converter", None, False),  # "moe" present but no qwen_moe converter
    ("Qwen3-VL-7B", "message_converter", None, False),  # "vl" present but no qwen_vl converter
    # Other models without message converter
    ("Hermes-3-70B", "message_converter", None, False),
    ("Solar-10.7B", "message_converter", None, False),
    ("Llama-3.1-70B", "message_converter", None, False),
]

# Edge cases requiring careful threshold tuning
EDGE_CASES = [
    # qwen3 vs qwen3_coder - MUST distinguish correctly
    ("mlx-community/Qwen3-8B-Instruct-4bit", "message_converter", None, False),
    ("mlx-community/Qwen3-Coder-7B-4bit", "message_converter", "qwen3_coder", True),
    # qwen3 vs qwen3_moe
    ("mlx-community/Qwen3-30B-A3.3B-MoE-4bit", "tool_call_parser", "qwen3_moe", True),
    ("mlx-community/Qwen3-8B-4bit", "tool_call_parser", "qwen3", True),
    # minimax vs minimax_m2 (both are valid message converters)
    ("mlx-community/MiniMax-M2-3bit", "message_converter", "minimax_m2", True),
]

# Combine all test cases
ALL_TEST_CASES = (
    TOOL_PARSER_TEST_CASES + REASONING_PARSER_TEST_CASES + MESSAGE_CONVERTER_TEST_CASES + EDGE_CASES
)


class TestFuzzyMatcherRapidfuzz:
    """Test fuzzy matcher using rapidfuzz library."""

    @pytest.fixture
    def matcher(self):
        from mlx_manager.utils.fuzzy_matcher import RapidfuzzMatcher

        return RapidfuzzMatcher()

    @pytest.mark.parametrize("model,parser_type,expected,should_match", TOOL_PARSER_TEST_CASES)
    def test_tool_call_parser(self, matcher, model, parser_type, expected, should_match):
        result = matcher.find_best_match(model, parser_type)
        if should_match:
            assert result == expected, f"Expected {expected} for {model}, got {result}"
        else:
            assert result is None, f"Expected no match for {model}, got {result}"

    @pytest.mark.parametrize("model,parser_type,expected,should_match", REASONING_PARSER_TEST_CASES)
    def test_reasoning_parser(self, matcher, model, parser_type, expected, should_match):
        result = matcher.find_best_match(model, parser_type)
        if should_match:
            assert result == expected, f"Expected {expected} for {model}, got {result}"
        else:
            assert result is None, f"Expected no match for {model}, got {result}"

    @pytest.mark.parametrize(
        "model,parser_type,expected,should_match", MESSAGE_CONVERTER_TEST_CASES
    )
    def test_message_converter(self, matcher, model, parser_type, expected, should_match):
        result = matcher.find_best_match(model, parser_type)
        if should_match:
            assert result == expected, f"Expected {expected} for {model}, got {result}"
        else:
            assert result is None, f"Expected no match for {model}, got {result}"

    @pytest.mark.parametrize("model,parser_type,expected,should_match", EDGE_CASES)
    def test_edge_cases(self, matcher, model, parser_type, expected, should_match):
        result = matcher.find_best_match(model, parser_type)
        if should_match:
            assert result == expected, f"Expected {expected} for {model}, got {result}"
        else:
            assert result is None, f"Expected no match for {model}, got {result}"


# Note: DifflibMatcher tests were removed. During TDD comparison:
# - Rapidfuzz: 100% accuracy (32/32 test cases)
# - Difflib: 90.6% accuracy (29/32 test cases)
# Rapidfuzz was chosen as the production implementation.
# See DifflibMatcher class in fuzzy_matcher.py for the reference implementation.


class TestMatcherComparison:
    """Compare both matchers and report scores."""

    def test_compare_accuracy(self):
        """Run all test cases and compare which library performs better."""
        from mlx_manager.utils.fuzzy_matcher import DifflibMatcher, RapidfuzzMatcher

        rapidfuzz_correct = 0
        difflib_correct = 0
        rapidfuzz_failures = []
        difflib_failures = []

        rf_matcher = RapidfuzzMatcher()
        dl_matcher = DifflibMatcher()

        for model, parser_type, expected, should_match in ALL_TEST_CASES:
            rf_result = rf_matcher.find_best_match(model, parser_type)
            dl_result = dl_matcher.find_best_match(model, parser_type)

            # Check rapidfuzz
            if should_match:
                if rf_result == expected:
                    rapidfuzz_correct += 1
                else:
                    rapidfuzz_failures.append(
                        f"  {model} [{parser_type}]: expected={expected}, got={rf_result}"
                    )
            else:
                if rf_result is None:
                    rapidfuzz_correct += 1
                else:
                    rapidfuzz_failures.append(
                        f"  {model} [{parser_type}]: expected=None, got={rf_result}"
                    )

            # Check difflib
            if should_match:
                if dl_result == expected:
                    difflib_correct += 1
                else:
                    difflib_failures.append(
                        f"  {model} [{parser_type}]: expected={expected}, got={dl_result}"
                    )
            else:
                if dl_result is None:
                    difflib_correct += 1
                else:
                    difflib_failures.append(
                        f"  {model} [{parser_type}]: expected=None, got={dl_result}"
                    )

        total = len(ALL_TEST_CASES)
        print("\n" + "=" * 60)
        print("FUZZY MATCHER COMPARISON RESULTS")
        print("=" * 60)
        print(f"\nTotal test cases: {total}")
        print(f"\nRapidfuzz: {rapidfuzz_correct}/{total} ({100 * rapidfuzz_correct / total:.1f}%)")
        if rapidfuzz_failures:
            print("Rapidfuzz failures:")
            for f in rapidfuzz_failures:
                print(f)

        print(f"\nDifflib: {difflib_correct}/{total} ({100 * difflib_correct / total:.1f}%)")
        if difflib_failures:
            print("Difflib failures:")
            for f in difflib_failures:
                print(f)

        print("\n" + "=" * 60)
        if rapidfuzz_correct > difflib_correct:
            print("WINNER: Rapidfuzz")
        elif difflib_correct > rapidfuzz_correct:
            print("WINNER: Difflib")
        else:
            print("TIE - Prefer Rapidfuzz for performance")
        print("=" * 60)

        # Test passes - we just want to see the comparison
        assert True


class TestFindParserOptions:
    """Test the high-level find_parser_options function."""

    def test_qwen3_coder_gets_all_options(self):
        """Qwen3-Coder should get tool_call_parser and message_converter."""
        from mlx_manager.utils.fuzzy_matcher import find_parser_options

        result = find_parser_options("mlx-community/Qwen3-Coder-7B-4bit")
        assert "tool_call_parser" in result
        assert result["tool_call_parser"] == "qwen3_coder"
        assert "message_converter" in result
        assert result["message_converter"] == "qwen3_coder"

    def test_qwen3_base_no_message_converter(self):
        """Base Qwen3 should NOT get message_converter."""
        from mlx_manager.utils.fuzzy_matcher import find_parser_options

        result = find_parser_options("mlx-community/Qwen3-8B-4bit")
        # Should have tool_call_parser and reasoning_parser
        assert "tool_call_parser" in result
        assert result["tool_call_parser"] == "qwen3"
        # Should NOT have message_converter
        assert "message_converter" not in result

    def test_llama_gets_no_options(self):
        """Llama models shouldn't match any parser options."""
        from mlx_manager.utils.fuzzy_matcher import find_parser_options

        result = find_parser_options("mlx-community/Llama-3.1-70B-4bit")
        assert result == {}

    def test_minimax_m2_gets_all_options(self):
        """MiniMax M2 should get all three parser options (has m2 in name)."""
        from mlx_manager.utils.fuzzy_matcher import find_parser_options

        result = find_parser_options("mlx-community/MiniMax-M2.1-3bit")
        assert result.get("tool_call_parser") == "minimax_m2"
        assert result.get("reasoning_parser") == "minimax_m2"
        assert result.get("message_converter") == "minimax_m2"

    def test_base_minimax_gets_all_options(self):
        """Base MiniMax without M2 should get all options (minimax alias exists)."""
        from mlx_manager.utils.fuzzy_matcher import find_parser_options

        result = find_parser_options("mlx-community/MiniMax-Text-01-3bit")
        # Should get tool and reasoning via single-option fallback
        assert result.get("tool_call_parser") == "minimax_m2"
        assert result.get("reasoning_parser") == "minimax_m2"
        # Gets "minimax" (alias) for message_converter - this is valid
        assert result.get("message_converter") == "minimax"
