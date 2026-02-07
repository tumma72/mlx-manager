#!/usr/bin/env python3
"""Model Capability Probe - Diagnostic tool for MLX model thinking & tool support.

Loads any MLX model, generates raw output with thinking-inducing and tool-inducing
prompts, and reports the exact tag structures used. Helps diagnose adapter issues
and identify when new model families use different tag formats.

Usage:
    python scripts/probe_model.py <model-id>
    python scripts/probe_model.py <model-id> --max-tokens 512
    python scripts/probe_model.py <model-id> --save-raw
    python scripts/probe_model.py <model-id> --thinking-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Add backend to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


# --- Terminal Colors ---


class Color:
    """ANSI color codes for terminal output."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        for attr in ("BOLD", "DIM", "GREEN", "YELLOW", "RED", "CYAN", "MAGENTA", "RESET"):
            setattr(cls, attr, "")


def header(text: str) -> str:
    rule = f"{Color.BOLD}{Color.CYAN}{'=' * 60}{Color.RESET}"
    return f"\n{rule}\n{Color.BOLD}{text}{Color.RESET}\n{rule}"


def section(text: str) -> str:
    return f"\n{Color.BOLD}{Color.YELLOW}--- {text} ---{Color.RESET}"


def ok(text: str) -> str:
    return f"{Color.GREEN}{text}{Color.RESET}"


def warn(text: str) -> str:
    return f"{Color.YELLOW}{text}{Color.RESET}"


def fail(text: str) -> str:
    return f"{Color.RED}{text}{Color.RESET}"


def dim(text: str) -> str:
    return f"{Color.DIM}{text}{Color.RESET}"


def label(name: str, value: str) -> str:
    return f"  {Color.BOLD}{name}:{Color.RESET} {value}"


# --- Tag Detection ---

KNOWN_THINKING_TAGS = ["think", "thinking", "reasoning", "reflection"]
XML_TAG_RE = re.compile(r"<(/?)([a-z_][a-z0-9_-]*)(?:\s[^>]*)?>", re.IGNORECASE)


def detect_xml_tags(text: str) -> dict[str, list[str]]:
    """Scan raw output for XML-style tags. Returns {tag_name: [open/close, ...]}."""
    tags: dict[str, list[str]] = {}
    for match in XML_TAG_RE.finditer(text):
        is_close = match.group(1) == "/"
        tag_name = match.group(2).lower()
        kind = "close" if is_close else "open"
        tags.setdefault(tag_name, []).append(kind)
    return tags


def classify_tags(
    tags: dict[str, list[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Classify detected tags into known-thinking, known-tool, and unknown."""
    known_thinking = []
    known_tool = []
    unknown = []
    tool_tags = {"tool_call", "function", "tool_use", "tool_result"}

    for tag_name in tags:
        if tag_name in KNOWN_THINKING_TAGS:
            known_thinking.append(tag_name)
        elif tag_name in tool_tags:
            known_tool.append(tag_name)
        else:
            unknown.append(tag_name)

    return known_thinking, known_tool, unknown


# --- Model Loading ---


def load_model(model_id: str) -> tuple[object, object]:
    """Load model and tokenizer via mlx_lm."""
    try:
        from mlx_lm import load  # type: ignore[import-untyped]
    except ImportError:
        print(fail("mlx-lm not installed. Run: pip install mlx-lm"))
        sys.exit(1)

    print(f"  Loading {Color.BOLD}{model_id}{Color.RESET} ...")
    start = time.time()
    model, tokenizer = load(model_id)
    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.1f}s")
    return model, tokenizer


def generate_raw(
    model: object,
    tokenizer: object,
    prompt_text: str,
    max_tokens: int,
) -> str:
    """Generate raw text from model using stream_generate."""
    from mlx_lm import stream_generate  # type: ignore[import-untyped]

    tokens: list[str] = []
    for response in stream_generate(  # type: ignore[call-overload]
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
    ):
        text = response.text if hasattr(response, "text") else str(response)
        tokens.append(text)

    return "".join(tokens)


# --- Phase 1: Model Info ---


def phase_model_info(model_id: str) -> tuple[str, object]:
    """Phase 1: Load model and report basic info (pre-load, no tokenizer yet)."""
    from mlx_manager.mlx_server.models.adapters.registry import (
        detect_model_family,
        get_adapter,
    )

    family = detect_model_family(model_id)
    adapter = get_adapter(model_id)

    print(header("Model Capability Probe"))
    print(label("Model", model_id))
    print(label("Family", f"{family} ({adapter.__class__.__name__})"))

    reasoning = ok("supported") if adapter.supports_reasoning_mode() else dim("not reported")
    print(label("Reasoning mode", reasoning))

    tools_status = ok("supported") if adapter.supports_tool_calling() else dim("not reported")
    print(label("Tool calling", tools_status))

    return family, adapter


def report_native_tool_support(adapter: object, tokenizer: object) -> None:
    """Report native tool support (requires tokenizer, called after load)."""
    try:
        has_native = adapter.has_native_tool_support(tokenizer)  # type: ignore[union-attr]
        native = ok("yes") if has_native else dim("no (prompt injection)")
    except Exception:
        native = dim("unknown")
    print(label("Native tool support", native))


# --- Phase 2: Template Inspection ---


def phase_template_inspection(tokenizer: object) -> None:
    """Phase 2: Inspect the tokenizer's chat template."""
    print(section("Chat Template Analysis"))

    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    template = getattr(actual_tokenizer, "chat_template", None)

    if template is None:
        print(label("Template", warn("none found")))
        return

    print(label("Template", ok("yes (Jinja2)")))
    print(label("Template length", f"{len(template)} chars"))

    # Check for thinking support
    has_thinking = "enable_thinking" in template or "thinking" in template.lower()
    print(label("enable_thinking ref", ok("yes") if has_thinking else warn("no")))

    # Check for tool sections
    has_tools = "tools" in template.lower() or "tool_call" in template.lower()
    print(label("Tool sections", ok("yes") if has_tools else warn("no")))

    # Show a compact preview
    lines = template.strip().split("\n")
    preview_lines = min(15, len(lines))
    print(f"\n  {dim('Template preview (first ' + str(preview_lines) + ' lines):')}")
    for line in lines[:preview_lines]:
        print(f"    {dim(line.rstrip()[:120])}")
    if len(lines) > preview_lines:
        print(f"    {dim(f'... ({len(lines) - preview_lines} more lines)')}")


# --- Phase 3: Thinking Test ---


def phase_thinking_test(
    model: object,
    tokenizer: object,
    adapter: object,
    family: str,
    max_tokens: int,
) -> str | None:
    """Phase 3: Test thinking output. Returns raw output."""
    print(section("Thinking Test"))

    thinking_prompt = "What is 15 * 27? Think step by step before giving the answer."
    messages = [{"role": "user", "content": thinking_prompt}]

    # Apply chat template
    try:
        prompt_text = adapter.apply_chat_template(  # type: ignore[union-attr]
            tokenizer, messages, add_generation_prompt=True
        )
    except Exception as e:
        print(f"  {fail('Template application failed:')} {e}")
        return None

    print(label("Prompt", '"What is 15 * 27? Think step by step..."'))
    print(f"  {dim('Generating...')}")

    raw_output = generate_raw(model, tokenizer, prompt_text, max_tokens)

    if not raw_output.strip():
        print(f"  {warn('Empty output from model')}")
        return raw_output

    # Display raw output (truncated)
    display_len = min(800, len(raw_output))
    print(f"\n  {Color.BOLD}Raw output ({len(raw_output)} chars):{Color.RESET}")
    for line in raw_output[:display_len].split("\n"):
        print(f"    {line}")
    if len(raw_output) > display_len:
        print(f"    {dim(f'... ({len(raw_output) - display_len} more chars)')}")

    # Detect tags
    tags = detect_xml_tags(raw_output)
    known_thinking, _, unknown = classify_tags(tags)

    print(f"\n  {Color.BOLD}Tag analysis:{Color.RESET}")
    if known_thinking:
        print(f"    Thinking tags: {ok(', '.join(f'<{t}>' for t in known_thinking))}")
    else:
        print(f"    Thinking tags: {warn('none detected')}")

    if unknown:
        # Filter out common noise tags (im_start, etc.)
        interesting = [t for t in unknown if not t.startswith("im_") and not t.startswith("|")]
        if interesting:
            print(f"    Unknown tags:  {warn(', '.join(f'<{t}>' for t in interesting))}")

    # Test our parser
    from mlx_manager.mlx_server.services.response_processor import (
        create_processor_for_family,
    )

    processor = create_processor_for_family(family)
    result = processor.process(raw_output)

    print(f"\n  {Color.BOLD}Parser result:{Color.RESET}")
    if result.reasoning:
        print(f"    Reasoning extracted: {ok('YES')}")
        preview = result.reasoning[:200]
        print(f"    Reasoning preview:   {dim(preview)}")
    else:
        print(f"    Reasoning extracted: {fail('NO')}")

    if result.content.strip():
        print(f"    Cleaned content:     {dim(result.content.strip()[:200])}")

    return raw_output


# --- Phase 4: Tool Calling Test ---

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use",
                },
            },
            "required": ["location"],
        },
    },
}


def phase_native_tool_test(
    model: object,
    tokenizer: object,
    adapter: object,
    family: str,
    max_tokens: int,
) -> str | None:
    """Phase 3b: Test native tool calling (tools passed to template). Returns raw output."""
    from mlx_manager.mlx_server.utils.template_tools import has_native_tool_support

    print(section("Native Tool Test"))

    if not has_native_tool_support(tokenizer):
        print(f"  {dim('Skipped - tokenizer does not accept tools= parameter')}")
        return None

    tools = [TOOL_DEFINITION]
    user_msg = "What is the current weather in Tokyo? Use the get_weather tool to find out."
    messages = [{"role": "user", "content": user_msg}]

    # Apply chat template with tools passed natively
    try:
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        from typing import cast

        prompt_text: str = cast(
            str,
            actual_tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
    except Exception as e:
        print(f"  {fail('Native template application failed:')} {e}")
        return None

    print(label("Prompt", '"What is the current weather in Tokyo?"'))
    print(label("Tools", "[get_weather(location, unit)] via native template"))
    print(f"  {dim('Generating...')}")

    raw_output = generate_raw(model, tokenizer, prompt_text, max_tokens)

    if not raw_output.strip():
        print(f"  {warn('Empty output from model')}")
        return raw_output

    # Display raw output
    display_len = min(800, len(raw_output))
    print(f"\n  {Color.BOLD}Raw output ({len(raw_output)} chars):{Color.RESET}")
    for line in raw_output[:display_len].split("\n"):
        print(f"    {line}")
    if len(raw_output) > display_len:
        print(f"    {dim(f'... ({len(raw_output) - display_len} more chars)')}")

    # Detect tags
    tags = detect_xml_tags(raw_output)
    _, known_tool, unknown = classify_tags(tags)

    print(f"\n  {Color.BOLD}Tag analysis:{Color.RESET}")
    if known_tool:
        print(f"    Tool tags: {ok(', '.join(f'<{t}>' for t in known_tool))}")
    else:
        print(f"    Tool tags: {warn('none detected')}")

    # Check for JSON-like structures
    json_matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*\}', raw_output)
    if json_matches:
        print(f"    JSON tool-like: {ok(f'{len(json_matches)} found')}")
        for jm in json_matches[:3]:
            print(f"      {dim(jm[:150])}")

    # Test our parser
    from mlx_manager.mlx_server.services.response_processor import (
        create_processor_for_family,
    )

    processor = create_processor_for_family(family)
    result = processor.process(raw_output)

    print(f"\n  {Color.BOLD}Parser result:{Color.RESET}")
    if result.tool_calls:
        print(f"    Tool calls extracted: {ok(f'YES ({len(result.tool_calls)})')}")
        for tc in result.tool_calls:
            print(f"      {ok(tc.function.name)}({dim(tc.function.arguments)})")
    else:
        print(f"    Tool calls extracted: {fail('NO')}")

    return raw_output


def phase_tool_test(
    model: object,
    tokenizer: object,
    adapter: object,
    family: str,
    max_tokens: int,
) -> str | None:
    """Phase 4: Test tool calling via prompt injection. Returns raw output."""
    print(section("Injected Tool Calling Test"))

    tools = [TOOL_DEFINITION]
    tools_system = adapter.format_tools_for_prompt(tools)  # type: ignore[union-attr]

    user_msg = "What is the current weather in Tokyo? Use the get_weather tool to find out."
    messages = [{"role": "user", "content": user_msg}]

    if tools_system:
        messages.insert(0, {"role": "system", "content": tools_system})

    # Apply chat template
    try:
        prompt_text = adapter.apply_chat_template(  # type: ignore[union-attr]
            tokenizer, messages, add_generation_prompt=True
        )
    except Exception as e:
        print(f"  {fail('Template application failed:')} {e}")
        return None

    print(label("Prompt", '"What is the current weather in Tokyo?"'))
    print(label("Tools", "[get_weather(location, unit)]"))
    print(f"  {dim('Generating...')}")

    raw_output = generate_raw(model, tokenizer, prompt_text, max_tokens)

    if not raw_output.strip():
        print(f"  {warn('Empty output from model')}")
        return raw_output

    # Display raw output
    display_len = min(800, len(raw_output))
    print(f"\n  {Color.BOLD}Raw output ({len(raw_output)} chars):{Color.RESET}")
    for line in raw_output[:display_len].split("\n"):
        print(f"    {line}")
    if len(raw_output) > display_len:
        print(f"    {dim(f'... ({len(raw_output) - display_len} more chars)')}")

    # Detect tags
    tags = detect_xml_tags(raw_output)
    _, known_tool, unknown = classify_tags(tags)

    print(f"\n  {Color.BOLD}Tag analysis:{Color.RESET}")
    if known_tool:
        print(f"    Tool tags: {ok(', '.join(f'<{t}>' for t in known_tool))}")
    else:
        print(f"    Tool tags: {warn('none detected')}")

    # Check for JSON-like structures
    json_matches = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*\}', raw_output)
    if json_matches:
        print(f"    JSON tool-like: {ok(f'{len(json_matches)} found')}")
        for jm in json_matches[:3]:
            print(f"      {dim(jm[:150])}")

    if unknown:
        interesting = [t for t in unknown if not t.startswith("im_") and not t.startswith("|")]
        if interesting:
            print(f"    Unknown tags:  {warn(', '.join(f'<{t}>' for t in interesting))}")

    # Test our parser
    from mlx_manager.mlx_server.services.response_processor import (
        create_processor_for_family,
    )

    processor = create_processor_for_family(family)
    result = processor.process(raw_output)

    print(f"\n  {Color.BOLD}Parser result:{Color.RESET}")
    if result.tool_calls:
        print(f"    Tool calls extracted: {ok(f'YES ({len(result.tool_calls)})')}")
        for tc in result.tool_calls:
            print(f"      {ok(tc.function.name)}({dim(tc.function.arguments)})")
    else:
        print(f"    Tool calls extracted: {fail('NO')}")

    if result.content.strip():
        print(f"    Cleaned content:     {dim(result.content.strip()[:200])}")

    return raw_output


# --- Phase 5: Summary ---


def phase_summary(
    family: str,
    thinking_output: str | None,
    tool_output: str | None,
    native_tool_output: str | None = None,
) -> None:
    """Phase 5: Print summary report."""
    from mlx_manager.mlx_server.services.response_processor import (
        create_processor_for_family,
    )

    print(header("Summary"))

    # Thinking assessment
    thinking_tags_found: list[str] = []
    thinking_parser_ok = False
    if thinking_output:
        tags = detect_xml_tags(thinking_output)
        thinking_tags_found, _, _ = classify_tags(tags)
        processor = create_processor_for_family(family)
        result = processor.process(thinking_output)
        thinking_parser_ok = result.reasoning is not None

    if thinking_tags_found:
        status = ok("SUPPORTED") if thinking_parser_ok else warn("DETECTED but parser fails")
        tag_list = ", ".join(f"<{t}>" for t in thinking_tags_found)
        print(label("Thinking", f"{status} (tags: {tag_list})"))
    elif thinking_output:
        msg = "no standard tags detected - model may use different format"
        print(label("Thinking", warn(msg)))
    else:
        print(label("Thinking", dim("not tested")))

    # Tool calling assessment
    tool_tags_found: list[str] = []
    tool_parser_ok = False
    if tool_output:
        tags = detect_xml_tags(tool_output)
        _, tool_tags_found, _ = classify_tags(tags)
        processor = create_processor_for_family(family)
        result = processor.process(tool_output)
        tool_parser_ok = len(result.tool_calls) > 0

    if tool_tags_found:
        status = ok("SUPPORTED") if tool_parser_ok else warn("DETECTED but parser fails")
        tag_list = ", ".join(f"<{t}>" for t in tool_tags_found)
        print(label("Tool calling", f"{status} (tags: {tag_list})"))
    elif tool_output:
        msg = "no standard tags detected - model may use different format"
        print(label("Tool calling", warn(msg)))
    else:
        print(label("Tool calling", dim("not tested")))

    # Native tool assessment
    native_tool_tags_found: list[str] = []
    native_tool_parser_ok = False
    if native_tool_output:
        tags = detect_xml_tags(native_tool_output)
        _, native_tool_tags_found, _ = classify_tags(tags)
        processor = create_processor_for_family(family)
        result = processor.process(native_tool_output)
        native_tool_parser_ok = len(result.tool_calls) > 0

    if native_tool_output is not None:
        if native_tool_tags_found:
            n_status = (
                ok("SUPPORTED") if native_tool_parser_ok else warn("DETECTED but parser fails")
            )
            tag_list = ", ".join(f"<{t}>" for t in native_tool_tags_found)
            print(label("Native tools", f"{n_status} (tags: {tag_list})"))
        elif native_tool_output:
            msg = "no standard tags detected - model may use different format"
            print(label("Native tools", warn(msg)))
        else:
            print(label("Native tools", dim("not tested")))

    # Recommendation
    if native_tool_parser_ok and tool_parser_ok:
        print(f"\n  {ok('Recommendation: Use native tools (preferred)')}")
    elif native_tool_parser_ok:
        print(f"\n  {ok('Recommendation: Use native tools (injection not tested or fails)')}")
    elif native_tool_output is not None and not native_tool_parser_ok and tool_parser_ok:
        print(f"\n  {warn('Recommendation: Use prompt injection (native output not parsed)')}")

    # Overall parser compatibility
    print()
    if thinking_parser_ok and (tool_parser_ok or tool_output is None):
        print(f"  {ok('Parser compatible: YES - no changes needed')}")
    elif not thinking_parser_ok and thinking_output:
        print(f"  {fail('Parser compatible: NO')}")
        print(f"  {warn('Action needed: inspect raw output above for tag patterns, then update')}")
        print(f"    {dim('response_processor.py COMMON_THINKING_TAGS or add new patterns')}")
    elif not tool_parser_ok and tool_output:
        print(f"  {warn('Parser compatible: PARTIAL (thinking ok, tool calls need work)')}")


# --- Save Raw Output ---


def save_raw_outputs(
    model_id: str,
    thinking_output: str | None,
    tool_output: str | None,
    native_tool_output: str | None = None,
) -> None:
    """Save raw outputs to probe_output/ directory."""
    safe_name = model_id.replace("/", "__")
    out_dir = Path("probe_output") / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if thinking_output is not None:
        path = out_dir / "thinking_raw.txt"
        path.write_text(thinking_output, encoding="utf-8")
        print(label("Saved", str(path)))

    if native_tool_output is not None:
        path = out_dir / "native_tool_raw.txt"
        path.write_text(native_tool_output, encoding="utf-8")
        print(label("Saved", str(path)))

    if tool_output is not None:
        path = out_dir / "injected_tool_raw.txt"
        path.write_text(tool_output, encoding="utf-8")
        print(label("Saved", str(path)))


# --- API Probe ---


def run_api_probe(model_id: str, base_url: str = "http://localhost:10242") -> None:
    """Probe model via API endpoint."""
    import json
    import urllib.error
    import urllib.request

    token = os.environ.get("MLX_MANAGER_TOKEN", "")
    if not token:
        print(warn("Set MLX_MANAGER_TOKEN env var for authentication"))
        print(dim("Get a token from the MLX Manager UI or API"))
        return

    url = f"{base_url}/api/models/probe/{model_id}?token={token}"

    print(header("Model Capability Probe (API)"))
    print(label("Model", model_id))
    print(label("Server", base_url))
    print()

    # Make POST request and consume SSE stream
    req = urllib.request.Request(url, method="POST")
    req.add_header("Accept", "text/event-stream")

    try:
        with urllib.request.urlopen(req) as response:
            capabilities: dict[str, bool | str] = {}
            for line_bytes in response:
                line = line_bytes.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                step = data.get("step", "")
                status = data.get("status", "")
                capability = data.get("capability")
                value = data.get("value")
                error = data.get("error")

                if status == "running":
                    print(f"  {Color.YELLOW}⟳{Color.RESET} {step}...")
                elif status == "completed":
                    detail = ""
                    if capability and value is not None:
                        capabilities[capability] = value
                        detail = f" → {ok(str(value)) if value else fail(str(value))}"
                    print(f"  {Color.GREEN}✓{Color.RESET} {step}{detail}")
                elif status == "failed":
                    print(f"  {Color.RED}✗{Color.RESET} {step}: {error or 'unknown error'}")
                elif status == "skipped":
                    print(f"  {Color.DIM}-{Color.RESET} {step} (skipped)")

            # Summary
            print(section("Capabilities"))
            for cap, val in capabilities.items():
                status_str = ok("YES") if val else fail("NO")
                print(
                    label(
                        cap.replace("_", " ").title(),
                        status_str if isinstance(val, bool) else str(val),
                    )
                )

    except urllib.error.HTTPError as e:
        print(fail(f"API error: {e.code} {e.reason}"))
        if e.code == 401:
            print(dim("Check your MLX_MANAGER_TOKEN"))
    except urllib.error.URLError as e:
        print(fail(f"Connection error: {e.reason}"))
        print(dim(f"Is MLX Manager running at {base_url}?"))


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe an MLX model's thinking and tool calling capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/probe_model.py mlx-community/Qwen3-0.6B-4bit-DWQ
  python scripts/probe_model.py lmstudio-community/Qwen3-Coder-Next-MLX-4bit --max-tokens 512
  python scripts/probe_model.py <model-id> --save-raw --thinking-only
        """,
    )
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens (default: 256)")
    parser.add_argument("--save-raw", action="store_true", help="Save raw outputs to probe_output/")
    parser.add_argument("--thinking-only", action="store_true", help="Skip tool calling tests")
    parser.add_argument(
        "--skip-injection", action="store_true", help="Skip injected tool test (only test native)"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--api", action="store_true", help="Probe via API instead of loading locally"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:10242",
        help="API base URL (default: http://localhost:10242)",
    )

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Color.disable()

    # If API mode, delegate to API probe and exit
    if args.api:
        run_api_probe(args.model_id, args.base_url)
        return

    # Check model availability
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore[import-untyped]

        cache_info = scan_cache_dir()
        cached_repos = {repo.repo_id for repo in cache_info.repos}
        if args.model_id not in cached_repos:
            print(warn(f"Model '{args.model_id}' not found in local HF cache."))
            print(dim("It will be downloaded on first load. Use Ctrl+C to cancel."))
            print()
    except ImportError:
        pass

    # Phase 1: Model info
    family, adapter = phase_model_info(args.model_id)

    # Load model
    print(section("Loading Model"))
    model, tokenizer = load_model(args.model_id)
    report_native_tool_support(adapter, tokenizer)

    # Phase 2: Template inspection
    phase_template_inspection(tokenizer)

    # Phase 3: Thinking test
    thinking_output = phase_thinking_test(model, tokenizer, adapter, family, args.max_tokens)

    # Phase 3b: Native tool test (optional)
    native_tool_output = None
    if not args.thinking_only:
        native_tool_output = phase_native_tool_test(
            model, tokenizer, adapter, family, args.max_tokens
        )

    # Phase 4: Injected tool calling test (optional)
    tool_output = None
    if not args.thinking_only and not args.skip_injection:
        tool_output = phase_tool_test(model, tokenizer, adapter, family, args.max_tokens)

    # Phase 5: Summary
    phase_summary(family, thinking_output, tool_output, native_tool_output)

    # Save raw outputs
    if args.save_raw:
        print(section("Saving Raw Outputs"))
        save_raw_outputs(args.model_id, thinking_output, tool_output, native_tool_output)

    print()


if __name__ == "__main__":
    # Suppress MLX/HF info logs unless DEBUG is set
    if "DEBUG" not in os.environ:
        import logging

        logging.basicConfig(level=logging.WARNING)

    main()
