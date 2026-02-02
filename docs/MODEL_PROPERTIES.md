# Heuristics to determine model properties

## Model Family and Type

The best approach uses Python's `transformers` library to parse `config.json` for model family and type, checks file extensions/metadata for quantization, and heuristically scans model cards or configs for tool use indicators, then displays results as badges in your app's UI.

Parsing Model Files
Load `config.json` with `AutoConfig.from_pretrained(local_path)` or `json.load()` to extract `architectures[0]` (e.g., "LlamaForCausalLM" → Llama family) and model_type for precise type.

For quantization:

Safetensors/PyTorch: Infer from filename (e.g., "q4_k.gguf" → Q4_K) or quantization_config in config.json (if present, e.g., BitsAndBytes).

GGUF: Use llama-cpp-python to load model and read llm.metadata (keys like "general.architecture", "llama.context_length", quantization hints).

## Tool Use

Tool use: No standard flag; check config for keys like "tools" or scan model card YAML/text for keywords ("function-calling", "tool-use") via ModelCard.load().

Handle local dirs post-download with huggingface_hub's list_repo_files or os.walk to find key files.

## Badge Implementation

Use HTML/Markdown badges like Hugging Face's style (e.g., via shields.io API or SVG generation) for visual cues: "Llama" "CausalLM" "Tool Use".<grok:render ...>

| Property |Detection Logic |
|----------|---------------|
| Family | config["architectures"].split("For") | 
| Type | config["model_type"] |
| Quant | Filename/metadata parse |
| Tool Use | Heuristic card/config scan |

Display badges in a model card UI post-download, caching parses for speed in your local workflows
