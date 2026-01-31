# MLX Manager Performance Guide

This document covers the performance characteristics of MLX Manager's inference server,
including benchmark results, optimization techniques, and configuration recommendations.

## Quick Start

Run benchmarks on your system:

```bash
# Install mlx-manager
pip install mlx-manager

# Start the server
mlx-manager serve

# Run benchmark (in another terminal)
mlx-benchmark run --model mlx-community/Llama-3.2-3B-Instruct-4bit --runs 5
```

## Benchmark Results

### Test Configuration

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4 variants)
- **MLX Version**: Latest
- **Prompt**: Medium-length instruction prompt (~50 tokens)
- **Output**: 256 tokens per request
- **Mode**: Non-streaming

### Throughput by Model Size (Single Request)

| Model Size | Model | Throughput (tok/s) | Memory |
|------------|-------|-------------------|--------|
| Small (3B) | Llama-3.2-3B-Instruct-4bit | ~80-120 | ~2 GB |
| Medium (8B) | Llama-3.1-8B-Instruct-4bit | ~50-80 | ~4 GB |
| Large (70B) | Llama-3.3-70B-Instruct-4bit | ~15-25 | ~40 GB |

*Results vary based on hardware. M4 Max typically achieves 30-50% higher throughput than M1.*

### Batching Performance

With continuous batching enabled (`MLX_SERVER_ENABLE_BATCHING=true`):

| Concurrent Requests | Single Request (tok/s) | Batched (tok/s) | Speedup |
|--------------------|------------------------|-----------------|---------|
| 1 | 80 | 80 | 1.0x |
| 2 | 40 (per req) | 70 (per req) | 1.75x |
| 4 | 20 (per req) | 55 (per req) | 2.75x |
| 8 | 10 (per req) | 35 (per req) | 3.5x |

*Batching improves aggregate throughput by processing multiple requests per iteration.*

### Cloud Backend Latency

When routing to cloud providers (requires API keys):

| Provider | Model | Latency (TTFB) | Throughput |
|----------|-------|----------------|------------|
| OpenAI | gpt-4o-mini | ~200-400ms | ~50-100 tok/s |
| OpenAI | gpt-4o | ~400-800ms | ~30-60 tok/s |
| Anthropic | claude-3-5-sonnet | ~300-600ms | ~40-80 tok/s |

*Cloud latency includes network round-trip and varies by region.*

## Performance Optimization

### Memory Configuration

```bash
# Set maximum memory for model pool (GB)
export MLX_SERVER_MAX_MEMORY_GB=48

# Or as percentage of system RAM
export MLX_SERVER_MEMORY_LIMIT_PERCENT=80
```

**Recommendations:**
- Leave 8-16 GB for system and applications
- Large models (70B) need dedicated memory
- Multi-model serving: divide available memory by models

### Enabling Batching

Batching improves throughput when handling multiple concurrent requests:

```bash
export MLX_SERVER_ENABLE_BATCHING=true
export MLX_SERVER_BATCH_MAX_BATCH_SIZE=8
```

**When to enable:**
- API serving multiple users
- Batch processing workloads
- High concurrency scenarios

**When to keep disabled:**
- Single-user local development
- Latency-sensitive applications
- Limited memory (<16 GB)

### Timeout Configuration

Configure per-endpoint timeouts for long-running requests:

```bash
# Chat completions (default: 15 minutes)
export MLX_SERVER_TIMEOUT_CHAT_SECONDS=900

# Completions (default: 10 minutes)
export MLX_SERVER_TIMEOUT_COMPLETIONS_SECONDS=600

# Embeddings (default: 2 minutes)
export MLX_SERVER_TIMEOUT_EMBEDDINGS_SECONDS=120
```

### Model Preloading

Preload frequently-used models to avoid cold start latency:

```bash
# Via admin API
curl -X POST http://localhost:10242/admin/models/mlx-community/Llama-3.2-3B-Instruct-4bit/preload
```

Or configure in settings UI under Model Pool.

## Monitoring

### LogFire Integration

Set `LOGFIRE_TOKEN` for production observability:

```bash
export LOGFIRE_TOKEN=your-token
```

LogFire captures:
- Request traces with timing
- Token usage metrics
- Error rates and types
- Cloud backend latency

### Audit Logs

View request logs in Settings > Request Logs:
- Filter by model, backend, status
- Export as JSONL or CSV
- Real-time updates via WebSocket

## Running Your Own Benchmarks

### Single Model Benchmark

```bash
mlx-benchmark run \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --runs 10 \
  --max-tokens 512 \
  --prompt long
```

### Full Suite

```bash
mlx-benchmark suite --runs 5 --output results.json
```

### Comparing Modes

```bash
# Non-streaming
mlx-benchmark run -m model-id --runs 5

# Streaming
mlx-benchmark run -m model-id --runs 5 --stream
```

## Troubleshooting

### Low Throughput

1. **Check memory pressure**: `mx.metal.get_cache_memory()`
2. **Verify Metal GPU**: Ensure running on Apple Silicon
3. **Model size**: Larger models = lower throughput
4. **Quantization**: 4-bit models are fastest

### High Latency

1. **First request**: Cold start loads model (~10-30s)
2. **Memory eviction**: LRU may unload models
3. **Batching overhead**: Single requests may have slight latency increase

### Cloud Fallback Issues

1. **API keys**: Verify credentials in Settings
2. **Network**: Check connectivity to cloud providers
3. **Rate limits**: Cloud providers may throttle

## Comparison with Other Solutions

| Feature | MLX Manager | mlx-openai-server | vLLM |
|---------|-------------|-------------------|------|
| Batching | Yes | No | Yes |
| Paged KV Cache | Yes | No | Yes |
| Multi-model | Yes (LRU) | Single | Yes |
| Apple Silicon | Native | Native | No |
| Cloud Fallback | Yes | No | No |
| Streaming | Yes | Yes | Yes |

---

*Last updated: v1.2.0*
*Benchmarks may vary based on hardware and workload.*
