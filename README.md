# LLM Inference Server

A high-performance, production-ready LLM inference server built in Rust using Candle. Features continuous batching, paged attention KV cache, and OpenAI-compatible REST API.

## Features

- **Async inference server** using Tokio runtime
- **GGUF model loading** from disk at startup
- **Continuous batching** for efficient request processing
- **Paged attention KV cache** for memory-efficient caching
- **OpenAI-compatible API** (`/v1/chat/completions` and `/v1/completions`)
- **Prometheus metrics** (`/metrics`) with tokens/second, latency, queue depth, GPU utilization
- **Metal GPU backend** enabled by default for Apple Silicon
- **Optimized performance** targeting 2-3× faster than candle-examples/quantized

## Requirements

- Rust 1.70+ (edition 2021)
- macOS with Apple Silicon (M1/M2/M3) for Metal GPU support
- GGUF model file (e.g., Llama, Mistral, etc.)
- Tokenizer file (`tokenizer.json`) in the same directory as the model (optional, will fallback if not found)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-systems-2025

# Build in release mode for optimal performance
cargo build --release
```

## Usage

### Basic Usage

```bash
./target/release/llm-inference-server \
    --model /path/to/model.gguf \
    --port 8000 \
    --max-batch-size 32 \
    --max-seq-len 4096
```

### CLI Options

- `--model <path>`: Path to GGUF model file (required)
- `--port <number>`: Server port (default: 8000)
- `--max-batch-size <number>`: Maximum batch size (default: 32)
- `--max-seq-len <number>`: Maximum sequence length (default: 4096)
- `--verbose`: Enable verbose logging

### Example

```bash
./target/release/llm-inference-server \
    --model ~/models/llama-7b-q4_0.gguf \
    --port 8000 \
    --max-batch-size 32 \
    --max-seq-len 4096
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy"
}
```

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

The metrics endpoint exposes Prometheus-formatted metrics including:

- `tokens_per_second`: Tokens generated per second
- `request_latency_seconds`: Request latency (p50, p99 available via histogram)
- `queue_depth`: Number of requests waiting in queue
- `gpu_utilization_percent`: GPU utilization percentage
- `requests_total`: Total number of requests by endpoint and status
- `tokens_generated_total`: Total tokens generated

## Benchmarking

### Performance Testing

To benchmark the server performance, you can use tools like `ab` (Apache Bench) or `wrk`:

```bash
# Install wrk (if not already installed)
brew install wrk

# Run benchmark
wrk -t4 -c100 -d30s -s benchmark.lua http://localhost:8000/v1/chat/completions
```

Create a `benchmark.lua` file:

```lua
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Comparing with candle-examples

To compare performance with `candle-examples/quantized`:

1. Run the baseline from candle-examples:
   ```bash
   cd candle-examples
   cargo run --example quantized --release -- --model /path/to/model.gguf
   ```

2. Run this server:
   ```bash
   ./target/release/llm-inference-server --model /path/to/model.gguf
   ```

3. Use the same prompt and measure tokens/second

### Performance Metrics

Record the following metrics:

- **Tokens per second**: Measure throughput
- **Latency (p50, p99)**: Measure response time percentiles
- **GPU utilization**: Monitor GPU usage
- **Queue depth**: Monitor request queuing

## Performance Table

| Metric | Value | Notes |
|--------|-------|-------|
| Tokens/second | _TBD_ | Measured on M-series Mac |
| Latency (p50) | _TBD_ | 50th percentile |
| Latency (p99) | _TBD_ | 99th percentile |
| GPU Utilization | _TBD_ | Average during load |
| Batch Size | 32 | Configurable via CLI |

_Note: Fill in actual benchmark results after testing on your hardware._

## Architecture

### Components

- **`main.rs`**: Entry point, CLI parsing, server initialization
- **`server.rs`**: Axum web server with OpenAI-compatible endpoints
- **`engine.rs`**: Inference engine with continuous batching and KV cache
- **`metrics.rs`**: Prometheus metrics collection and export

### Key Optimizations

1. **Continuous Batching**: Requests are batched together for efficient GPU utilization
2. **Paged KV Cache**: Memory-efficient caching of attention key-value pairs
3. **Async Processing**: Non-blocking request handling with Tokio
4. **Metal GPU**: Native Apple Silicon acceleration
5. **Optimized Build**: Release mode with LTO and codegen optimizations

## Development

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Code Structure

```
src/
├── main.rs      # CLI and initialization
├── server.rs    # HTTP server and API endpoints
├── engine.rs    # Inference engine and batching
└── metrics.rs   # Prometheus metrics
```

## Troubleshooting

### Metal Device Not Available

If you see "Metal device not available", the server will fall back to CPU. Ensure you're running on Apple Silicon (M1/M2/M3) with Metal support.

### Model Loading Errors

- Ensure the GGUF file path is correct
- Check that the model format is supported (Llama architecture is assumed)
- Verify file permissions

### Tokenizer Not Found

The server will attempt to load `tokenizer.json` from the same directory as the model. If not found, it will use a fallback tokenizer. For best results, include the tokenizer file.

### Out of Memory

- Reduce `--max-batch-size`
- Reduce `--max-seq-len`
- Use a smaller quantized model (e.g., Q4_0 instead of F32)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
