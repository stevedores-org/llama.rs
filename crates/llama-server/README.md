# llama-server

OpenAI-compatible HTTP API server for the llama.rs inference engine.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's chat completion endpoint
- **Streaming Support**: Server-Sent Events (SSE) for real-time token generation (coming soon)
- **Embeddings Support**: Generate embeddings for text inputs
- **Modular Architecture**: Built with Axum for high-performance async HTTP handling

## Building

```bash
cd crates/llama-server
cargo build --release
```

## Running

```bash
# With default settings (localhost:8080)
cargo run --bin llama-server

# Or using the release binary
./target/release/llama-server
```

The server will start on `http://127.0.0.1:8080`.

## API Endpoints

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok"
}
```

### Chat Completions

Non-streaming chat completion:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b-mock",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "stream": false,
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "id": "chatcmpl-550e8400e29b41d4a716446655440000",
  "object": "chat.completion",
  "created": 1708123456,
  "model": "llama-3-8b-mock",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

#### Request Parameters

- `model` (required): Model name to use
- `messages` (required): Array of message objects with `role` and `content`
- `stream` (optional): Enable streaming responses (default: false)
- `max_tokens` (optional): Maximum tokens to generate (default: 256)
- `temperature` (optional): Sampling temperature (default: 0.7)
- `top_p` (optional): Nucleus sampling parameter

#### Response Fields

- `id`: Unique identifier for the completion (format: `chatcmpl-{uuid}`)
- `object`: Type of object ("chat.completion")
- `created`: Unix timestamp when the completion was created
- `model`: Model used for the completion
- `choices`: Array of completion choices
  - `message`: Generated message with `role` and `content`
  - `finish_reason`: Reason completion stopped ("stop", "length", etc.)
- `usage`: Token usage statistics

### Embeddings

Generate embeddings for text:

```bash
# Single input
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b-mock",
    "input": "The quick brown fox"
  }'

# Batch input
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b-mock",
    "input": ["Hello", "world"]
  }'
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "llama-3-8b-mock",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 0,
    "total_tokens": 4
  }
}
```

## Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8080/v1"
)

# Chat completion
response = client.chat.completions.create(
    model="llama-3-8b-mock",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)
print(response.choices[0].message.content)

# Embeddings
embeddings = client.embeddings.create(
    model="llama-3-8b-mock",
    input=["Hello", "world"]
)
print(embeddings.data[0].embedding)
```

## Testing

```bash
# Run tests
cargo test -p llama-server

# Start server and manually test
cargo run --bin llama-server &
curl -X POST http://localhost:8080/v1/chat/completions ...
```

## Architecture

The server is built with:

- **Axum**: High-performance async web framework
- **Tokio**: Async runtime
- **Serde**: JSON serialization/deserialization
- **Tower**: Middleware and service composition

### Module Structure

```
src/
├── lib.rs           # Public API exports
├── main.rs          # CLI entry point
├── server.rs        # Router and server setup
├── handlers/        # HTTP request handlers
│   ├── chat.rs      # Chat completion endpoint
│   ├── embeddings.rs # Embeddings endpoint
│   └── health.rs    # Health check
├── models/          # OpenAI-compatible types
│   ├── chat.rs      # Chat request/response
│   ├── embeddings.rs # Embedding request/response
│   ├── streaming.rs # SSE types
│   └── common.rs    # Shared types
├── state.rs         # Application state
├── error.rs         # Error handling
└── streaming.rs     # SSE utilities (TODO)
```

## Future Work

- [ ] Implement SSE streaming for real-time token generation
- [ ] Add function calling support
- [ ] Support multiple models via registry
- [ ] Add authentication/API key validation
- [ ] Implement rate limiting
- [ ] Multi-turn conversation sessions
- [ ] Llama 3 / Qwen chat templates
- [ ] Comprehensive error codes matching OpenAI spec

## License

MIT
