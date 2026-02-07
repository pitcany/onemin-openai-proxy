# 1minAI OpenAI-Compatible Proxy

A local HTTP service that provides an OpenAI-compatible API interface while internally translating requests to 1minAI's AI Feature API.

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI API clients
- **Multi-Feature Support**: Automatically detects and routes to appropriate 1minAI feature
  - **Text Chat** (CHAT_WITH_AI) - Standard text conversations
  - **Vision** (CHAT_WITH_IMAGE) - Multimodal chat with images
  - **PDF Analysis** (CHAT_WITH_PDF) - Chat with PDF documents
  - **YouTube** (CHAT_WITH_YOUTUBE_VIDEO) - Discuss YouTube videos
- **Streaming Support**: Real-time streaming responses with automatic fallback
- **Retry Logic**: Exponential backoff with jitter for rate limits
- **Error Translation**: Upstream errors mapped to OpenAI-style error responses
- **Token Estimation**: Approximate usage stats when not provided by upstream
- **Request Logging**: Optional redacted logging for debugging

## Quickstart

### Prerequisites

- Python 3.11+
- 1minAI API key (get one at [1min.ai](https://1min.ai))

### Local Development

```bash
# Clone and navigate to the project
cd onemin-openai-proxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env and set your ONE_MIN_AI_API_KEY

# Start the server
make dev
```

The server will start at `http://localhost:8080`.

### Docker

```bash
# Build and run with docker-compose
export ONE_MIN_AI_API_KEY=your-api-key-here
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

Or run directly with Docker:

```bash
docker build -t onemin-openai-proxy .

docker run -d \
  -p 8080:8080 \
  -e ONE_MIN_AI_API_KEY=your-api-key-here \
  --name onemin-proxy \
  onemin-openai-proxy
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ONE_MIN_AI_API_KEY` | **Yes** | - | Your 1minAI API key |
| `ONE_MIN_AI_BASE_URL` | No | `https://api.1min.ai` | 1minAI API base URL |
| `PROXY_HOST` | No | `0.0.0.0` | Host to bind |
| `PROXY_PORT` | No | `8080` | Port to bind |
| `DEFAULT_1MIN_MODEL` | No | `gpt-4o-mini` | Default model when not specified |
| `DEFAULT_WEBSEARCH` | No | `false` | Enable web search by default |
| `DEFAULT_NUM_OF_SITE` | No | `1` | Number of sites for web search |
| `DEFAULT_MAX_WORD` | No | `500` | Max words from web search |
| `REQUEST_TIMEOUT_SECS` | No | `60` | Request timeout |
| `RETRIES` | No | `2` | Number of retries for rate limits |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `ENABLE_REQUEST_LOGGING` | No | `false` | Enable request logging (redacted) |
| `EXTRA_MODELS` | No | - | Comma-separated extra models for /v1/models |

## API Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

With streaming:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{\"role\": \"user\", \"content\": \"Tell me a joke\"}],
    \"stream\": true
  }'
```

### Vision (Image Analysis)

The proxy automatically detects image URLs in multimodal content and uses CHAT_WITH_IMAGE:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }]
  }'
```

### PDF Analysis

The proxy detects PDF URLs and automatically creates conversations for document chat:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{
      "role": "user",
      "content": "Summarize this PDF: https://example.com/document.pdf"
    }]
  }'
```

### YouTube Video Chat

The proxy detects YouTube URLs and creates conversations for video discussions:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{
      "role": "user",
      "content": "What is this video about? https://www.youtube.com/watch?v=VIDEO_ID"
    }]
  }'
```

### GET /v1/models

List available models.

```bash
curl http://localhost:8080/v1/models
```

### GET /healthz

Health check endpoint.

```bash
curl http://localhost:8080/healthz
```

## OpenClaw Configuration

To use this proxy with OpenClaw or other OpenAI-compatible clients:

1. **Base URL**: `http://localhost:8080/v1`
2. **API Key**: Any string (the proxy ignores client API keys)
3. **Model**: Use any model name supported by 1minAI (e.g., `gpt-4o-mini`)

The proxy uses your `ONE_MIN_AI_API_KEY` environment variable for authentication with 1minAI.

## Model Names & Feature Support

The `model` field in requests is passed directly to 1minAI. Different features support different models:

### CHAT_WITH_AI (Text Chat)
- `gpt-4o-mini` (default), `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- `gpt-5`, `gpt-5-mini`, `o3`, `o3-mini`, `o3-pro`
- `deepseek-chat`, `deepseek-reasoner`
- `qwen-plus`, `qwen-max`
- `mistral-large-latest`, `mistral-small-latest`
- And others...

### CHAT_WITH_IMAGE (Vision)
- `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- `o3`, `o3-mini`, `o3-pro`
- Note: Claude and Gemini models are NOT supported for CHAT_WITH_IMAGE

### CHAT_WITH_PDF & CHAT_WITH_YOUTUBE_VIDEO
- `gpt-4o-mini`, `gpt-4o` (recommended)
- Check 1minAI documentation for the full list

**Feature Detection**: The proxy automatically detects which feature to use based on message content:
- Contains image URLs → CHAT_WITH_IMAGE
- Contains PDF URLs → CHAT_WITH_PDF
- Contains YouTube URLs → CHAT_WITH_YOUTUBE_VIDEO
- Text only → CHAT_WITH_AI

## Running Tests

```bash
make test
```

## Troubleshooting

### 401 Unauthorized

- Verify your `ONE_MIN_AI_API_KEY` is correct
- Check that the API key is active in your 1minAI account

### 429 Rate Limited

- The proxy automatically retries with exponential backoff
- If persists, check your 1minAI rate limits/quotas
- Increase `RETRIES` if needed

### Timeouts

- Increase `REQUEST_TIMEOUT_SECS` for longer-running requests
- Check your network connectivity to 1minAI

### Streaming Issues

If streaming doesn't work as expected:

- The proxy will automatically fall back to non-streaming
- Check response header `X-Proxy-Streaming: passthrough-unsupported` if fallback was used
- Enable `ENABLE_REQUEST_LOGGING=true` for debugging

### Token Usage Shows "estimated"

- 1minAI doesn't always return token counts for chat
- The proxy estimates tokens (~1.3 per word)
- Check response header `X-Proxy-Usage: estimated`

## Architecture

```
┌─────────────────┐     ┌────────────────────┐     ┌──────────────┐
│  OpenAI Client  │────▶│  1minAI Proxy      │────▶│   1minAI     │
│  (OpenClaw etc) │     │  (localhost:8080)  │     │   API        │
└─────────────────┘     └────────────────────┘     └──────────────┘
                              │
                              ├── POST /v1/chat/completions
                              ├── GET  /v1/models
                              └── GET  /healthz
```

## Project Structure

```
onemin-openai-proxy/
├── app/
│   ├── main.py           # FastAPI entry, routing
│   ├── settings.py       # Environment configuration
│   ├── openai_schemas.py # OpenAI-compatible schemas
│   ├── onemin_schemas.py # 1minAI request schemas
│   ├── translate.py      # OpenAI ↔ 1minAI translation
│   ├── onemin_client.py  # HTTP client with retry
│   └── errors.py         # Error handling
├── tests/
│   ├── test_healthz.py
│   ├── test_chat_completions_nonstream.py
│   └── test_chat_completions_stream.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

## License

MIT
