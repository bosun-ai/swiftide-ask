# Swiftide Ask

Use swiftide-ask to ask questions about your codebase. It uses QDrant and OpenAI.

## Installation

```bash
cargo install swiftide-ask
```

## Usage

First, you need to start QDrant and Redis. You can use the following commands to start them:

```bash
docker run --rm -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.11.0
docker run --rm -d -p 6379:6379 redis
# Only if you want to debug performance:
docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest
```

Then make sure swiftide-ask can connect to QDrant and Redis as well as OpenAI. You can use the following command to set the environment variables:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
export REDIS_URL=redis://localhost:6379
export QDRANT_URL=http://localhost:6334
```

Then, you can use the following command to ask questions about your codebase:

```bash
swiftide-ask "How do I create a new view controller?"
```

If you've enabled Jaeger, then you can find the traces in the browser at [http://localhost:16686](http://localhost:16686)

## License

See [LICENSE](LICENSE.md)

## Sponsor

Created by [Bosun.Ai](https://bosun.ai)

[![Bosun.Ai](https://bosun.ai/assets/images/small_logo.png)](https://bosun.ai)
