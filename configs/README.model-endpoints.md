# Unified Qwen3.5 endpoint pool

`model_endpoints.qwen35.example.json` is shared by the EntiTables, WDC, and
WDC200K builders through `--model_endpoint_config`. The top-level
`served_model_name` replaces both modality-specific model names. Each physical
URL appears once and declares three independent limits:

- `text`: maximum text calls to this endpoint;
- `image`: maximum visual calls to this endpoint;
- `total`: shared ceiling across both modalities.

The local text endpoint is expected to start with `--language-model-only`; the
local image endpoint and every remote A100 endpoint must omit that flag. To add
a second A100, copy the `remote-a100-0` object, give it a unique ID and tunnel
URL, and increase the builder's remote text/image worker counts to the sum of
the two endpoint capacities.

The `local-4090-image-donated` route is the dynamic runner's port `8002`. It is
withdrawn until the text round finishes and that GPU is relaunched for visual
work. Remove this entry when invoking a builder directly without the dynamic
runner.

All dataset-model requests unconditionally send:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

The legacy `--enable_thinking` option is rejected.
