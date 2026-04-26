from vllm.entrypoints.openai.api_server import serve

serve(
    model="./merged-model",   # your saved model path
    host="0.0.0.0",
    port=8000,
    api_key="token-abc123"
)
