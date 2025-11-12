# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

from mlx_lm import load, generate

model, tokenizer = load("/Users/choemanseung/789/hft/mlx_models/Qwen/Qwen3-4B-Instruct-2507")
response = generate(model, tokenizer, prompt="Who are you?", max_tokens=100)
print(response)