  # Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-256M-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
            },
            {"type": "text", "text": "What animal is on the candy? answer with only the animal name"},
        ],
    }
]

print(pipe(text=messages))  