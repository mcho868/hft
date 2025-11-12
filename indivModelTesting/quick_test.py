import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

print("Loading model...")
try:
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    print("Processor loaded successfully")
    
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    print("Model loaded successfully")
    
except Exception as e:
    print(f"Error loading model: {e}")