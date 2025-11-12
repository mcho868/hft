import torch
import time
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Test with the existing image from original script
image_path = "/Users/choemanseung/789/hft/PAD-UFES-20/imgs_part_1/imgs_part_1/PAT_100_393_595.png"

try:
    print("Loading image...")
    image = load_image(image_path)
    print("Image loaded successfully")
    
    print("Loading model components...")
    start_time = time.time()
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    print(f"Processor loaded in {time.time() - start_time:.2f}s")
    
    model_start = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    print(f"Model loaded in {time.time() - model_start:.2f}s")
    
    # Test inference
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Classify this skin lesion as one of: NEV, BCC, ACK, SEK, or SCC. Answer with just the 3-letter code."}
            ]
        },
    ]
    
    print("Running inference...")
    inference_start = time.time()
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)
    
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    inference_time = time.time() - inference_start
    print(f"Inference completed in {inference_time:.2f}s")
    print(f"Response: {generated_texts[0]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()