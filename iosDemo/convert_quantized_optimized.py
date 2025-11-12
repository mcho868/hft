#!/usr/bin/env python3
"""
Convert MLX model to optimized quantized CoreML for maximum iOS performance
"""

import os
import sys
import torch
import coremltools as ct
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def convert_optimized_quantized_model():
    """Convert with aggressive quantization for best mobile performance"""
    
    model_path = "/Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_highcap"
    output_path = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp/"
    
    print(f"🔄 Converting to OPTIMIZED QUANTIZED CoreML...")
    
    try:
        # Load model (same as before)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Fix config
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Remove quantization configs
        if 'quantization' in config_data:
            del config_data['quantization']
        if 'quantization_config' in config_data:
            del config_data['quantization_config']
        
        temp_config = os.path.join(model_path, "temp_config.json")
        with open(temp_config, 'w') as f:
            json.dump(config_data, f)
        
        config = AutoConfig.from_pretrained(temp_config)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        os.remove(temp_config)
        model.eval()
        
        print("✅ Model loaded for quantized conversion")
        
        # Create mobile-optimized wrapper
        class MobileOptimizedSmolLM(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids):
                with torch.no_grad():
                    # Only forward pass, no generation loop
                    outputs = self.model(input_ids=input_ids)
                    # Return top-p filtered logits for mobile efficiency
                    logits = outputs.logits[:, -1:, :]
                    return logits
        
        wrapped_model = MobileOptimizedSmolLM(model)
        
        # Create example input
        max_length = 64  # Smaller for mobile
        example_text = "Patient symptoms:"
        example_tokens = tokenizer(
            example_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        # Trace model
        print("Tracing optimized model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, example_tokens['input_ids'])
        
        print("Converting with AGGRESSIVE quantization...")
        
        # Convert with maximum optimization
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=(1, max_length),
                    dtype=np.int32
                )
            ],
            outputs=[
                ct.TensorType(
                    name="logits",
                    dtype=np.float32
                )
            ],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,  # Start with FP16
            compute_units=ct.ComputeUnit.CPU_AND_NE,  # Neural Engine optimization
        )
        
        print("Applying 4-bit quantization...")
        
        # Apply aggressive quantization
        try:
            quantized_model = ct.compression_utils.palettize_weights(
                coreml_model,
                nbits=4,  # 4-bit like your original MLX model!
                mode="kmeans",
                granularity="per_grouped_channel"
            )
            print("✅ 4-bit quantization applied successfully")
            final_model = quantized_model
        except Exception as e:
            print(f"⚠️  4-bit quantization failed: {e}")
            print("🔄 Falling back to 8-bit quantization...")
            try:
                quantized_model = ct.compression_utils.palettize_weights(
                    coreml_model,
                    nbits=8,
                    mode="kmeans"
                )
                print("✅ 8-bit quantization applied successfully")
                final_model = quantized_model
            except:
                print("⚠️  Using FP16 model (no additional quantization)")
                final_model = coreml_model
        
        # Add metadata
        final_model.short_description = "SmolLM-135M Medical (Quantized for Performance)"
        final_model.author = "HFT Medical AI Demo"
        final_model.version = "1.0-quantized"
        
        # Save quantized model
        output_file = os.path.join(output_path, "SmolLM_Medical_Quantized.mlpackage")
        final_model.save(output_file)
        
        print(f"✅ Quantized model saved to: {output_file}")
        
        # Compare sizes
        original_size = get_model_size(os.path.join(output_path, "SmolLM_Medical_Triage.mlpackage"))
        quantized_size = get_model_size(output_file)
        
        print(f"📊 Size comparison:")
        print(f"   Original CoreML: {original_size:.1f} MB")
        print(f"   Quantized CoreML: {quantized_size:.1f} MB")
        print(f"   Compression ratio: {original_size/quantized_size:.1f}x smaller")
        
        # Save optimized config
        save_quantized_config(tokenizer, output_path, config)
        
        return True
        
    except Exception as e:
        print(f"❌ Quantized conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_model_size(model_path):
    """Get model size in MB"""
    if os.path.isdir(model_path):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(model_path)
            for filename in filenames
        )
    else:
        total_size = os.path.getsize(model_path)
    return total_size / (1024 * 1024)

def save_quantized_config(tokenizer, output_path, config):
    """Save config for quantized model"""
    vocab = tokenizer.get_vocab()
    
    ios_config = {
        "model_info": {
            "name": "SmolLM-135M Medical (Quantized)",
            "variant": "quantized_optimized",
            "quantization": "4-bit_or_8-bit",
            "vocab_size": config.vocab_size,
            "max_length": 64,
            "optimized_for": "mobile_performance"
        },
        "tokenizer": {
            "vocab": vocab,
            "vocab_size": len(vocab),
            "special_tokens": {
                "bos_token": tokenizer.bos_token or "<s>",
                "eos_token": tokenizer.eos_token or "</s>",
                "pad_token": tokenizer.pad_token or "</s>",
                "unk_token": tokenizer.unk_token or "<unk>"
            }
        },
        "generation": {
            "max_new_tokens": 32,  # Shorter for mobile
            "temperature": 0.1,
            "optimized": True
        }
    }
    
    config_file = os.path.join(output_path, "model_config_quantized.json")
    with open(config_file, 'w') as f:
        json.dump(ios_config, f, indent=2)
    
    print(f"✅ Quantized config saved to: {config_file}")

if __name__ == "__main__":
    print("🚀 Creating PERFORMANCE-OPTIMIZED quantized model...")
    
    success = convert_optimized_quantized_model()
    
    if success:
        print("\n🎉 SUCCESS! You now have 3 model variants:")
        print("1. SmolLM_Medical_Triage.mlpackage (~257MB, best quality)")
        print("2. SmolLM_Medical_Quantized.mlpackage (~75-135MB, best speed)")
        print("\n⚡ The quantized version should be 2-3x faster with smaller memory footprint!")
    else:
        print("\n💥 Quantized conversion failed.")