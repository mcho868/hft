#!/usr/bin/env python3
"""
Convert the perfsafe model variant to CoreML
"""

import os
import sys
import torch
import coremltools as ct
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path

def fix_quantization_config(model_path):
    """Remove problematic quantization configs from model"""
    config_path = os.path.join(model_path, "config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove quantization configs that cause issues
    if 'quantization' in config:
        del config['quantization']
        print("Removed 'quantization' from config")
    
    if 'quantization_config' in config:
        del config['quantization_config']
        print("Removed 'quantization_config' from config")
    
    # Save cleaned config to temp location
    temp_config_path = os.path.join(model_path, "config_clean.json")
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_config_path

def convert_perfsafe_model():
    """Convert the perfsafe model to CoreML"""
    
    model_path = "/Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_perfsafe"
    output_path = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp/"
    
    print(f"🔄 Converting perfsafe model from: {model_path}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully")
        
        # Fix the config file
        print("Fixing quantization config...")
        clean_config_path = fix_quantization_config(model_path)
        
        # Load config without quantization
        print("Loading cleaned config...")
        config = AutoConfig.from_pretrained(clean_config_path)
        
        # Load model with cleaned config
        print("Loading perfsafe model with cleaned config...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,  # Dequantize to float32
            device_map="cpu",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        print("✅ Perfsafe model loaded successfully")
        
        # Clean up temp file
        os.remove(clean_config_path)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create example input for tracing
        max_length = 128  # Optimized for mobile
        example_text = "Patient symptoms: abdominal pain"
        
        # Tokenize example
        example_tokens = tokenizer(
            example_text, 
            return_tensors="pt", 
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        print(f"Example input shape: {example_tokens['input_ids'].shape}")
        
        # Create simplified wrapper
        class SimplifiedSmolLMPerfsafe(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids):
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    return outputs.logits[:, -1:, :]  # Only last token logits
        
        wrapped_model = SimplifiedSmolLMPerfsafe(model)
        
        # Test the wrapped model
        print("Testing wrapped perfsafe model...")
        with torch.no_grad():
            test_output = wrapped_model(example_tokens['input_ids'])
            print(f"Test output shape: {test_output.shape}")
        
        # Trace the model
        print("Tracing perfsafe model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model, 
                example_tokens['input_ids']
            )
        
        print("Converting perfsafe to CoreML...")
        
        # Convert to CoreML
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
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        
        # Add metadata
        coreml_model.short_description = "SmolLM-135M Medical Triage (Perfsafe Variant)"
        coreml_model.author = "HFT Medical AI Demo"
        coreml_model.license = "Research Use"
        coreml_model.version = "1.0-perfsafe"
        
        # Save with different name
        output_file = os.path.join(output_path, "SmolLM_Medical_Triage_Perfsafe.mlpackage")
        coreml_model.save(output_file)
        
        print(f"✅ Perfsafe CoreML model saved to: {output_file}")
        
        # Calculate size
        model_size = get_model_size(output_file)
        print(f"📊 Perfsafe model size: {model_size:.1f} MB")
        
        # Save tokenizer for iOS
        save_perfsafe_config(tokenizer, output_path, config)
        
        print("\n🎉 Perfsafe conversion completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Perfsafe conversion failed: {e}")
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

def save_perfsafe_config(tokenizer, output_path, config):
    """Save perfsafe model config for iOS"""
    
    vocab = tokenizer.get_vocab()
    
    ios_config = {
        "model_info": {
            "name": "SmolLM-135M Medical Triage (Perfsafe)",
            "variant": "perfsafe",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "max_length": 128,
            "architecture": "llama",
            "description": "Performance-safe variant optimized for stable inference"
        },
        "tokenizer": {
            "vocab": vocab,
            "vocab_size": len(vocab),
            "special_tokens": {
                "bos_token": tokenizer.bos_token if tokenizer.bos_token else "<s>",
                "eos_token": tokenizer.eos_token if tokenizer.eos_token else "</s>",
                "pad_token": tokenizer.pad_token if tokenizer.pad_token else "</s>",
                "unk_token": tokenizer.unk_token if tokenizer.unk_token else "<unk>",
                "bos_token_id": getattr(tokenizer, 'bos_token_id', 1),
                "eos_token_id": getattr(tokenizer, 'eos_token_id', 2),
                "pad_token_id": getattr(tokenizer, 'pad_token_id', 2),
                "unk_token_id": getattr(tokenizer, 'unk_token_id', 0)
            }
        },
        "generation": {
            "max_new_tokens": 64,
            "temperature": 0.05,  # Lower temperature for more stable output
            "do_sample": False
        }
    }
    
    config_file = os.path.join(output_path, "model_config_perfsafe.json")
    with open(config_file, 'w') as f:
        json.dump(ios_config, f, indent=2)
    
    print(f"✅ Perfsafe iOS config saved to: {config_file}")

if __name__ == "__main__":
    print("🔄 Converting SmolLM-135M Perfsafe variant to CoreML...")
    
    # Check dependencies
    try:
        import coremltools
        import transformers
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        sys.exit(1)
    
    # Run conversion
    success = convert_perfsafe_model()
    
    if success:
        print("\n🎉 SUCCESS! Both model variants are now iOS-ready!")
        print("\nYou now have:")
        print("1. SmolLM_Medical_Triage.mlpackage (highcap variant)")
        print("2. SmolLM_Medical_Triage_Perfsafe.mlpackage (perfsafe variant)")
        print("\nNext: Add both to Xcode and create model selector in app!")
    else:
        print("\n💥 Perfsafe conversion failed.")
        sys.exit(1)