#!/usr/bin/env python3
"""
Fixed conversion script for MLX quantized models to CoreML
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

def convert_model_to_coreml():
    """Convert the SmolLM model to CoreML format with quantization handling"""
    
    model_path = "/Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_highcap"
    output_path = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp/"
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load tokenizer (this should work fine)
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
        print("Loading model with cleaned config...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,  # Dequantize to float32
            device_map="cpu",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        print("✅ Model loaded successfully")
        
        # Clean up temp file
        os.remove(clean_config_path)
        
        # Set model to evaluation mode
        model.eval()
        
        # Create example input for tracing
        max_length = 128  # Smaller for better mobile performance
        example_text = "Patient symptoms: chest pain"
        
        # Tokenize example with proper padding
        example_tokens = tokenizer(
            example_text, 
            return_tensors="pt", 
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        print(f"Example input shape: {example_tokens['input_ids'].shape}")
        print(f"Vocab size: {config.vocab_size}")
        
        # Create a simplified wrapper for mobile
        class SimplifiedSmolLM(torch.nn.Module):
            def __init__(self, model, max_new_tokens=32):
                super().__init__()
                self.model = model
                self.max_new_tokens = max_new_tokens
                
            def forward(self, input_ids):
                # Just return logits for next token prediction
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    return outputs.logits[:, -1:, :]  # Only last token logits
        
        wrapped_model = SimplifiedSmolLM(model)
        
        # Test the wrapped model
        print("Testing wrapped model...")
        with torch.no_grad():
            test_output = wrapped_model(example_tokens['input_ids'])
            print(f"Test output shape: {test_output.shape}")
        
        # Trace the model
        print("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model, 
                example_tokens['input_ids']
            )
        
        print("Converting to CoreML...")
        
        # Convert to CoreML with optimizations
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
            compute_units=ct.ComputeUnit.CPU_AND_NE  # CPU + Neural Engine
        )
        
        # Add metadata
        coreml_model.short_description = "SmolLM-135M Medical Triage (Dequantized)"
        coreml_model.author = "HFT Medical AI Demo"
        coreml_model.license = "Research Use"
        coreml_model.version = "1.1"
        
        # Save the model
        output_file = os.path.join(output_path, "SmolLM_Medical_Triage.mlpackage")
        coreml_model.save(output_file)
        
        print(f"✅ CoreML model saved to: {output_file}")
        
        # Calculate size
        model_size = get_model_size(output_file)
        print(f"📊 Model size: {model_size:.1f} MB")
        
        # Save tokenizer for iOS
        save_tokenizer_for_ios(tokenizer, output_path, config)
        
        print("\n🎉 Conversion completed successfully!")
        print("Your quantized MLX model has been converted to CoreML!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
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

def save_tokenizer_for_ios(tokenizer, output_path, config):
    """Save tokenizer and model info for iOS"""
    
    # Get vocabulary
    vocab = tokenizer.get_vocab()
    
    # Create comprehensive config for iOS
    ios_config = {
        "model_info": {
            "name": "SmolLM-135M Medical Triage",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "max_length": 128,
            "architecture": "llama"
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
            "temperature": 0.1,
            "do_sample": False
        }
    }
    
    # Save configuration
    config_file = os.path.join(output_path, "model_config.json")
    with open(config_file, 'w') as f:
        json.dump(ios_config, f, indent=2)
    
    print(f"✅ iOS model config saved to: {config_file}")

if __name__ == "__main__":
    print("🔄 Converting quantized MLX SmolLM-135M to CoreML for iOS...")
    
    # Check dependencies
    try:
        import coremltools
        import transformers
        print("✅ Dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install coremltools transformers torch")
        sys.exit(1)
    
    # Run conversion
    success = convert_model_to_coreml()
    
    if success:
        print("\n🎉 SUCCESS! Your MLX model is now iOS-ready!")
        print("\nNext steps:")
        print("1. Add SmolLM_Medical_Triage.mlpackage to Xcode project")
        print("2. Add model_config.json to Xcode project") 
        print("3. Update TriageViewModel to use CoreMLGenerator")
        print("4. Build and run - you'll have REAL AI inference on iOS!")
    else:
        print("\n💥 Conversion failed. Check error messages above.")
        sys.exit(1)