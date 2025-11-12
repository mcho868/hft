#!/usr/bin/env python3
"""
Convert SmolLM-135M MLX model to CoreML for iOS deployment
"""

import os
import sys
import torch
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def convert_model_to_coreml():
    """Convert the SmolLM model to CoreML format"""
    
    model_path = "/Users/choemanseung/789/hft/iosDemo/models/SmolLM2-135M_4bit_highcap"
    output_path = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp/"
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with special handling for quantized MLX models
        print("Attempting to load quantized model...")
        try:
            # Try loading with trust_remote_code and ignoring quantization config
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                load_in_8bit=False,
                load_in_4bit=False
            )
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            print("Trying alternative loading method...")
            
            # Try loading without quantization configs
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # Remove quantization config if present
            if hasattr(config, 'quantization_config'):
                delattr(config, 'quantization_config')
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
        
        print("Model loaded successfully")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create example input for tracing
        max_length = 256  # Reduced for mobile constraints
        example_text = "Patient query: 25-year-old Female. Symptoms: chest pain, difficulty breathing"
        
        # Tokenize example
        example_tokens = tokenizer(
            example_text, 
            return_tensors="pt", 
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        print(f"Example input shape: {example_tokens['input_ids'].shape}")
        
        # Create a wrapper for generation that's more CoreML friendly
        class SmolLMWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids):
                # Simple forward pass for next token prediction
                outputs = self.model(input_ids=input_ids)
                return outputs.logits
        
        wrapped_model = SmolLMWrapper(model)
        
        # Trace the model
        print("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model, 
                example_tokens['input_ids']
            )
        
        print("Converting to CoreML...")
        
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
            compute_precision=ct.precision.FLOAT16,  # Use FP16 for mobile efficiency
            compute_units=ct.ComputeUnit.ALL  # Use Neural Engine when available
        )
        
        # Add model metadata
        coreml_model.short_description = "SmolLM-135M Medical Triage Model"
        coreml_model.author = "HFT Medical AI Demo"
        coreml_model.license = "Research Use"
        coreml_model.version = "1.0"
        
        # Save the model
        output_file = os.path.join(output_path, "SmolLM_Medical_Triage.mlpackage")
        coreml_model.save(output_file)
        
        print(f"✅ CoreML model saved to: {output_file}")
        print(f"📊 Model size: {get_model_size(output_file):.1f} MB")
        
        # Save tokenizer vocab for iOS
        save_tokenizer_for_ios(tokenizer, output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
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

def save_tokenizer_for_ios(tokenizer, output_path):
    """Save tokenizer vocabulary for iOS Swift usage"""
    
    # Create a simplified vocab mapping
    vocab = tokenizer.get_vocab()
    
    # Save as JSON for iOS
    import json
    vocab_file = os.path.join(output_path, "tokenizer_vocab.json")
    
    with open(vocab_file, 'w') as f:
        json.dump({
            "vocab": vocab,
            "special_tokens": {
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": tokenizer.bos_token,
                "unk_token": tokenizer.unk_token
            },
            "vocab_size": len(vocab)
        }, f, indent=2)
    
    print(f"✅ Tokenizer vocab saved to: {vocab_file}")

if __name__ == "__main__":
    print("🔄 Converting SmolLM-135M to CoreML for iOS...")
    
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
        print("\n🎉 Conversion completed successfully!")
        print("Next steps:")
        print("1. Add SmolLM_Medical_Triage.mlpackage to your Xcode project")
        print("2. Update MLXGenerator.swift to use CoreML")
        print("3. Build and run on iOS device")
    else:
        print("\n💥 Conversion failed. Check error messages above.")
        sys.exit(1)