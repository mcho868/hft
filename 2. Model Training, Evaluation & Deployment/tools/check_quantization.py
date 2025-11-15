#!/usr/bin/env python3
"""
Script to check quantization levels of HuggingFace and MLX models
"""

import json
import os
from transformers import AutoConfig
from mlx_lm import load
import mlx.core as mx
from mlx.utils import tree_flatten

def check_hf_model_quantization(model_name):
    """Check HuggingFace model quantization"""
    print(f"\n=== Checking HuggingFace Model: {model_name} ===")
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        print(f"Model Type: {config.model_type}")
        print(f"Torch Dtype: {getattr(config, 'torch_dtype', 'Not specified')}")
        
        # Check for quantization config
        if hasattr(config, 'quantization_config') and config.quantization_config:
            print(f"Quantization Config: {config.quantization_config}")
        else:
            print("Quantization Config: None (Full precision)")
            
        # Check for other quantization indicators
        quant_attrs = ['bits', 'group_size', 'quantization']
        for attr in quant_attrs:
            if hasattr(config, attr):
                print(f"{attr}: {getattr(config, attr)}")
                
    except Exception as e:
        print(f"Error checking HF model: {e}")

def check_mlx_model_quantization(model_path):
    """Check MLX model quantization"""
    print(f"\n=== Checking MLX Model: {model_path} ===")
    try:
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Model Type: {config.get('model_type', 'Unknown')}")
            print(f"Torch Dtype: {config.get('torch_dtype', 'Not specified')}")
            
            # Check for quantization config
            if 'quantization' in config:
                print(f"Quantization Config: {config['quantization']}")
            else:
                print("Quantization Config: None (Full precision)")
        
        # Load the actual model to check weights
        print("Loading MLX model to inspect weights...")
        model, tokenizer = load(model_path)
        
        # Check a sample weight's dtype using tree_flatten
        flat_params = tree_flatten(model.parameters())
        sample_weights = []
        
        for i, (name, param) in enumerate(flat_params):
            if 'weight' in name:
                sample_weights.append((name, param.dtype, param.shape))
                if len(sample_weights) >= 3:  # Just check first few
                    break
        
        print("\nSample Weight Information:")
        for name, dtype, shape in sample_weights:
            print(f"  {name}: dtype={dtype}, shape={shape}")
            
        # Calculate model size
        total_params = sum(param.size for _, param in flat_params)
        
        # Check if any weights are quantized by looking for scales/biases
        quantized_layers = []
        for name, param in flat_params:
            if 'scales' in name or 'biases' in name:
                quantized_layers.append(name)
        
        if quantized_layers:
            print(f"\nQuantized layers detected: {len(quantized_layers)} layers")
            print("Sample quantized layers:")
            for layer in quantized_layers[:3]:
                print(f"  {layer}")
            precision = "Quantized (likely 4-bit)"
        else:
            print("\nNo quantized layers detected")
            
        # Estimate size based on dtype and quantization
        if quantized_layers:
            # If quantized, estimate 4-bit (0.5 bytes per param)
            size_gb = total_params * 0.5 / (1024**3)
            precision = "Quantized (4-bit)"
        elif any('float16' in str(dtype) for _, dtype, _ in sample_weights):
            size_gb = total_params * 2 / (1024**3)  # 2 bytes per float16
            precision = "16-bit (float16)"
        elif any('bfloat16' in str(dtype) for _, dtype, _ in sample_weights):
            size_gb = total_params * 2 / (1024**3)  # 2 bytes per bfloat16 
            precision = "16-bit (bfloat16)"
        elif any('float32' in str(dtype) for _, dtype, _ in sample_weights):
            size_gb = total_params * 4 / (1024**3)  # 4 bytes per float32
            precision = "32-bit (float32)"
        else:
            size_gb = total_params * 2 / (1024**3)  # Assume 16-bit
            precision = "Unknown (assuming 16-bit)"
            
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Estimated Size: {size_gb:.2f} GB")
        print(f"  Precision: {precision}")
        
        # Check actual file size
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            actual_size_gb = os.path.getsize(safetensors_path) / (1024**3)
            print(f"  Actual File Size: {actual_size_gb:.2f} GB")
        
    except Exception as e:
        print(f"Error checking MLX model: {e}")

def main():
    print("MLX Quantization Analysis")
    print("=" * 50)
    
    print("\n## MLX Quantization Methods ##")
    print("MLX supports the following quantization methods:")
    print("- Bits: 2, 3, 4, 5, 6, 8 bits per weight")
    print("- Group Size: Default 64 (weights quantized in groups)")
    print("- Mixed Quantization: Different layers can use different bit widths")
    print("- Default: 4-bit quantization with group size 64")
    print("- Affine Quantization: w_quantized = round((w - bias) / scale)")
    
    # Check HuggingFace models
    hf_models = [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct"
    ]
    
    for model_name in hf_models:
        check_hf_model_quantization(model_name)
    
    # Check MLX models
    mlx_models = [
        "./mlx/SmolLM2-135M-Instruct-MLX",
        "./mlx/SmolLM2-360M-Instruct-MLX"
    ]
    
    for model_path in mlx_models:
        if os.path.exists(model_path):
            check_mlx_model_quantization(model_path)
        else:
            print(f"\n=== MLX Model Not Found: {model_path} ===")

if __name__ == "__main__":
    main() 