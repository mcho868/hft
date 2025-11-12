#!/usr/bin/env python3
"""
Quick quantization of the existing CoreML model
"""

import coremltools as ct
import os

def quantize_existing_model():
    """Quantize the already converted CoreML model"""
    
    input_path = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp/SmolLM_Medical_Triage_Perfsafe.mlpackage"
    output_path = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp/SmolLM_Medical_persafe_4bit.mlpackage"
    
    print(f"🔄 Loading existing CoreML model...")
    
    try:
        # Load the existing model
        model = ct.models.MLModel(input_path)
        print("✅ Model loaded successfully")
        
        # Get original size
        original_size = get_model_size(input_path)
        print(f"📊 Original model size: {original_size:.1f} MB")
        
        # Apply 4-bit quantization (more reliable than 4-bit)
        print("🔄 Applying 4-bit quantization...")
        
        quantized_model = ct.compression_utils.palettize_weights(
            model,
            nbits=4,  # 4-bit quantization
            mode="kmeans"
        )
        
        print("✅ 4-bit quantization completed!")
        
        # Update metadata
        quantized_model.short_description = "SmolLM-135M Medical (4-bit Quantized)"
        quantized_model.version = "1.0-quantized-4bit"
        
        # Save quantized model
        quantized_model.save(output_path)
        
        # Compare sizes
        quantized_size = get_model_size(output_path)
        compression_ratio = original_size / quantized_size
        
        print(f"✅ Quantized model saved to: {output_path}")
        print(f"📊 Results:")
        print(f"   Original:  {original_size:.1f} MB")
        print(f"   Quantized: {quantized_size:.1f} MB")
        print(f"   Compression: {compression_ratio:.1f}x smaller")
        print(f"   Expected speedup: {compression_ratio:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False

def get_model_size(model_path):
    """Get model size in MB"""
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(model_path)
        for filename in filenames
    )
    return total_size / (1024 * 1024)

if __name__ == "__main__":
    print("⚡ Quick quantization of existing CoreML model...")
    
    success = quantize_existing_model()
    
    if success:
        print("\n🎉 SUCCESS! You now have a quantized model!")
        print("Expected performance improvement:")
        print("• 2-3x faster inference")
        print("• ~50% smaller size")
        print("• Lower memory usage")
        print("\nAdd to Xcode and test the speed difference!")
    else:
        print("\n💥 Quick quantization failed.")