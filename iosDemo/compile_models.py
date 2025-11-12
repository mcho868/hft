#!/usr/bin/env python3
"""
Script to compile CoreML .mlpackage files to .mlmodelc format for iOS deployment
"""

import os
import sys
import coremltools as ct

def compile_model(mlpackage_path, output_dir):
    """Compile a .mlpackage to .mlmodelc format"""
    if not os.path.exists(mlpackage_path):
        print(f"❌ Model not found: {mlpackage_path}")
        return False
    
    try:
        print(f"🔄 Loading model: {mlpackage_path}")
        
        # Get model name from path
        model_name = os.path.basename(mlpackage_path).replace('.mlpackage', '')
        output_path = os.path.join(output_dir, f"{model_name}.mlmodelc")
        
        print(f"🔄 Compiling to: {output_path}")
        
        # Use the correct CoreML compilation API
        compiled_model_url = ct.models.utils.compile_model(mlpackage_path)
        
        # Move the compiled model to our desired location
        import shutil
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.move(compiled_model_url, output_path)
        
        print(f"✅ Successfully compiled: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to compile {mlpackage_path}: {e}")
        return False

def main():
    base_dir = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp"
    
    models_to_compile = [
        f"{base_dir}/SmolLM_Medical_highcap_4bit.mlpackage",
        f"{base_dir}/SmolLM_Medical_persafe_4bit.mlpackage"
    ]
    
    print("🚀 Starting CoreML model compilation...")
    
    success_count = 0
    for model_path in models_to_compile:
        if compile_model(model_path, base_dir):
            success_count += 1
    
    print(f"\n📊 Compilation complete: {success_count}/{len(models_to_compile)} models compiled successfully")
    
    if success_count == len(models_to_compile):
        print("✅ All models compiled successfully!")
        return 0
    else:
        print("❌ Some models failed to compile")
        return 1

if __name__ == "__main__":
    sys.exit(main())