#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files from iOS TriageApp
Removes .mlpackage files, MLX models, and unused Swift files
"""

import os
import shutil
import subprocess
from pathlib import Path

def get_file_size(path):
    """Get size of file or directory in MB"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total / (1024 * 1024)
    return 0

def remove_file_or_dir(path, description):
    """Remove file or directory and report savings"""
    if os.path.exists(path):
        size_mb = get_file_size(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"✅ Removed {description}: {size_mb:.1f} MB saved")
        return size_mb
    else:
        print(f"⚠️  {description} not found, skipping")
        return 0

def remove_from_xcode_project(file_path):
    """Remove file reference from Xcode project"""
    project_file = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp.xcodeproj/project.pbxproj"
    filename = os.path.basename(file_path)
    
    try:
        with open(project_file, 'r') as f:
            content = f.read()
        
        # Remove lines containing the filename
        lines = content.split('\n')
        filtered_lines = [line for line in lines if filename not in line]
        
        if len(filtered_lines) != len(lines):
            with open(project_file, 'w') as f:
                f.write('\n'.join(filtered_lines))
            print(f"📝 Removed {filename} from Xcode project")
        
    except Exception as e:
        print(f"⚠️  Could not update Xcode project for {filename}: {e}")

def main():
    app_dir = "/Users/choemanseung/789/hft/iosDemo/TriageApp/TriageApp"
    
    if not os.path.exists(app_dir):
        print(f"❌ App directory not found: {app_dir}")
        return
    
    print("🧹 CLEANING UP TRIAGE APP")
    print("=" * 50)
    
    total_saved = 0
    
    # Files and directories to remove
    cleanup_items = [
        # .mlpackage files (only need .mlmodelc)
        (f"{app_dir}/SmolLM_Medical_Triage.mlpackage", ".mlpackage model (original)"),
        (f"{app_dir}/SmolLM_Medical_Triage_Perfsafe.mlpackage", ".mlpackage model (perfsafe)"),
        (f"{app_dir}/SmolLM_Medical_highcap_4bit.mlpackage", ".mlpackage model (highcap)"),
        (f"{app_dir}/SmolLM_Medical_persafe_4bit.mlpackage", ".mlpackage model (persafe)"),
        
        # MLX model directory (not used by iOS)
        (f"{app_dir}/mlx_model", "MLX model directory"),
        
        # Unused Swift files
        (f"{app_dir}/DatabaseTester.swift", "Database tester"),
        (f"{app_dir}/PipelineTester.swift", "Pipeline tester"),
        (f"{app_dir}/MLXGenerator.swift", "MLX generator"),
        
        # Unused config files
        (f"{app_dir}/model_config.json", "Model config"),
        (f"{app_dir}/model_config_perfsafe.json", "Model config (perfsafe)")
    ]
    
    print("\n🗑️  REMOVING UNNECESSARY FILES:")
    print("-" * 40)
    
    for file_path, description in cleanup_items:
        saved = remove_file_or_dir(file_path, description)
        total_saved += saved
        
        # Remove from Xcode project
        if saved > 0:
            remove_from_xcode_project(file_path)
    
    print("\n" + "=" * 50)
    print(f"💾 TOTAL SPACE SAVED: {total_saved:.1f} MB")
    print("=" * 50)
    
    # Check remaining files
    print("\n📦 REMAINING ESSENTIAL FILES:")
    print("-" * 30)
    
    essential_files = [
        "SmolLM_Medical_highcap_4bit.mlmodelc",
        "SmolLM_Medical_persafe_4bit.mlmodelc",
        "chunks.sqlite",
        "faiss.index",
        "ids.bin"
    ]
    
    remaining_size = 0
    for filename in essential_files:
        path = f"{app_dir}/{filename}"
        if os.path.exists(path):
            size = get_file_size(path)
            remaining_size += size
            print(f"✅ {filename}: {size:.1f} MB")
        else:
            print(f"❌ Missing: {filename}")
    
    print(f"\nTotal essential files: {remaining_size:.1f} MB")
    print(f"App size reduction: {total_saved:.1f} MB → {remaining_size:.1f} MB")
    print(f"Reduction: {(total_saved / (total_saved + remaining_size)) * 100:.1f}%")
    
    print("\n🎉 Cleanup complete!")
    print("💡 Remember to clean and rebuild in Xcode")

if __name__ == "__main__":
    main()