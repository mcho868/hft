#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Medical Triage Models

This script fine-tunes MLX models using LoRA (Low-Rank Adaptation) on the 
triage dialogues dataset for medical decision support.
"""

import subprocess
import json
import os
import time
import csv
from datetime import datetime
import sys
import mlx.core as mx
from mlx_lm import load, generate
from pathlib import Path

# Model configurations - only quantized models
AVAILABLE_MODELS = {
    "SmolLM2-360M_8bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX_8bit",
    "SmolLM2-360M_4bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX_4bit",
    "SmolLM2-135M_8bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-135M-Instruct-MLX_8bit",
    "SmolLM2-135M_4bit": "/Users/choemanseung/789/hft/mlx_models/SmolLM2-135M-Instruct-MLX_4bit",
    "Gemma-270M_8bit": "/Users/choemanseung/789/hft/mlx_models/gemma-270m-mlx_8bit",
    "Gemma-270M_4bit": "/Users/choemanseung/789/hft/mlx_models/gemma-270m-mlx_4bit"
}

# Filter to only include models that actually exist
ALL_MODELS = {}
for name, path in AVAILABLE_MODELS.items():
    if Path(path).exists():
        ALL_MODELS[name] = path
    else:
        print(f"⚠️ Model not found: {path}")

# LoRA configurations optimized for medical triage
TRIAGE_CONFIGS = [
    # Configuration 1: Baseline for medical tasks
    {
        "name": "medical_baseline",
        "learning_rate": 3e-5,
        "batch_size": 4,
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
        "iters": 800
    },
    # Configuration 2: High precision medical
    {
        "name": "medical_precision",
        "learning_rate": 1e-5,
        "batch_size": 4,
        "rank": 16,
        "scale": 25.0,
        "dropout": 0.05,
        "iters": 1000
    },
    # Configuration 3: Fast medical training
    {
        "name": "medical_fast",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
        "iters": 600
    },
    # Configuration 4: Conservative medical
    {
        "name": "medical_conservative",
        "learning_rate": 1e-5,
        "batch_size": 2,
        "rank": 4,
        "scale": 15.0,
        "dropout": 0.1,
        "iters": 1200
    },
    # Configuration 5: High capacity medical
    {
        "name": "medical_high_capacity",
        "learning_rate": 2e-5,
        "batch_size": 4,
        "rank": 32,
        "scale": 30.0,
        "dropout": 0.05,
        "iters": 800
    },
    # Configuration 6: Optimized for triage decisions
    {
        "name": "triage_optimized",
        "learning_rate": 4e-5,
        "batch_size": 6,
        "rank": 12,
        "scale": 22.0,
        "dropout": 0.02,
        "iters": 800
    }
]

def create_lora_config(config, model_path, adapter_path):
    """Create LoRA configuration file for triage training"""
    lora_config = {
        "adapter_path": adapter_path,
        "batch_size": config["batch_size"],
        "config": None,
        "data": "./Final_dataset/final_triage_dialogues_mlx",  # Updated to use cleaned triage data
        "fine_tune_type": "lora",
        "grad_checkpoint": False,
        "iters": config["iters"],
        "learning_rate": config["learning_rate"],
        "lora_parameters": {
            "rank": config["rank"],
            "dropout": config["dropout"],
            "scale": config["scale"]
        },
        "lr_schedule": None,
        "mask_prompt": True,
        "max_seq_length": 2048,
        "model": model_path,
        "num_layers": 16,
        "optimizer": "adamw",  # AdamW often works better for medical tasks
        "optimizer_config": {
            "adamw": {
                "weight_decay": 0.01  # Add weight decay for better generalization
            }
        },
        "resume_adapter_file": None,
        "save_every": 100,
        "seed": 42,  # Fixed seed for reproducibility
        "steps_per_eval": 200,
        "steps_per_report": 50,
        "test": True,
        "test_batches": -1,
        "train": True,
        "val_batches": -1,
        "warmup_ratio": 0.1,  # Add warmup for stable training
        "wandb": None
    }
    return lora_config

def run_training_live(model_name, model_path, config, results_dir):
    """Run training with live output visible"""
    experiment_name = f"triage_{model_name}_{config['name']}"
    adapter_path = f"triage_adapters/adapter_{experiment_name}"
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting triage training: {experiment_name}")
    print(f"📊 Config: {config}")
    print(f"📁 Adapter path: {adapter_path}")
    print(f"🏥 Training on medical triage data")
    print(f"{'='*60}")
    
    # Create adapter directory
    os.makedirs(adapter_path, exist_ok=True)
    
    # Create config file
    lora_config = create_lora_config(config, model_path, adapter_path)
    config_file = f"{results_dir}/config_{experiment_name}.json"
    
    with open(config_file, 'w') as f:
        json.dump(lora_config, f, indent=2)
    
    # Build training command
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--config", config_file
    ]
    
    # Run training with LIVE output
    start_time = time.time()
    print(f"🏃 Running: {' '.join(cmd)}")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("\n" + "🔥" * 50 + " LIVE TRAINING OUTPUT " + "🔥" * 50)
    
    try:
        # Stream output in real-time while also capturing it
        log_file = f"{results_dir}/training_log_{experiment_name}.txt"
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                # Show live output
                print(line.rstrip())
                sys.stdout.flush()
                
                # Save to log file
                log.write(line)
                log.flush()
                
                # Collect for analysis
                output_lines.append(line.rstrip())
            
            process.wait()
            return_code = process.returncode
        
        training_time = time.time() - start_time
        
        print(f"\n" + "✅" * 50 + f" COMPLETED IN {training_time:.1f}s " + "✅" * 50)
        
        # Parse results
        full_output = '\n'.join(output_lines)
        metrics = parse_training_output(full_output, "")
        
        if return_code == 0:
            print(f"✅ Training SUCCESS for {experiment_name}")
            metrics["status"] = "completed"
            
            # Run triage-specific evaluation
            test_accuracy = evaluate_triage_accuracy(model_path, adapter_path)
            if test_accuracy is not None:
                metrics["triage_test_accuracy"] = test_accuracy
                print(f"🎯 Triage Test Accuracy: {test_accuracy:.4f}")
            else:
                print(f"⚠️ Could not compute triage test accuracy")
        else:
            print(f"❌ Training FAILED for {experiment_name}")
            metrics["status"] = "failed"
        
        # Save results
        metrics.update({
            "experiment_name": experiment_name,
            "model_name": model_name,
            "config": config,
            "training_time": training_time,
            "timestamp": datetime.now().isoformat(),
            "data_type": "medical_triage"
        })
        
        results_file = f"{results_dir}/results_{experiment_name}.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
        
    except Exception as e:
        print(f"❌ Exception during training: {e}")
        return {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "config": config,
            "status": "error",
            "error": str(e)
        }

def parse_training_output(stdout, stderr):
    """Parse training output to extract metrics"""
    metrics = {}
    
    lines = stdout.split('\n')
    
    final_train_loss = None
    final_val_loss = None
    final_test_loss = None
    final_test_accuracy = None
    
    for line in lines:
        line = line.strip()
        
        if "Train loss" in line:
            try:
                final_train_loss = float(line.split("Train loss:")[1].split()[0])
            except:
                pass
        
        if "Val loss" in line:
            try:
                final_val_loss = float(line.split("Val loss:")[1].split()[0])
            except:
                pass
        
        if "Test loss" in line:
            try:
                final_test_loss = float(line.split("Test loss:")[1].split()[0])
            except:
                pass
                
        if "Test accuracy" in line or "Test acc" in line:
            try:
                if "Test accuracy" in line:
                    final_test_accuracy = float(line.split("Test accuracy:")[1].split()[0])
                else:
                    final_test_accuracy = float(line.split("Test acc:")[1].split()[0])
            except:
                pass
    
    metrics.update({
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy
    })
    
    return metrics

def evaluate_triage_accuracy(model_path, adapter_path, max_examples=100):
    """Evaluate triage decision accuracy on test set"""
    print(f"\n🏥 Running triage decision evaluation...")
    
    try:
        # Load test dataset
        test_file = "./Final_dataset/final_triage_dialogues_mlx/test.jsonl"
        test_data = []
        
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        # Limit examples for faster evaluation
        if len(test_data) > max_examples:
            test_data = test_data[:max_examples]
            print(f"📊 Using first {len(test_data)} test examples")
        else:
            print(f"📊 Loaded {len(test_data)} test examples")
        
        # Load model and adapter
        print(f"🔄 Loading model from {model_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        correct = 0
        total = len(test_data)
        
        print(f"🚀 Starting triage evaluation on {total} examples...")
        
        for i, example in enumerate(test_data):
            if i % 10 == 0:
                print(f"   Progress: {i}/{total} ({i/total*100:.1f}%) - Accuracy so far: {correct/max(i,1):.3f}")
            
            prompt = example["prompt"]
            true_completion = example["completion"].strip()
            
            # Generate response
            try:
                response = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=200,  # Longer for triage decisions
                    verbose=False
                )
                
                # Extract just the generated part (remove the prompt)
                generated = response.replace(prompt, "").strip()
                
                # Extract triage decision
                pred_decision = extract_triage_decision(generated)
                true_decision = extract_triage_decision(true_completion)
                
                if pred_decision == true_decision:
                    correct += 1
                
                # Show a few examples for debugging
                if i < 3:
                    print(f"   Example {i}: True='{true_decision}' Pred='{pred_decision}' ({'✅' if pred_decision == true_decision else '❌'})")
                    
            except Exception as e:
                print(f"   Error on example {i}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        print(f"✅ Triage Decision Accuracy: {correct}/{total} = {accuracy:.4f}")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Triage evaluation failed: {e}")
        return None

def extract_triage_decision(text):
    """Extract triage decision (ED, GP, HOME) from text"""
    text = text.lower().strip()
    
    # Look for explicit triage decisions
    if "triage decision: ed" in text or "triage decision:ed" in text:
        return "ED"
    elif "triage decision: gp" in text or "triage decision:gp" in text:
        return "GP"
    elif "triage decision: home" in text or "triage decision:home" in text:
        return "HOME"
    
    # Look for keywords
    if "emergency" in text or " ed " in text:
        return "ED"
    elif "gp" in text or "doctor" in text:
        return "GP"
    elif "home" in text or "rest" in text:
        return "HOME"
    else:
        return "UNKNOWN"

def generate_triage_summary(all_results, results_dir):
    """Generate summary of triage training results"""
    successful = [r for r in all_results if r.get("status") == "completed" and r.get("triage_test_accuracy") is not None]
    
    if not successful:
        print("❌ No successful triage training experiments!")
        return
    
    successful.sort(key=lambda x: x.get("triage_test_accuracy", 0), reverse=True)
    
    print(f"\n🥇 TOP PERFORMING TRIAGE MODELS:")
    for i, result in enumerate(successful[:5]):
        config = result.get("config", {})
        acc = result.get('triage_test_accuracy', 0) or 0
        time_val = result.get('training_time', 0) or 0
        print(f"{i+1}. {result['experiment_name']:<30} | "
              f"Acc: {acc:.4f} | "
              f"Time: {time_val:.1f}s")
    
    if successful:
        best = successful[0]
        best_acc = best.get('triage_test_accuracy', 0) or 0
        best_time = best.get('training_time', 0) or 0
        print(f"\n🏆 BEST TRIAGE MODEL: {best['experiment_name']}")
        print(f"   🎯 Triage Accuracy: {best_acc:.4f}")
        print(f"   ⏱️  Training Time: {best_time:.1f} seconds")
        print(f"   📁 Adapter Path: triage_adapters/adapter_{best['experiment_name']}")
    
    # Save CSV summary
    csv_file = f"{results_dir}/triage_results_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'experiment_name', 'model_name', 'config_name', 'triage_accuracy',
            'train_loss', 'val_loss', 'test_loss', 'training_time'
        ])
        
        for result in successful:
            writer.writerow([
                result['experiment_name'],
                result['model_name'],
                result['config']['name'],
                result.get('triage_test_accuracy', 0),
                result.get('final_train_loss', 0),
                result.get('final_val_loss', 0),
                result.get('final_test_loss', 0),
                result.get('training_time', 0)
            ])
    
    print(f"💾 Summary saved to: {csv_file}")

def main():
    print("🏥 Medical Triage LoRA Fine-tuning with Live Output")
    print(f"📊 Will test {len(TRIAGE_CONFIGS)} configurations on {len(ALL_MODELS)} models")
    print(f"🎯 Total experiments: {len(TRIAGE_CONFIGS) * len(ALL_MODELS)}")
    print(f"👀 You'll see LIVE training progress for medical triage tasks!")
    print(f"⏱️  Estimated time: ~{len(TRIAGE_CONFIGS) * len(ALL_MODELS) * 6} minutes")
    print(f"🏥 Training on medical triage dialogues dataset")
    print("\n🧪 Available Models:")
    for name, path in ALL_MODELS.items():
        print(f"   • {name}: {path}")
    print("\n🧪 Configuration Categories:")
    print("   • Medical Baseline: Standard medical fine-tuning")
    print("   • Medical Precision: High-rank, low learning rate")  
    print("   • Medical Fast: Quick training for testing")
    print("   • Medical Conservative: Stable, regularized training")
    print("   • Medical High Capacity: Large rank for complex patterns")
    print("   • Triage Optimized: Specifically tuned for triage decisions")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"triage_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create adapters directory
    os.makedirs("triage_adapters", exist_ok=True)
    
    all_results = []
    
    # Run experiments
    for model_name, model_path in ALL_MODELS.items():
        if not Path(model_path).exists():
            print(f"⚠️  Model not found: {model_path}, skipping...")
            continue
            
        print(f"\n🏃 Testing model: {model_name}")
        
        for config in TRIAGE_CONFIGS:
            try:
                result = run_training_live(model_name, model_path, config, results_dir)
                all_results.append(result)
                
                # Show quick summary
                if result.get("status") == "completed":
                    acc = result.get("triage_test_accuracy")
                    if acc is not None:
                        print(f"📈 Quick Result: Triage Accuracy = {acc:.4f}")
                    else:
                        print(f"📈 Quick Result: Triage Accuracy = Not Available")
                
                # Save progress
                progress_file = f"{results_dir}/progress.json"
                with open(progress_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                continue
    
    # Final analysis
    print("\n" + "🏆" * 50 + " FINAL TRIAGE RESULTS " + "🏆" * 50)
    generate_triage_summary(all_results, results_dir)

if __name__ == "__main__":
    main()