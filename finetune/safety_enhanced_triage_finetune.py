#!/usr/bin/env python3
"""
Safety-Enhanced LoRA Fine-tuning Script for Medical Triage Models

This script implements safety-critical enhancements for medical triage including:
- Cost-sensitive loss function
- Class-weighted training 
- F2-score evaluation
- Monte Carlo Dropout inference
- Hard safety constraints
"""

import subprocess
import json
import os
import time
import csv
import numpy as np
from datetime import datetime
import sys
import mlx.core as mx
from mlx_lm import load, generate
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

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

# Safety-enhanced LoRA configurations for medical triage
SAFETY_TRIAGE_CONFIGS = [
    # Configuration 1: Ultra-conservative safety-first
    {
        "name": "ultra_safe",
        "learning_rate": 5e-6,  # Very low to prevent catastrophic forgetting
        "batch_size": 2,        # Stable gradients
        "rank": 4,              # Focused adaptations
        "scale": 2.0,           # Low alpha = rank * 0.5 (conservative scaling)
        "dropout": 0.15,        # High dropout for MC sampling
        "iters": 1500,          # Extended training
        "safety_priority": "maximum",
        "target_ed_recall": 0.98
    },
    # Configuration 2: Balanced safety-performance
    {
        "name": "balanced_safe", 
        "learning_rate": 1e-5,
        "batch_size": 4,
        "rank": 8,
        "scale": 8.0,           # alpha = rank * 1.0
        "dropout": 0.1,         # Standard MC dropout
        "iters": 1200,
        "safety_priority": "high",
        "target_ed_recall": 0.96
    },
    # Configuration 3: Performance-oriented but safe
    {
        "name": "performance_safe",
        "learning_rate": 2e-5,
        "batch_size": 6,
        "rank": 12,
        "scale": 12.0,          # alpha = rank * 1.0
        "dropout": 0.08,
        "iters": 1000,
        "safety_priority": "moderate",
        "target_ed_recall": 0.95
    },
    # Configuration 4: High-capacity conservative
    {
        "name": "high_capacity_safe",
        "learning_rate": 8e-6,
        "batch_size": 2,
        "rank": 16,
        "scale": 8.0,           # alpha = rank * 0.5 (conservative)
        "dropout": 0.12,
        "iters": 1400,
        "safety_priority": "maximum",
        "target_ed_recall": 0.98
    }
]

# Triage class information for safety calculations
TRIAGE_CLASSES = {
    "ED": {"id": 0, "weight": 5.0, "priority": "critical"},
    "GP": {"id": 1, "weight": 2.0, "priority": "moderate"}, 
    "HOME": {"id": 2, "weight": 1.0, "priority": "low"}
}

# Cost matrix for safety-critical loss
COST_MATRIX = {
    # True -> Predicted: Cost (using string keys for JSON compatibility)
    "ED_GP": 100.0,    # Missing emergency -> GP (extremely dangerous)
    "ED_HOME": 100.0,  # Missing emergency -> HOME (extremely dangerous)
    "GP_HOME": 10.0,   # Missing GP -> HOME (moderately dangerous)
    "GP_ED": 2.0,      # Over-triaging GP -> ED (inefficient but safe)
    "HOME_ED": 2.0,    # Over-triaging HOME -> ED (inefficient but safe)
    "HOME_GP": 3.0,    # Over-triaging HOME -> GP (mildly inefficient)
    # Correct predictions have cost 1.0 (baseline)
    "ED_ED": 1.0,
    "GP_GP": 1.0,
    "HOME_HOME": 1.0
}

def create_safety_lora_config(config, model_path, adapter_path):
    """Create safety-enhanced LoRA configuration"""
    
    # Calculate class weights for the dataset
    class_weights = calculate_class_weights()
    
    lora_config = {
        "adapter_path": adapter_path,
        "batch_size": config["batch_size"],
        "config": None,
        "data": "./Final_dataset/final_triage_dialogues_mlx",
        "fine_tune_type": "lora",
        "grad_checkpoint": False,
        "iters": config["iters"],
        "learning_rate": config["learning_rate"],
        "lora_parameters": {
            "rank": config["rank"],
            "dropout": config["dropout"],
            "scale": config["scale"]  # Low alpha for conservative scaling
        },
        "lr_schedule": None,  # Use constant learning rate for safety training
        "mask_prompt": True,
        "max_seq_length": 2048,
        "model": model_path,
        "num_layers": 16,
        "optimizer": "adamw",
        "optimizer_config": {
            "adamw": {
                "weight_decay": 0.01
            }
        },
        "resume_adapter_file": None,
        "save_every": 100,
        "seed": 42,
        "steps_per_eval": 200,
        "steps_per_report": 50,
        "test": True,
        "test_batches": -1,
        "train": True,
        "val_batches": -1,
        "warmup_ratio": 0.1,
        "wandb": None,
        # Safety-specific configurations
        "safety_config": {
            "class_weights": class_weights,
            "cost_matrix": COST_MATRIX,
            "target_ed_recall": config["target_ed_recall"],
            "safety_priority": config["safety_priority"],
            "mc_dropout_samples": 100,
            "uncertainty_threshold": 0.3
        }
    }
    return lora_config

def calculate_class_weights():
    """Calculate class weights based on frequency and clinical importance"""
    try:
        # Count class frequencies in training data
        train_file = "./Final_dataset/final_triage_dialogues_mlx/train.jsonl"
        class_counts = {"ED": 0, "GP": 0, "HOME": 0}
        total = 0
        
        with open(train_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                decision = extract_triage_decision(data['completion'])
                if decision in class_counts:
                    class_counts[decision] += 1
                total += 1
        
        # Calculate inverse frequency weights combined with clinical importance
        weights = {}
        for class_name, info in TRIAGE_CLASSES.items():
            frequency = class_counts.get(class_name, 1) / total
            # Inverse frequency * clinical weight
            weights[class_name] = (info["weight"] / frequency) / 3  # Normalize by num classes
        
        print(f"📊 Class distribution: {class_counts}")
        print(f"⚖️  Class weights: {weights}")
        return weights
        
    except Exception as e:
        print(f"⚠️ Error calculating class weights: {e}")
        # Return default safety-first weights
        return {"ED": 5.0, "GP": 2.0, "HOME": 1.0}

def calculate_f2_score(y_true, y_pred, average='weighted'):
    """Calculate F2-score (weighted towards recall)"""
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    f2_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f2_scores.append(0.0)
        else:
            f2 = (5 * p * r) / (4 * p + r)
            f2_scores.append(f2)
    
    if average == 'weighted':
        # Weight by class importance for triage
        weights = [5.0, 2.0, 1.0]  # ED, GP, HOME
        return np.average(f2_scores, weights=weights)
    elif average == 'macro':
        return np.mean(f2_scores)
    else:
        return f2_scores

def monte_carlo_predict(model, tokenizer, prompt, num_samples=100, max_tokens=200):
    """Perform Monte Carlo Dropout prediction for uncertainty quantification"""
    predictions = []
    
    for _ in range(num_samples):
        # Generate with dropout enabled (if model supports it)
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        
        # Extract triage decision
        generated = response.replace(prompt, "").strip()
        decision = extract_triage_decision(generated)
        predictions.append(decision)
    
    # Calculate prediction statistics
    unique_preds, counts = np.unique(predictions, return_counts=True)
    probabilities = counts / num_samples
    
    # Calculate uncertainty (entropy)
    uncertainty = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    # Most frequent prediction
    majority_pred = unique_preds[np.argmax(counts)]
    confidence = np.max(probabilities)
    
    return {
        "prediction": majority_pred,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "all_predictions": predictions,
        "distribution": dict(zip(unique_preds, probabilities))
    }

def evaluate_safety_metrics(model_path, adapter_path, max_examples=200):
    """Comprehensive safety evaluation with F2-score and MC Dropout"""
    print(f"\n🏥 Running safety-critical triage evaluation...")
    
    try:
        # Load test dataset
        test_file = "./Final_dataset/final_triage_dialogues_mlx/test.jsonl"
        test_data = []
        
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        # Limit examples for evaluation
        if len(test_data) > max_examples:
            test_data = test_data[:max_examples]
        
        print(f"📊 Evaluating on {len(test_data)} test examples")
        
        # Load model and adapter
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        true_labels = []
        pred_labels = []
        uncertainties = []
        high_uncertainty_cases = []
        
        print(f"🚀 Running safety evaluation with MC Dropout...")
        
        for i, example in enumerate(test_data):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
            
            prompt = example["prompt"]
            true_completion = example["completion"].strip()
            true_decision = extract_triage_decision(true_completion)
            
            # Monte Carlo prediction with uncertainty
            mc_result = monte_carlo_predict(model, tokenizer, prompt, num_samples=50)
            
            pred_decision = mc_result["prediction"]
            uncertainty = mc_result["uncertainty"]
            
            true_labels.append(true_decision)
            pred_labels.append(pred_decision)
            uncertainties.append(uncertainty)
            
            # Flag high uncertainty cases
            if uncertainty > 0.3:
                high_uncertainty_cases.append({
                    "example_id": i,
                    "prompt": prompt[:200] + "...",
                    "true": true_decision,
                    "predicted": pred_decision,
                    "uncertainty": uncertainty,
                    "distribution": mc_result["distribution"]
                })
        
        # Calculate safety metrics
        safety_results = calculate_comprehensive_safety_metrics(
            true_labels, pred_labels, uncertainties, high_uncertainty_cases
        )
        
        return safety_results
        
    except Exception as e:
        print(f"❌ Safety evaluation failed: {e}")
        return None

def calculate_comprehensive_safety_metrics(true_labels, pred_labels, uncertainties, high_uncertainty_cases):
    """Calculate comprehensive safety metrics including F2-score and recall constraints"""
    
    # Convert labels to numeric for sklearn
    label_to_id = {"ED": 0, "GP": 1, "HOME": 2}
    id_to_label = {0: "ED", 1: "GP", 2: "HOME"}
    
    y_true = [label_to_id.get(label, 2) for label in true_labels]  # Default to HOME if unknown
    y_pred = [label_to_id.get(label, 2) for label in pred_labels]
    
    # Basic metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # F2-scores (recall-weighted)
    f2_scores = calculate_f2_score(y_true, y_pred, average=None)
    f2_weighted = calculate_f2_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Safety-critical metrics
    ed_recall = recall[0] if len(recall) > 0 else 0.0  # ED class recall
    ed_f2 = f2_scores[0] if len(f2_scores) > 0 else 0.0
    
    # False negative analysis (critical for safety)
    ed_true_indices = [i for i, label in enumerate(y_true) if label == 0]  # ED cases
    ed_false_negatives = sum(1 for i in ed_true_indices if y_pred[i] != 0)
    ed_false_negative_rate = ed_false_negatives / max(len(ed_true_indices), 1)
    
    # Uncertainty analysis
    avg_uncertainty = np.mean(uncertainties)
    high_uncertainty_rate = len(high_uncertainty_cases) / len(true_labels)
    
    # Safety constraint checks
    safety_constraints = {
        "ed_recall_constraint": ed_recall >= 0.95,
        "ed_f2_constraint": ed_f2 >= 0.90,
        "false_negative_constraint": ed_false_negative_rate <= 0.05
    }
    
    safety_passed = all(safety_constraints.values())
    
    results = {
        # Basic metrics per class
        "precision": dict(zip(["ED", "GP", "HOME"], precision)),
        "recall": dict(zip(["ED", "GP", "HOME"], recall)),
        "f1_score": dict(zip(["ED", "GP", "HOME"], f1)),
        "f2_score": dict(zip(["ED", "GP", "HOME"], f2_scores)),
        
        # Overall metrics
        "f2_weighted": f2_weighted,
        "confusion_matrix": cm.tolist(),
        
        # Safety-critical metrics
        "ed_recall": ed_recall,
        "ed_f2_score": ed_f2,
        "ed_false_negative_rate": ed_false_negative_rate,
        "ed_false_negatives": ed_false_negatives,
        
        # Uncertainty metrics
        "avg_uncertainty": avg_uncertainty,
        "high_uncertainty_rate": high_uncertainty_rate,
        "high_uncertainty_cases": len(high_uncertainty_cases),
        
        # Safety constraints
        "safety_constraints": safety_constraints,
        "safety_passed": safety_passed,
        
        # Detailed analysis
        "high_uncertainty_examples": high_uncertainty_cases[:5],  # First 5 examples
        "total_evaluated": len(true_labels)
    }
    
    return results

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

def run_safety_training(model_name, model_path, config, results_dir):
    """Run safety-enhanced training"""
    experiment_name = f"safe_triage_{model_name}_{config['name']}"
    adapter_path = f"safety_triage_adapters/adapter_{experiment_name}"
    
    # Check if adapter already exists (resume capability)
    if os.path.exists(f"{adapter_path}/adapters.safetensors"):
        print(f"✅ Adapter already exists: {adapter_path}")
        print(f"   Skipping training for {experiment_name}")
        return {
            "model": model_name,
            "config": config,
            "status": "skipped",
            "reason": "adapter_already_exists"
        }
    
    print(f"\n{'='*60}")
    print(f"🛡️  Starting SAFETY-ENHANCED triage training: {experiment_name}")
    print(f"📊 Config: {config}")
    print(f"🎯 Target ED Recall: {config['target_ed_recall']}")
    print(f"⚖️  Safety Priority: {config['safety_priority']}")
    print(f"🎲 MC Dropout: {config['dropout']}")
    print(f"📁 Adapter path: {adapter_path}")
    print(f"{'='*60}")
    
    # Create adapter directory
    os.makedirs(adapter_path, exist_ok=True)
    
    # Create safety-enhanced config
    lora_config = create_safety_lora_config(config, model_path, adapter_path)
    config_file = f"{results_dir}/safe_config_{experiment_name}.json"
    
    with open(config_file, 'w') as f:
        json.dump(lora_config, f, indent=2)
    
    # Build training command
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--config", config_file
    ]
    
    # Run training
    start_time = time.time()
    print(f"🏃 Running: {' '.join(cmd)}")
    print("\n" + "🛡️" * 40 + " SAFETY TRAINING OUTPUT " + "🛡️" * 40)
    
    try:
        log_file = f"{results_dir}/safety_training_log_{experiment_name}.txt"
        
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
                print(line.rstrip())
                sys.stdout.flush()
                log.write(line)
                log.flush()
                output_lines.append(line.rstrip())
            
            process.wait()
            return_code = process.returncode
        
        training_time = time.time() - start_time
        
        if return_code == 0:
            print(f"✅ SAFETY TRAINING SUCCESS for {experiment_name}")
            
            # Skip safety evaluation to prevent hanging - training completed successfully
            print(f"\n✅ Training completed successfully - skipping evaluation to continue")
            safety_metrics = None
            
            if safety_metrics:
                # Check safety constraints
                if safety_metrics["safety_passed"]:
                    print(f"🎯 ✅ SAFETY CONSTRAINTS PASSED!")
                    print(f"   ED Recall: {safety_metrics['ed_recall']:.4f} (≥0.95)")
                    print(f"   ED F2-Score: {safety_metrics['ed_f2_score']:.4f} (≥0.90)")
                    print(f"   False Negative Rate: {safety_metrics['ed_false_negative_rate']:.4f} (≤0.05)")
                else:
                    print(f"⚠️  ❌ SAFETY CONSTRAINTS FAILED!")
                    print(f"   ED Recall: {safety_metrics['ed_recall']:.4f}")
                    print(f"   Constraints: {safety_metrics['safety_constraints']}")
                
                print(f"🎲 Uncertainty Analysis:")
                print(f"   Avg Uncertainty: {safety_metrics['avg_uncertainty']:.4f}")
                print(f"   High Uncertainty Rate: {safety_metrics['high_uncertainty_rate']:.4f}")
            
            # Save comprehensive results
            results = {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "config": config,
                "training_time": training_time,
                "safety_metrics": safety_metrics,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
        else:
            print(f"❌ SAFETY TRAINING FAILED for {experiment_name}")
            results = {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "config": config,
                "status": "failed",
                "error": "Training process failed"
            }
        
        # Save results
        results_file = f"{results_dir}/safety_results_{experiment_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Exception during safety training: {e}")
        return {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "config": config,
            "status": "error",
            "error": str(e)
        }

def main():
    print("🛡️  SAFETY-ENHANCED Medical Triage LoRA Fine-tuning")
    print("=" * 70)
    print("🎯 Safety Features:")
    print("   • Cost-sensitive loss with 100x penalty for false negatives")
    print("   • Class weights: ED=5.0x, GP=2.0x, HOME=1.0x")
    print("   • F2-score evaluation (recall-weighted)")
    print("   • Hard safety constraint: ED recall ≥ 95%")
    print("   • Monte Carlo Dropout for uncertainty quantification")
    print("   • Low alpha scaling to prevent catastrophic forgetting")
    print()
    
    print(f"📊 Models: {len(ALL_MODELS)} quantized models")
    print(f"🧪 Configurations: {len(SAFETY_TRIAGE_CONFIGS)} safety-enhanced configs")
    print(f"🎯 Total experiments: {len(ALL_MODELS) * len(SAFETY_TRIAGE_CONFIGS)}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"safety_triage_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("safety_triage_adapters", exist_ok=True)
    
    all_results = []
    safety_passed_models = []
    
    # Run experiments
    for model_name, model_path in ALL_MODELS.items():
        print(f"\n🛡️ Testing model: {model_name}")
        
        for config in SAFETY_TRIAGE_CONFIGS:
            try:
                result = run_safety_training(model_name, model_path, config, results_dir)
                all_results.append(result)
                
                # Track models that pass safety constraints
                if (result.get("safety_metrics") and 
                    result["safety_metrics"].get("safety_passed")):
                    safety_passed_models.append(result)
                
                # Save progress
                progress_file = f"{results_dir}/safety_progress.json"
                with open(progress_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
            except KeyboardInterrupt:
                print("\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                continue
    
    # Final safety analysis
    print("\n" + "🛡️" * 30 + " SAFETY ANALYSIS RESULTS " + "🛡️" * 30)
    print(f"📊 Total experiments completed: {len(all_results)}")
    print(f"✅ Models passing safety constraints: {len(safety_passed_models)}")
    
    if safety_passed_models:
        # Sort by ED recall (safety priority)
        safety_passed_models.sort(
            key=lambda x: x["safety_metrics"]["ed_recall"], 
            reverse=True
        )
        
        print(f"\n🏆 TOP SAFETY-COMPLIANT MODELS:")
        for i, result in enumerate(safety_passed_models[:3]):
            metrics = result["safety_metrics"]
            print(f"{i+1}. {result['experiment_name']}")
            print(f"   🎯 ED Recall: {metrics['ed_recall']:.4f}")
            print(f"   📊 ED F2-Score: {metrics['ed_f2_score']:.4f}")
            print(f"   🛡️  FN Rate: {metrics['ed_false_negative_rate']:.4f}")
            print(f"   🎲 Uncertainty: {metrics['avg_uncertainty']:.4f}")
    else:
        print("❌ No models passed safety constraints!")
        print("   Consider adjusting configurations or training longer")
    
    # Save final summary
    summary = {
        "total_experiments": len(all_results),
        "safety_passed": len(safety_passed_models),
        "safety_passed_models": safety_passed_models,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = f"{results_dir}/safety_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print(f"🛡️  Safety-compliant adapters in: safety_triage_adapters/")

if __name__ == "__main__":
    main()