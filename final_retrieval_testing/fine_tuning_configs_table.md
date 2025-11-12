  # Fine-Tuning Configuration Tables

## Base Models

| Model Name | Size | Quantization | Path |
|------------|------|--------------|------|
| SmolLM2-360M | 360M | 8-bit | `/mlx_models/SmolLM2-360M-Instruct-MLX_8bit` |
| SmolLM2-360M | 360M | 4-bit | `/mlx_models/SmolLM2-360M-Instruct-MLX_4bit` |
| SmolLM2-135M | 135M | 8-bit | `/mlx_models/SmolLM2-135M-Instruct-MLX_8bit` |
| SmolLM2-135M | 135M | 4-bit | `/mlx_models/SmolLM2-135M-Instruct-MLX_4bit` |
| Gemma-270M | 270M | 8-bit | `/mlx_models/gemma-270m-mlx_8bit` |
| Gemma-270M | 270M | 4-bit | `/mlx_models/gemma-270m-mlx_4bit` |

**Total Model Variants:** 6 (3 base models × 2 quantization levels)

---

## Standard LoRA Configurations (`triage_lora_finetune.py`)

| Config Name | Learning Rate | Batch Size | LoRA Rank | LoRA Alpha (Scale) | Dropout | Iterations | Purpose |
|-------------|---------------|------------|-----------|-------------------|---------|------------|---------|
| medical_baseline | 3×10⁻⁵ | 4 | 8 | 20.0 | 0.0 | 800 | Baseline for medical tasks |
| medical_precision | 1×10⁻⁵ | 4 | 16 | 25.0 | 0.05 | 1000 | High precision medical adaptation |
| medical_fast | 5×10⁻⁵ | 8 | 8 | 20.0 | 0.0 | 600 | Fast training, larger batches |
| medical_conservative | 1×10⁻⁵ | 2 | 4 | 15.0 | 0.1 | 1200 | Conservative, small rank |
| medical_high_capacity | 2×10⁻⁵ | 4 | 32 | 30.0 | 0.05 | 800 | Maximum adaptation capacity |
| triage_optimized | 4×10⁻⁵ | 6 | 12 | 22.0 | 0.02 | 800 | Optimized for triage decisions |

**Configuration Count:** 6 standard configs

---

## Safety-Enhanced LoRA Configurations (`safety_enhanced_triage_finetune.py`)

| Config Name | Learning Rate | Batch Size | LoRA Rank | LoRA Alpha (Scale) | Dropout | Iterations | Target ED Recall | Safety Priority | Cost-Sensitive Loss |
|-------------|---------------|------------|-----------|-------------------|---------|------------|------------------|-----------------|---------------------|
| ultra_safe | 5×10⁻⁶ | 2 | 4 | 2.0 | 0.15 | 1500 | ≥98% | Maximum | 100× penalty for FN |
| balanced_safe | 1×10⁻⁵ | 4 | 8 | 8.0 | 0.1 | 1200 | ≥96% | High | 100× penalty for FN |
| performance_safe | 2×10⁻⁵ | 6 | 12 | 12.0 | 0.08 | 1000 | ≥95% | Moderate | 100× penalty for FN |
| high_capacity_safe | 8×10⁻⁶ | 2 | 16 | 8.0 | 0.12 | 1400 | ≥98% | Maximum | 100× penalty for FN |

**Configuration Count:** 4 safety-enhanced configs

**Safety Features:**
- **Cost Matrix:** ED→GP/HOME misclassification penalty = 100×
- **Class Weights:** ED=5.0×, GP=2.0×, HOME=1.0×
- **F2-Score:** Recall-weighted evaluation metric
- **MC Dropout:** 50-100 samples for uncertainty quantification
- **Hard Constraints:**
  - ED Recall ≥ 95%
  - ED F2-Score ≥ 90%
  - False Negative Rate ≤ 5%

---

## Training Configuration Common Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | AdamW | Adaptive learning with weight decay |
| Weight Decay | 0.01 | L2 regularization |
| Warmup Ratio | 0.1 | Learning rate warmup |
| Max Sequence Length | 2048 | Maximum token length |
| Num Layers | 16 | Fine-tuning depth |
| Seed | 42 | Reproducibility |
| Steps per Eval | 200 | Evaluation frequency |
| Steps per Report | 50 | Progress reporting |
| Save Every | 100 | Checkpoint frequency |
| Gradient Checkpointing | False | Memory optimization (disabled) |
| Mask Prompt | True | Only compute loss on completions |

---

## Total Experimental Configurations

### Standard Fine-Tuning
- **Models:** 6 quantized variants
- **Configs:** 6 standard LoRA configurations
- **Total Experiments:** 6 × 6 = **36 experiments**

### Safety-Enhanced Fine-Tuning
- **Models:** 6 quantized variants
- **Configs:** 4 safety-enhanced LoRA configurations
- **Total Experiments:** 6 × 4 = **24 experiments**

### Grand Total
**60 fine-tuning experiments** across both standard and safety-enhanced approaches

---

## Memory Footprint Comparison (Approximate)

| Model | 8-bit Memory | 4-bit Memory | Memory Reduction |
|-------|-------------|--------------|------------------|
| SmolLM2-360M | ~720 MB | ~360 MB | ~50% |
| SmolLM2-135M | ~270 MB | ~135 MB | ~50% |
| Gemma-270M | ~540 MB | ~270 MB | ~50% |

**Note:** 4-bit quantization achieves approximately 50% memory reduction with <2% accuracy degradation

---

## LoRA Parameter Efficiency

| LoRA Rank | Approximate Trainable Parameters | % of Full Model |
|-----------|----------------------------------|-----------------|
| 4 | ~0.1-0.2M | <0.1% |
| 8 | ~0.2-0.4M | ~0.1% |
| 12 | ~0.4-0.6M | ~0.2% |
| 16 | ~0.5-0.8M | ~0.2% |
| 32 | ~1.0-1.5M | ~0.4% |

**QLoRA Efficiency:** Only 0.1-0.4% of base model parameters are trainable, enabling resource-constrained fine-tuning

---

## Evaluation Metrics Tracked

### Standard Training
- Final Train Loss
- Final Validation Loss
- Final Test Loss
- Test Accuracy
- Training Time
- Memory Usage

### Safety-Enhanced Training
All standard metrics plus:
- **Per-Class Metrics:** Precision, Recall, F1-Score, F2-Score (ED, GP, HOME)
- **Safety Metrics:**
  - ED Recall (target ≥95%)
  - ED F2-Score (target ≥90%)
  - False Negative Rate (target ≤5%)
  - ED False Negatives (absolute count)
- **Uncertainty Metrics:**
  - Average Uncertainty (entropy-based)
  - High Uncertainty Rate
  - High Uncertainty Cases (flagged examples)
- **Confusion Matrix:** Full 3×3 matrix for ED/GP/HOME
- **Safety Constraints:** Binary pass/fail for each constraint

---

## Dataset Configuration

| Split | File | Format | Purpose |
|-------|------|--------|---------|
| Training | `train.jsonl` | JSONL | Model training |
| Validation | `valid.jsonl` | JSONL | Hyperparameter tuning |
| Test | `test.jsonl` | JSONL | Final evaluation |

**Data Format:** Prompt-completion pairs for medical triage dialogues
**Path:** `./Final_dataset/final_triage_dialogues_mlx/`
