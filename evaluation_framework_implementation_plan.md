# Medical Triage System Evaluation Framework Implementation Plan

## Overview

This comprehensive evaluation framework will systematically test all combinations of RAG configurations with trained adapters to evaluate:
- **Triage Accuracy**: Correct ED/GP/HOME classification
- **Next Steps Accuracy**: Appropriate medical recommendations  
- **Clinical Appropriateness**: LLM-as-judge evaluation of decision quality
- **System Performance**: Memory usage and inference speed profiling

## Current Assets

### Trained Adapters (60 total)
- **Standard Adapters**: 36 configurations across SmolLM2-360M, SmolLM2-135M, Gemma-270M
- **Safety-Enhanced Adapters**: 24 configurations with cost-sensitive loss and uncertainty quantification
- **Locations**: `/Users/choemanseung/789/hft/triage_adapters/` and `/Users/choemanseung/789/hft/safety_triage_adapters/`

### RAG Configuration Matrix
- **Chunking Methods**: Fixed-size, semantic, recursive
- **Retrieval Methods**: Dense (embedding), sparse (BM25), hybrid
- **Top-k Values**: 3, 5, 10
- **Estimated Combinations**: 1,200-1,500 RAG configs × adapters

## Implementation Plan

### Phase 1: Configuration Matrix Generation
**Script**: `config_matrix_generator.py`

```python
# Generate all possible combinations of:
RAG_CONFIGS = {
    "chunking_methods": ["fixed_size", "semantic", "recursive"],
    "retrieval_methods": ["dense", "sparse", "hybrid"], 
    "top_k_values": [3, 5, 10],
    "chunk_sizes": [256, 512, 1024],
    "overlap_ratios": [0.1, 0.2, 0.3]
}

ADAPTER_CONFIGS = {
    "standard_adapters": scan_directory("/triage_adapters/"),
    "safety_adapters": scan_directory("/safety_triage_adapters/")
}
```

**Key Functions**:
- `generate_rag_combinations()`: Create comprehensive RAG config matrix
- `scan_adapter_directories()`: Enumerate all trained adapters
- `create_evaluation_matrix()`: Combine RAG configs with adapters
- `estimate_execution_time()`: Calculate total evaluation runtime

### Phase 2: Modular Evaluation Pipeline
**Script**: `evaluation_pipeline.py`

**Core Components**:
```python
class EvaluationPipeline:
    def __init__(self, config_matrix, batch_size=32):
        self.config_matrix = config_matrix
        self.batch_size = batch_size
        self.results_db = ResultsDatabase()
    
    def run_evaluation(self, config_combo):
        # Load RAG system with specific config
        # Load adapter with specific model
        # Run test dataset through system
        # Collect all metrics
        # Store results with configuration metadata
```

**Pipeline Stages**:
1. **RAG System Setup**: Configure chunking, retrieval, and ranking
2. **Model Loading**: Load base model with specific adapter
3. **Batch Processing**: Process test cases in memory-efficient batches
4. **Metrics Collection**: Gather accuracy, performance, and safety metrics
5. **Results Storage**: Persist results with full configuration traceability

### Phase 3: Metrics Collection System
**Script**: `metrics_collector.py`

**Accuracy Metrics**:
```python
def evaluate_triage_accuracy(predictions, ground_truth):
    # Standard classification metrics
    # F1-score, precision, recall per class
    # Confusion matrix analysis
    # Cost-sensitive accuracy using medical cost matrix

def evaluate_next_steps_accuracy(generated_steps, reference_steps):
    # Semantic similarity scoring
    # Medical terminology matching
    # Treatment pathway appropriateness
```

**Performance Metrics**:
```python
def profile_memory_usage(model, tokenizer, test_batch):
    # Peak memory consumption during inference
    # Memory efficiency per token generated
    # GPU/CPU memory distribution

def measure_inference_speed(model, tokenizer, test_cases):
    # Tokens per second generation rate
    # Latency per request (p50, p95, p99)
    # Throughput under concurrent load
```

### Phase 4: LLM-as-Judge Appropriateness Evaluator
**Script**: `appropriateness_judge.py`

**Clinical Decision Evaluation**:
```python
class ClinicalAppropriatenessJudge:
    def __init__(self, judge_model="gpt-4-turbo"):
        self.judge_model = judge_model
        self.evaluation_prompt = self.load_clinical_prompt()
    
    def evaluate_decision(self, case_description, model_recommendation):
        # Assess clinical reasoning quality
        # Evaluate risk assessment appropriateness  
        # Check for dangerous misclassifications
        # Score medical recommendation safety
```

**Evaluation Dimensions**:
- **Safety**: Risk of harm from incorrect triage
- **Efficiency**: Appropriate resource utilization
- **Completeness**: Coverage of relevant medical factors
- **Clinical Reasoning**: Quality of diagnostic logic

### Phase 5: Memory and Performance Profiling
**Script**: `performance_profiler.py`

**System Resource Monitoring**:
```python
class PerformanceProfiler:
    def profile_model_hosting(self, model_path, adapter_path):
        # Memory footprint during model loading
        # RAM usage during inference
        # GPU utilization patterns
        # Disk I/O requirements
    
    def benchmark_inference_speed(self, model, test_dataset):
        # Single-request latency measurements
        # Batch processing throughput
        # Concurrent request handling
        # Scaling characteristics
```

**Profiling Metrics**:
- **Memory Efficiency**: MB per model parameter, peak usage
- **Inference Speed**: Tokens/sec, requests/sec, latency distribution
- **Resource Utilization**: CPU%, GPU%, memory bandwidth
- **Scalability**: Performance degradation under load

### Phase 6: Statistical Analysis and Visualization
**Script**: `results_analyzer.py`

**Analysis Components**:
```python
def analyze_configuration_impact():
    # Statistical significance testing
    # Configuration parameter importance ranking
    # Performance trade-off analysis
    # Best configuration identification per use case

def generate_visualization_dashboard():
    # Performance heatmaps by configuration
    # Accuracy comparison charts
    # Resource usage profiles
    # Clinical appropriateness scoring
```

**Output Deliverables**:
- **Performance Comparison Matrix**: All configs ranked by multiple criteria
- **Best Configuration Recommendations**: Optimized for different use cases
- **Trade-off Analysis**: Performance vs accuracy vs resource usage
- **Clinical Safety Report**: Risk assessment of different configurations

## Execution Strategy

### Parallel Processing Architecture
```python
# Utilize multiple CPU cores for concurrent evaluation
# GPU memory management for model loading
# Batch processing optimization for throughput
# Progressive result saving for resume capability
```

### Resume and Progress Tracking
```python
class EvaluationProgress:
    def __init__(self):
        self.completed_combinations = set()
        self.failed_combinations = set()
        self.progress_file = "evaluation_progress.json"
    
    def should_skip_combination(self, config_combo):
        return config_combo in self.completed_combinations
```

### Resource Management
- **Memory Management**: Efficient model loading/unloading
- **Batch Size Optimization**: Balance throughput vs memory usage
- **Error Handling**: Robust recovery from individual evaluation failures
- **Progress Persistence**: Resume capability for long-running evaluations

## Expected Outcomes

### Performance Insights
- **Optimal RAG Configurations**: Best chunking, retrieval, and ranking strategies
- **Model-Adapter Performance**: Effectiveness of different fine-tuning approaches
- **Resource Requirements**: Memory and compute needs for production deployment
- **Speed-Accuracy Trade-offs**: Performance characteristics across configurations

### Clinical Validation
- **Safety Assurance**: Identification of dangerous misclassification patterns
- **Appropriateness Validation**: Clinical quality of automated triage decisions
- **Confidence Calibration**: Uncertainty quantification effectiveness
- **Edge Case Handling**: Performance on rare or complex medical scenarios

### Production Readiness
- **Deployment Configuration**: Optimal settings for production environment
- **Performance Benchmarks**: Expected throughput and latency characteristics  
- **Resource Planning**: Hardware requirements for target load
- **Quality Assurance**: Comprehensive testing coverage and validation

## Timeline Estimate

**Phase 1-2**: Configuration generation and pipeline setup (2-3 days)
**Phase 3-4**: Metrics collection and LLM judge implementation (3-4 days)  
**Phase 5-6**: Performance profiling and analysis (2-3 days)
**Execution**: Full evaluation run across all combinations (1-2 weeks)
**Analysis**: Results interpretation and report generation (2-3 days)

**Total Estimated Duration**: 3-4 weeks for complete implementation and evaluation

## Technical Requirements

- **Compute Resources**: High-memory system for concurrent model hosting
- **Storage**: Significant disk space for results database and intermediate files
- **MLX Framework**: Optimized inference for Apple Silicon architecture
- **Dependencies**: Statistical analysis libraries, visualization tools, LLM APIs
- **Resume Capability**: Robust progress tracking for long-running evaluations