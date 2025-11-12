# Medical Triage System - Enhanced Evaluation Framework

**Streamlined evaluation pipeline with validated RAG configurations and TinfoilAgent clinical assessment.**

## 🎯 **Quick Start**

```bash
# Complete evaluation with clinical assessment
python run_evaluation.py

# Quick test (50 combinations)
python run_evaluation.py --max-combinations 50

# Skip clinical evaluation (faster)
python run_evaluation.py --skip-clinical

# Test TinfoilAgent integration
python test_tinfoil_integration.py
```

## 📁 **Core Components**

### **Essential Scripts**
- `run_evaluation.py` - Main entry point
- `optimized_master_runner.py` - Complete pipeline orchestrator
- `enhanced_evaluation_pipeline.py` - Core evaluation engine with TinfoilAgent
- `optimized_config_generator.py` - Uses validated RAG configs (600 combinations)
- `integrated_rag_system.py` - Interfaces with existing RAG implementations
- `appropriateness_judge.py` - TinfoilAgent clinical evaluation

### **Testing & Documentation**
- `test_tinfoil_integration.py` - Test TinfoilAgent integration
- `WORKFLOW_EXPLANATION.md` - Detailed workflow documentation
- `TINFOIL_INTEGRATION.md` - TinfoilAgent integration guide

## ⚡ **Key Optimizations**

### **Validated RAG Configurations**
- Uses empirically tested RAG configs from `top_pass_5_performers.csv`
- **600 combinations** instead of 14,580 (96% reduction)
- **Hours instead of weeks** for evaluation

### **Real Clinical Validation**
- **TinfoilAgent integration** using your existing API setup
- **4-dimension scoring**: Safety (40%), Efficiency (20%), Completeness (20%), Reasoning (20%)
- **Clinical rationales** for each evaluation

### **Integrated RAG System**
- **Pure RAG**: Semantic-only retrieval
- **Contextual RAG**: Hybrid semantic + BM25 fusion
- Uses your existing `HybridRetrievalSystem` from `final_retrieval_testing`

## 🔧 **Environment Setup**

```bash
# Required environment variables
export TINFOIL_API_KEY=your_api_key_here  
export TINFOIL_ENDPOINT=your_endpoint_url_here

# Install dependencies
pip install -r requirements.txt
```

## 📊 **Evaluation Flow**

1. **Configuration Generation**: Load validated RAG configs (10) × adapters (60) = 600 combinations
2. **RAG Integration**: Use existing `pure_rag` and `contextual_rag` retrieval methods  
3. **Model Evaluation**: Test each RAG+adapter combination on medical cases
4. **Clinical Assessment**: TinfoilAgent evaluates clinical appropriateness of decisions
5. **Results Analysis**: Comprehensive performance and safety metrics

## 🎯 **Expected Results**

```json
{
  "evaluation_summary": {
    "total_combinations": 600,
    "completion_time": "2-3 hours", 
    "clinical_evaluations": 600,
    "top_performers": [
      {
        "config_id": "S360_1_structured_agent_tinfoil_medical_contextual_rag",
        "triage_accuracy": 0.847,
        "clinical_appropriateness": 8.3,
        "rag_pass_at_5": 0.595
      }
    ]
  }
}
```

## 📈 **Benefits**

- **Practical Timeline**: Hours instead of 4-6 weeks
- **Real Clinical Validation**: TinfoilAgent LLM-as-judge evaluation
- **Proven RAG Performance**: Uses empirically validated configurations
- **Complete Integration**: Works with your existing RAG and TinfoilAgent systems
- **Production Ready**: Comprehensive evaluation with safety metrics

## 🔍 **Troubleshooting**

**TinfoilAgent Issues:**
```bash
# Test integration
python test_tinfoil_integration.py

# Check environment
echo $TINFOIL_API_KEY | wc -c  # Should show key length
echo $TINFOIL_ENDPOINT          # Should show your endpoint
```

**Fallback Mode:**
If TinfoilAgent unavailable, system automatically uses mock clinical evaluation while maintaining all other functionality.

This streamlined framework provides comprehensive medical triage system evaluation with real clinical validation in a practical timeframe.

---

## 📋 **Detailed Framework Components (Reference)**

### 1. **Optimized Configuration Generator** (`optimized_config_generator.py`)
**Purpose**: Creates evaluation matrix using your empirically validated RAG configurations.

**Key Functions**:
- **Validated RAG Configs**: Uses top 10 performers from `top_pass_5_performers.csv`
  - structured_agent_tinfoil_medical + contextual_rag (Pass@5: 0.595)
  - structured_agent_tinfoil_medical + pure_rag (Pass@5: 0.590)
  - contextual_sentence_c1024_o2_tinfoil + contextual_rag (Pass@5: 0.525)
  - And 7 more validated configurations

- **Adapter Discovery**: Automatically scans your trained adapters
  - Standard adapters: 36 configurations (SmolLM2-360M, SmolLM2-135M, Gemma-270M)
  - Safety-enhanced adapters: 24 configurations with cost-sensitive training

- **Optimized Combinations**: 10 validated RAG configs × 60 adapters = **600 total combinations**

**Output**: Optimized evaluation matrix with performance metadata and 96% reduction in evaluation space

### 2. **Integrated Evaluation Pipeline** (`evaluation_pipeline.py`)
**Purpose**: Core evaluation engine using your actual RAG system implementations.

**Key Features**:
- **Real RAG Integration**: Uses your `HybridRetrievalSystem` with `pure_rag` and `contextual_rag`
- **Resume Capability**: Detects completed evaluations to avoid duplication
- **Batch Processing**: Memory-efficient processing with configurable batch sizes
- **Validated Configs**: Processes your empirically proven configurations
- **Progress Tracking**: Real-time progress monitoring and results persistence

**Enhanced Evaluation Process**:
1. Load model + adapter combination
2. Initialize `IntegratedRAGSystemWrapper` with validated configuration  
3. Use your actual retrieval methods (contextual_rag/pure_rag)
4. Process test dataset through complete integrated system
5. Extract triage decisions using real RAG context
6. Measure performance metrics integrated with RAG performance data
7. Save results with full configuration and empirical metadata

### 3. **Metrics Collection System** (`metrics_collector.py`)
**Purpose**: Comprehensive medical-specific evaluation metrics.

**Medical Accuracy Metrics**:
- **Standard Classification**: Precision, recall, F1-score per class (ED/GP/HOME)
- **F2-Score**: Recall-weighted metric emphasizing sensitivity for medical safety
- **Cost-Sensitive Accuracy**: Weighted by clinical cost matrix
  - ED→HOME misclassification: 100× penalty (extremely dangerous)
  - ED→GP misclassification: 100× penalty (very dangerous)
  - GP→HOME misclassification: 10× penalty (moderately dangerous)

**Safety Metrics**:
- **False Negative Rates**: Critical for emergency cases
- **Under-Triage Rate**: Dangerous classifications (ED→GP, ED→HOME)
- **Over-Triage Rate**: Conservative but safe classifications  
- **Safety Score**: Composite metric penalizing dangerous errors

**Performance Metrics**:
- **Inference Speed**: Tokens per second, requests per second
- **Memory Efficiency**: Peak usage, memory growth patterns
- **Latency Distribution**: P50, P95, P99 response times
- **Scalability**: Performance under concurrent load

**Next Steps Evaluation**:
- **Semantic Similarity**: Comparison with reference recommendations
- **Medical Terminology**: Appropriate use of clinical language
- **Treatment Appropriateness**: Context-specific recommendation quality

### 4. **Clinical Appropriateness Judge** (`appropriateness_judge.py`)
**Purpose**: LLM-as-judge evaluation of clinical decision quality using advanced language models.

**Evaluation Dimensions** (0-10 scale):
- **Safety (40% weight)**: Risk assessment and harm prevention
  - Emergency recognition accuracy
  - Dangerous misclassification avoidance
  - Patient risk stratification quality

- **Efficiency (20% weight)**: Healthcare resource utilization
  - Appropriate care level assignment
  - Cost-effective decision making
  - System resource optimization

- **Completeness (20% weight)**: Coverage of relevant factors
  - Comprehensive symptom consideration
  - Relevant medical history integration  
  - Diagnostic thoroughness

- **Reasoning (20% weight)**: Clinical logic and decision process
  - Sound medical reasoning
  - Evidence-based conclusions
  - Appropriate clinical protocols

**Judge Model Integration**:
- Supports GPT-4, Claude, or other advanced LLMs
- Structured evaluation prompts with clinical context
- Rate limiting and error handling for API reliability
- Detailed rationale extraction for decision transparency

### 5. **Performance Profiler** (`performance_profiler.py`)
**Purpose**: Detailed system performance analysis for production readiness assessment.

**Model Loading Profiling**:
- Memory footprint during model initialization
- Loading time across different model sizes
- Peak memory usage patterns
- Adapter overhead quantification

**Inference Profiling**:
- Single request latency measurements
- Batch processing throughput analysis
- Memory usage per token generated
- CPU/GPU utilization monitoring

**Scalability Benchmarking**:
- Performance across different batch sizes (1, 5, 10, 20)
- Concurrent request handling capacity
- Memory scaling characteristics
- Throughput degradation analysis

**System Monitoring**:
- Real-time resource usage tracking
- Memory leak detection
- Performance regression identification
- Production deployment readiness metrics

### 6. **Results Analyzer** (`results_analyzer.py`)
**Purpose**: Statistical analysis and visualization of evaluation results.

**Statistical Analysis**:
- **Configuration Impact Analysis**: ANOVA testing for parameter significance
- **Performance Distribution**: Statistical summaries across all metrics
- **Trade-off Analysis**: Correlation analysis between competing objectives
- **Significance Testing**: Statistical validation of performance differences

**Best Configuration Identification**:
- **Multi-Objective Optimization**: Composite scoring across all dimensions
- **Pareto Frontier Analysis**: Trade-off boundary identification
- **Use-Case Optimization**: Best configs for specific scenarios (speed vs accuracy)
- **Production Recommendations**: Deployment-ready configuration suggestions

**Visualization Suite**:
- **Performance Heatmaps**: Configuration performance across all metrics
- **Interactive Dashboards**: Multi-dimensional exploration tools
- **Trade-off Plots**: Speed vs accuracy, safety vs efficiency
- **Distribution Analysis**: Performance variance across configurations

**Reporting**:
- **Executive Summary**: High-level findings for stakeholders
- **Technical Deep-dive**: Detailed analysis for development teams
- **Configuration Comparison**: Side-by-side performance analysis
- **Deployment Guide**: Production configuration recommendations

### 7. **Master Evaluation Runner** (`master_evaluation_runner.py`)
**Purpose**: Orchestrates the complete evaluation pipeline with session management.

**Pipeline Orchestration**:
- **Phase Management**: Configurable execution of evaluation phases
- **Session Tracking**: Timestamped execution with full audit trail
- **Error Recovery**: Graceful handling of phase failures
- **Resource Management**: Memory cleanup between evaluation phases

**Execution Phases**:
1. **Configuration Generation**: Create evaluation matrix
2. **Evaluation Pipeline**: Run systematic testing  
3. **Clinical Assessment**: LLM-as-judge appropriateness evaluation
4. **Performance Profiling**: System benchmarking and optimization
5. **Statistical Analysis**: Results analysis and visualization

**Session Management**:
- **Unique Session IDs**: Timestamped execution tracking
- **Organized Output**: Structured directory hierarchy
- **Progress Persistence**: Resume capability across sessions
- **Audit Trail**: Complete execution logging

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Required Python packages
pip install mlx-lm numpy pandas scikit-learn matplotlib seaborn plotly psutil

# Ensure your assets are in place:
# - /Users/choemanseung/789/hft/triage_adapters/ (36 standard adapters)
# - /Users/choemanseung/789/hft/safety_triage_adapters/ (24 safety adapters)  
# - /Users/choemanseung/789/hft/final_retrieval_testing/analysis_output/top_pass_5_performers.csv
```

### Basic Usage
```bash
cd /Users/choemanseung/789/hft/evaluation_framework

# Test integration with your RAG system
python test_integration.py

# Run optimized evaluation pipeline (600 combinations)
python optimized_master_runner.py

# Quick test with subset 
python optimized_master_runner.py --max-combinations 50

# Skip clinical evaluation if no LLM API access
python optimized_master_runner.py --skip-clinical

# Custom configuration
python optimized_master_runner.py --batch-size 4 --max-workers 1 --output-dir ./my_evaluation
```

### Advanced Configuration
```python
# Custom configuration file (config.json)
{
  "phases": {
    "generate_matrix": true,
    "run_evaluation": true,
    "clinical_judgement": false,  # Requires OpenAI API
    "performance_profiling": true,
    "statistical_analysis": true
  },
  "evaluation": {
    "batch_size": 8,
    "max_workers": 2,
    "resume": true,
    "max_combinations": 100  # Limit for faster testing
  },
  "clinical_judge": {
    "model": "gpt-4-turbo",
    "rate_limit_delay": 1.0
  }
}

# Run with custom config
python master_evaluation_runner.py --config-file config.json
```

## 📊 Expected Results Structure

### Session Directory Structure
```
evaluation_session_20241226_143022/
├── configurations/
│   └── evaluation_matrix.json          # Complete configuration matrix
├── results/
│   ├── evaluation_results.json         # Raw evaluation results
│   └── clinical_appropriateness.json   # LLM judge results
├── profiles/
│   └── performance_profiles.json       # System performance data
├── analysis/
│   └── comprehensive_analysis_report.json # Statistical analysis
├── visualizations/
│   ├── performance_heatmap.png         # Configuration performance heatmap
│   └── interactive_dashboard.html      # Interactive exploration dashboard
└── execution_log.json                  # Complete audit trail
```

### Key Metrics in Results
```json
{
  "combo_id": "S360_fixden3_a1b2c3d4",
  "triage_accuracy": 0.847,
  "safety_score": 0.923,
  "clinical_appropriateness": 8.2,
  "inference_speed_tps": 45.3,
  "memory_usage_mb": 1247.8,
  "latency_ms": 127.4,
  "detailed_metrics": {
    "precision_ed": 0.891,
    "recall_ed": 0.934,
    "f2_score": 0.865,
    "false_negative_ed_rate": 0.066,
    "cost_sensitive_accuracy": 0.912
  }
}
```

## ⏱️ Execution Timeline

### Phase Duration Estimates
- **Configuration Generation**: 2-5 minutes (loads validated configs)
- **Evaluation Pipeline**: 
  - **Optimized evaluation (600 combinations): 1-2 days**
  - Quick test (50 combinations): 4-6 hours  
  - Original space (14,580 combinations): 4-6 weeks *(avoided through optimization)*
- **Clinical Appropriateness**: 2-4 hours (API dependent, now feasible with smaller set)
- **Performance Profiling**: 1-2 hours
- **Statistical Analysis**: 10-30 minutes

### Resource Requirements
- **Memory**: 4-8GB RAM (depending on model size)
- **Storage**: 5-10GB for complete results
- **Compute**: Multi-core CPU recommended, GPU optional
- **Network**: Required for LLM-as-judge evaluation

## 🔧 Customization Options

### Evaluation Scope
```python
# Modify RAG parameter space in config_matrix_generator.py
self.rag_params = {
    "chunking_methods": ["fixed_size", "semantic"],  # Reduce combinations
    "retrieval_methods": ["dense", "hybrid"],
    "top_k_values": [3, 5],
    "chunk_sizes": [512],  # Focus on single chunk size
    "overlap_ratios": [0.2]
}
```

### Metrics Customization
```python
# Adjust medical cost matrix in metrics_collector.py
self.cost_matrix = {
    ("ED", "GP"): 150.0,    # Increase penalty
    ("ED", "HOME"): 200.0,  # Increase penalty
    # ... other combinations
}
```

### Judge Model Configuration
```python
# Switch LLM judge model in appropriateness_judge.py
judge = ClinicalAppropriatenessJudge(
    judge_model="claude-3-opus",  # or "gpt-4-turbo"
    rate_limit_delay=2.0
)
```

## 📈 Expected Outcomes

### Performance Insights
- **Optimal RAG Configurations**: Best chunking, retrieval, and ranking strategies
- **Model-Adapter Effectiveness**: Performance comparison across all fine-tuned models
- **Resource Requirements**: Memory and compute needs for production deployment
- **Speed-Accuracy Trade-offs**: Performance characteristics for different use cases

### Clinical Validation
- **Safety Assurance**: Identification of dangerous misclassification patterns
- **Appropriateness Validation**: Clinical quality of automated triage decisions
- **Confidence Calibration**: Uncertainty quantification effectiveness
- **Edge Case Analysis**: Performance on rare or complex medical scenarios

### Production Readiness
- **Deployment Configuration**: Optimal settings for production environment
- **Performance Benchmarks**: Expected throughput and latency characteristics
- **Scaling Guidelines**: Hardware requirements for different load scenarios
- **Quality Assurance**: Comprehensive testing coverage and validation metrics

## 🛠️ Troubleshooting

### Common Issues
1. **MLX Import Errors**: Ensure MLX is properly installed for your platform
2. **Memory Constraints**: Reduce batch size or limit combinations
3. **API Rate Limits**: Increase delays in clinical appropriateness evaluation
4. **Resume Failures**: Check file permissions in output directories

### Performance Optimization
- **Parallel Processing**: Increase max_workers for faster evaluation
- **Memory Management**: Monitor memory usage and adjust batch sizes
- **Storage Optimization**: Use SSD storage for better I/O performance
- **Network Optimization**: Consider local LLM deployment for judge evaluation

## 📚 Technical Documentation

### Dependencies
- **MLX Framework**: Apple Silicon optimized inference
- **Scientific Computing**: NumPy, Pandas, SciPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **System Monitoring**: psutil for resource tracking
- **Data Processing**: JSON handling, file I/O optimization

### Architecture Patterns
- **Modular Design**: Independent components with clear interfaces
- **Configuration Management**: Centralized parameter management
- **Error Handling**: Comprehensive exception handling and recovery
- **Progress Persistence**: Resume capability across long-running operations
- **Resource Management**: Memory-efficient processing with cleanup

This comprehensive evaluation framework provides the foundation for systematic optimization of your medical triage system, ensuring optimal performance across accuracy, safety, and efficiency dimensions for clinical deployment.