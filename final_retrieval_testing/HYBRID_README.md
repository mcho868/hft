# Hybrid Retrieval Evaluation System

This system combines the best of both approaches from `retrieval_testv2.py` and `final_retrieval_testing`:

1. **Multi-source bias configurations** (like retrieval_testv2.py)
2. **Pure RAG vs Contextual retrieval evaluation** (like final_retrieval_testing)  
3. **Comprehensive performance metrics** and analysis

## Key Features

### 🎯 Bias Configurations
- **Balanced**: 4:4:4 (equal representation)
- **Healthify Bias**: 6:2:2, 5:3:2 (favor Healthify)
- **Mayo Bias**: 2:6:2 (favor Mayo Clinic)
- **NHS Bias**: 2:2:6 (favor NHS)
- **Medical Authority**: 3:4:3 (Mayo-focused)

### 🧠 Retrieval Types
- **Pure RAG**: Uses original text only
- **Contextual RAG**: Uses contextual information when available

### 📊 Evaluation Metrics
- **Pass@5 (PRIMARY)**: Percentage of correct retrievals in top 5 - optimized for small models
- **Pass@10, Pass@20**: Additional metrics for broader analysis
- **Source Distribution**: Actual vs expected source distribution
- **Retrieval Time**: Performance benchmarking
- **Comprehensive Analysis**: By bias, retrieval type, and chunking method

## Quick Start

### 1. Run Interactive Menu
```bash
python run_hybrid_test.py
```

### 2. Direct Command Line
```bash
# Quick test
python hybrid_retrieval_evaluator.py --test-data eval/test_data.json --results-dir results/hybrid_quick --max-configs 10

# Bias comparison (using one chunking method)
python hybrid_retrieval_evaluator.py --test-data eval/test_data.json --results-dir results/bias_test --chunking-filter fixed_c1024_o150

# RAG type comparison (using balanced bias)
python hybrid_retrieval_evaluator.py --test-data eval/test_data.json --results-dir results/rag_test --bias-filter balanced
```

## Test Types Available

### 1. 🚀 Quick Test
- Tests 10 configurations
- Good for rapid evaluation
- Includes mix of bias and retrieval types

### 2. ⚖️ Bias Configuration Comparison  
- Tests all bias configurations
- Uses single chunking method for fair comparison
- Shows impact of source distribution

### 3. 🧠 Pure RAG vs Contextual RAG
- Compares retrieval types
- Uses balanced bias for fair comparison
- Shows contextual information benefit

### 4. 🎯 Contextual-Focused Test
- Tests only contextual chunking methods
- Evaluates contextual retrieval performance
- Good for contextual-specific analysis

### 5. 🔍 Full Test
- Tests all available configurations
- Comprehensive but time-consuming
- Complete analysis across all dimensions

## Configuration Files

### Data Structure Required
```
RAGdatav4/
├── healthify_chunks_{method}.json
├── mayo_chunks_{method}.json  
├── nhs_chunks_{method}.json
└── embeddings/
    ├── healthify_vector_db_{method}.index
    ├── mayo_vector_db_{method}.index
    └── nhs_vector_db_{method}.index
```

### Test Data Format
```json
[
  {
    "symptom": "Condition_name (Case N)",
    "patient_query": "Patient's initial query",
    "patient_response": "Patient's response to clarifying question",
    "final_triage_decision": "ED/GP/Self-care",
    "next_step": "Recommended action",
    ...
  }
]
```

## Results Structure

### Files Generated
```
results/hybrid_test_TIMESTAMP/
├── hybrid_summary_TIMESTAMP.json          # Quantitative results
├── hybrid_detailed_report_TIMESTAMP.txt   # Comprehensive analysis
└── (detailed configs for top performers)
```

### Summary Report Sections
1. **Overall Performance Rankings**: Best configurations
2. **Retrieval Type Analysis**: Pure RAG vs Contextual RAG
3. **Bias Configuration Analysis**: Impact of source distribution  
4. **Chunking Method Analysis**: Performance by chunking strategy

## Key Differences from Original Systems

| Feature | Original retrieval_testv2.py | Original final_retrieval_testing | Hybrid System |
|---------|------------------------------|----------------------------------|---------------|
| **Source Control** | ✅ Multi-source bias | ❌ Combined only | ✅ Multi-source bias |
| **Retrieval Types** | ❌ Single approach | ✅ Multiple types | ✅ Multiple types |
| **Evaluation** | Hit@K, Recall@K | Pass@K | Pass@K + source analysis |
| **Architecture** | Single script | Modular | Modular with bias control |

## Customization

### Modify Bias Configurations
Edit `hybrid_config.json`:
```json
{
  "name": "custom_bias",
  "healthify": 6,
  "mayo": 2, 
  "nhs": 2,
  "description": "Custom 6:2:2 configuration"
}
```

### Add New Retrieval Types
Extend `_process_chunk_by_type()` in `hybrid_retrieval_evaluator.py`

### Filter Configurations
```bash
# Test only specific chunking methods
--chunking-filter contextual

# Test only specific bias configurations  
--bias-filter healthify_bias

# Limit number of configurations
--max-configs 15
```

## Expected Results

### Bias Configuration Impact
- **Healthify bias** should improve performance for general health queries
- **Mayo bias** should excel for complex medical conditions
- **NHS bias** should perform well for UK-specific medical guidance
- **Balanced** provides good overall coverage

### RAG Type Comparison
- **Contextual RAG** should outperform Pure RAG when contextual information is available
- **Pure RAG** may be more consistent across different chunk types
- Performance difference depends on quality of contextual information

## Performance Tips

1. **Start with Quick Test** to validate setup
2. **Use Bias Comparison** to find optimal source distribution
3. **Use RAG Comparison** to validate contextual benefits
4. **Run Full Test** only when you have validated the setup

## Troubleshooting

### Common Issues
1. **Missing index files**: Ensure all source indices exist for each chunking method
2. **Chunk count mismatch**: Verify chunk files have same structure across sources
3. **Memory issues**: Reduce max_configs for large-scale testing
4. **No results**: Check test data format and file paths

### Debugging
```bash
# Enable verbose logging
export PYTHONPATH=.
python -v hybrid_retrieval_evaluator.py --test-data eval/test_data.json --max-configs 2
```

This hybrid system gives you the best of both worlds: precise control over source diversity while comprehensively evaluating different retrieval approaches.