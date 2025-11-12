# 🔬 Retrieval Performance Testing System

A comprehensive system for evaluating the performance of different chunking methods and retrieval techniques for medical triage document retrieval.

## 📁 Directory Structure

```
final_retrieval_testing/
├── main.py                           # 🎯 Main entry point
├── retrieval_performance_tester.py   # 🔬 Core testing engine  
├── visualize_results.py              # 📊 Results visualization
├── create_sample_test_data.py        # 📝 Test data generator
├── run_retrieval_test.py             # 🚀 Simple runner interface
├── setup_test_environment.py         # ⚙️ Environment setup
├── config.py                         # ⚙️ Configuration settings
├── utils.py                          # 🛠️ Utility functions
├── requirements.txt                  # 📦 Dependencies
├── README.md                         # 📖 This file
├── eval/                            # 📝 Test data directory
├── results/                         # 📊 Test results output
└── visualizations/                  # 📈 Generated charts
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd final_retrieval_testing
python main.py --setup
```

### 2. Run Quick Test (Recommended)
```bash
python main.py --quick
```

### 3. Interactive Menu
```bash
python main.py
```

## 📋 Available Operations

### Command Line Usage
```bash
# Setup environment
python main.py --setup

# Quick test (5 configurations)
python main.py --quick

# Full test (all configurations)  
python main.py --full

# Contextual methods only
python main.py --contextual

# Complete pipeline
python main.py --pipeline

# Visualize existing results
python main.py --visualize results/quick_test
```

### Interactive Menu
Run `python main.py` for an interactive menu with these options:

1. 🔧 **Setup Environment** - Check dependencies and create directories
2. 📝 **Create Test Data** - Generate sample medical triage test cases
3. 🚀 **Quick Test** - Test 5 configurations (recommended for first run)
4. 🔍 **Full Test** - Test all available configurations
5. 🧠 **Contextual Only** - Test only contextual retrieval methods
6. 📊 **Visualize Results** - Generate charts and analysis
7. 🎯 **Complete Pipeline** - Run setup → test → visualize
8. ❌ **Exit**

## 🔧 System Requirements

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Required Files (in parent directory)
- `offline_contextual_retrieval.py` - Base retrieval classes
- `RAGdatav4/` - Chunk data directory
- `RAGdatav4/embeddings/` - Embedding files and indices

## 📊 How It Works

### 1. Configuration Detection
The system automatically scans `RAGdatav4/embeddings/` and identifies:
- **Fixed chunking**: `fixed_c512_o0`, `fixed_c768_o100`, etc.
- **Sentence chunking**: `sentence_t384_o2`, `sentence_t512_o3`, etc.
- **Paragraph chunking**: `paragraph_m25`, `paragraph_m50`, etc.
- **Contextual chunking**: `contextual_sentence_c1024_o2`, etc.

### 2. Retrieval Methods
For each chunking method, tests:
- **Semantic**: Pure embedding similarity
- **BM25**: Keyword-based search
- **Hybrid**: Semantic + BM25 with Reciprocal Rank Fusion
- **Contextual**: Enhanced retrieval (for contextual chunks)

### 3. Evaluation Metrics
- **Pass@5/10/20**: Percentage of queries where correct document appears in top 5/10/20 results
- **Retrieval Time**: Average time per query
- **Success Rate**: Overall accuracy across all test cases

### 4. Medical Triage Evaluation
Success criteria: Retrieved chunk's `source_document` contains the symptom name
- Example: symptom "Abdominal_aortic_aneurysm" → should retrieve from "Abdominal_aortic_aneurysm.txt"

## 📈 Results and Visualization

### Output Files
- `results/summary_TIMESTAMP.json` - Performance metrics for all configurations
- `results/detailed_CONFIG_TIMESTAMP.json` - Detailed results for top performers
- `visualizations/overview_dashboard_TIMESTAMP.png` - Performance comparison charts
- `visualizations/detailed_analysis_TIMESTAMP.png` - In-depth analysis plots

### Performance Dashboard
- **Top 10 configurations** ranked by Pass@10
- **Performance by chunking method** with error bars
- **Performance by retrieval type** comparison
- **Speed vs accuracy tradeoff** scatter plot
- **Chunk size impact** analysis
- **Pass@K comparison** for top methods

### Recommendations
The system automatically identifies:
- 🏆 **Best overall configuration**
- 🚀 **Best for production** (balanced speed + accuracy)
- 📊 **Optimal chunking method**
- 🔍 **Best retrieval type**
- 📏 **Optimal chunk size**

## 🎯 Expected Performance

### Performance Tiers
- **Good**: Pass@10 > 85%
- **Excellent**: Pass@10 > 90%
- **Production Ready**: Pass@10 > 92% with <50ms retrieval time

### Typical Results
```
Rank Configuration                    Pass@5   Pass@10  Pass@20  Time(ms)
1    contextual_sentence_hybrid      92.1%    96.3%    98.2%    45.2
2    fixed_c768_o100_semantic        89.4%    94.1%    96.8%    15.7
3    sentence_t512_o3_hybrid         87.3%    92.9%    95.4%    38.1
```

## 🔧 Customization

### Custom Test Data
Replace `eval/test_data.json` with your medical triage cases following this format:
```json
{
  "symptom": "Condition_name (Case 1)",
  "patient_query": "Patient's initial query...",
  "patient_response": "Patient's follow-up response...",
  "final_triage_decision": "ED",
  "next_step": "Recommended action...",
  "reasoning_question": "Triage reasoning...",
  "reasoning_decision": "Final reasoning...",
  "generation_timestamp": 1234567890
}
```

### Configuration Filtering
```bash
# Test specific configurations
python retrieval_performance_tester.py --config-filter contextual --max-configs 10
```

### Custom Results Directory
```bash
python retrieval_performance_tester.py --results-dir my_custom_results/
```

## 🐛 Troubleshooting

### Common Issues

**"offline_contextual_retrieval.py not found"**
- Ensure the file exists in the parent directory (`../offline_contextual_retrieval.py`)

**"RAGdatav4 directory not found"**
- Ensure `RAGdatav4/` exists in the parent directory
- Check that `RAGdatav4/embeddings/` contains `.pkl` and `.index` files

**"No embedding files found"**
- Verify embedding files exist in `RAGdatav4/embeddings/`
- Files should follow pattern: `embeddings_METHOD_all-MiniLM-L6-v2.pkl`

**Memory issues**
- Use `--max-configs 5` to limit configurations tested
- Test in smaller batches

**Import errors**
- Run `python setup_test_environment.py` to check dependencies
- Install missing packages: `pip install -r requirements.txt`

### Getting Help
1. Run setup: `python main.py --setup`
2. Check the logs for specific error messages
3. Verify file paths in `config.py`
4. Test with quick mode first: `python main.py --quick`

## 📝 Example Workflow

```bash
# 1. Initial setup
cd final_retrieval_testing
python main.py --setup

# 2. Quick test to verify everything works
python main.py --quick

# 3. Review results
ls results/quick_test/
ls visualizations/

# 4. Run full test if satisfied
python main.py --full

# 5. Generate final visualizations
python main.py --visualize results/full_test
```

This system provides a comprehensive evaluation framework to identify the optimal chunking and retrieval combination for your medical triage use case! 🎯