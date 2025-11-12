# Retrieval Performance Testing System

This system evaluates the retrieval performance of different chunking methods and retrieval techniques for medical triage dialogues.

## Overview

The testing system systematically compares:
- **Chunking Methods**: Fixed size, sentence-based, paragraph-based, and contextual chunking
- **Retrieval Techniques**: Semantic search, BM25, hybrid search, and contextual retrieval
- **Evaluation Metrics**: Pass@5, Pass@10, Pass@20 accuracy rates

## Quick Start

1. **Create test data and run a quick test:**
   ```bash
   python run_retrieval_test.py
   ```
   Choose option 1 for a quick test with 5 configurations.

2. **Run full comprehensive test:**
   ```bash
   python retrieval_performance_tester.py --test-data eval/test_data.json --results-dir results/full_test
   ```

## Files Structure

```
├── retrieval_performance_tester.py  # Main testing script
├── create_sample_test_data.py       # Creates sample test data
├── run_retrieval_test.py            # Convenience runner script  
├── offline_contextual_retrieval.py  # Base retrieval classes
├── eval/
│   └── test_data.json              # Test cases (created automatically)
├── results/                        # Test results output
└── RAGdatav4/
    ├── embeddings/                 # Embedding files and indices
    └── *.json                      # Chunk data files
```

## Test Data Format

Each test case follows this structure:
```json
{
  "symptom": "Abdominal_aortic_aneurysm (Case 1)",
  "patient_query": "Hi, I'm having severe belly pain...",
  "patient_response": "The pain is getting worse...",
  "final_triage_decision": "ED",
  "next_step": "Call 111 immediately...",
  "reasoning_question": "...",
  "reasoning_decision": "...",
  "generation_timestamp": 1758429942.574643
}
```

## Evaluation Logic

**Success Criteria:** A retrieval is considered successful if:
- The retrieved chunk's `source_document` field contains the symptom name
- Example: symptom "Abdominal_aortic_aneurysm" should retrieve chunks from "Abdominal_aortic_aneurysm.txt"

**Query Construction:** 
- Combines `patient_query` + `patient_response` for retrieval

## Available Chunking Methods

Based on your RAGdatav4/embeddings directory:

### Fixed Size Chunking
- `fixed_c512_o0`: 512 characters, no overlap
- `fixed_c768_o100`: 768 characters, 100 character overlap  
- `fixed_c1024_o150`: 1024 characters, 150 character overlap

### Sentence-Based Chunking
- `sentence_t384_o2`: 384 tokens, 2 sentence overlap
- `sentence_t512_o3`: 512 tokens, 3 sentence overlap
- `sentence_t768_o1`: 768 tokens, 1 sentence overlap

### Paragraph-Based Chunking
- `paragraph_m25`: Minimum 25 tokens per paragraph
- `paragraph_m50`: Minimum 50 tokens per paragraph

### Contextual Chunking
- `contextual_sentence_c1024_o2`: Sentence chunking with contextual information

## Retrieval Methods

For each chunking method, the system tests:

1. **Semantic Search**: Pure embedding similarity
2. **BM25**: Keyword-based search
3. **Hybrid**: Combines semantic + BM25 with Reciprocal Rank Fusion
4. **Contextual**: Enhanced retrieval using contextual information (where available)

## Results Output

### Summary Results (`summary_TIMESTAMP.json`)
```json
{
  "config_name": "fixed_c512_o0_semantic",
  "chunking_method": "fixed_c512_o0", 
  "retrieval_type": "semantic",
  "pass_at_5": 85.2,
  "pass_at_10": 91.4,
  "pass_at_20": 94.1,
  "avg_retrieval_time_ms": 12.3
}
```

### Console Output
```
Rank Configuration               Pass@5   Pass@10  Pass@20  Avg Time (ms)
1    contextual_sentence_hybrid  92.1%    96.3%    98.2%    45.2
2    fixed_c768_o100_semantic    89.4%    94.1%    96.8%    15.7
3    sentence_t512_o3_hybrid     87.3%    92.9%    95.4%    38.1
...
```

## Advanced Usage

### Custom Test Data
Replace `eval/test_data.json` with your own test cases following the same format.

### Filtering Configurations
```bash
# Test only contextual methods
python retrieval_performance_tester.py --test-data eval/test_data.json --config-filter contextual

# Limit number of configurations
python retrieval_performance_tester.py --test-data eval/test_data.json --max-configs 10
```

### Custom Results Directory
```bash
python retrieval_performance_tester.py --test-data eval/test_data.json --results-dir my_results/
```

## Dependencies

The system requires:
- `numpy`, `faiss-cpu`, `pickle`
- `sentence-transformers`
- `rank-bm25`, `scikit-learn`
- `tqdm` for progress bars

Install dependencies:
```bash
pip install numpy faiss-cpu sentence-transformers rank-bm25 scikit-learn tqdm
```

## Troubleshooting

**Missing embedding files**: Ensure your RAGdatav4/embeddings directory contains the required `.pkl` and `.index` files.

**Missing chunk data**: Verify that corresponding chunk JSON files exist in RAGdatav4/.

**Import errors**: Make sure `offline_contextual_retrieval.py` is in the same directory.

**Memory issues**: Use `--max-configs` to limit the number of configurations tested simultaneously.

## Interpreting Results

- **Pass@5/10/20**: Percentage of queries where the correct document appears in top 5/10/20 results
- **High Pass@5, low Pass@20**: Method is precise but may miss relevant documents
- **Low Pass@5, high Pass@20**: Method retrieves relevant documents but not in top positions
- **Retrieval Time**: Average time per query (lower is better for production use)

## Expected Performance

Based on similar medical retrieval systems:
- **Good performance**: Pass@10 > 85%
- **Excellent performance**: Pass@10 > 90%  
- **Production ready**: Pass@10 > 92% with <50ms average retrieval time

The best performing methods typically combine:
- Contextual chunking for better context preservation
- Hybrid retrieval for balanced precision/recall
- Appropriate chunk sizes (512-1024 characters for medical content)