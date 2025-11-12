# Tinfoil REST API Setup Guide

## 1. Set Your API Key

```bash
export TINFOIL_API_KEY="your_actual_api_key_here"
```

## 2. Test the API Connection

```bash
python3 test_tinfoil_rest.py
```

Expected output:
```
🔑 API key found: sk_xxxxx...
✅ Initialized Tinfoil REST client with model: llama3-3-70b
🧪 Testing Tinfoil REST API...
✅ Response received: REST API working
```

## 3. Run Full Evaluation with LLM-as-Judge

### Quick Test (3 combinations)
```bash
python3 evaluation_framework/run_evaluation.py --max-combinations 3 --enable-clinical
```

### Medium Test (50 combinations)  
```bash
python3 evaluation_framework/run_evaluation.py --max-combinations 50 --enable-clinical
```

### Full Evaluation (600 combinations)
```bash
python3 evaluation_framework/run_evaluation.py --enable-clinical
```

## 4. What to Expect

- **Without `--enable-clinical`**: Only triage accuracy calculated (fast)
- **With `--enable-clinical`**: LLM-as-judge evaluates clinical appropriateness (slower, more comprehensive)

### Clinical Evaluation Includes:
- **Safety Score (0-10)**: Patient harm prevention  
- **Efficiency Score (0-10)**: Resource utilization
- **Completeness Score (0-10)**: Factor coverage
- **Reasoning Score (0-10)**: Clinical logic quality
- **Overall Score**: Weighted average (Safety=40%, others=20% each)

## 5. Monitoring Progress

The evaluation will show:
```
🤖 Running clinical appropriateness evaluation on 10 cases...
   Case G270_1_12345_10554: 8.5/10
   Case G270_1_12345_10555: 7.2/10
   ...
```

## 6. Results Location

Results saved to: `/Users/choemanseung/789/hft/optimized_evaluation_session_YYYYMMDD_HHMMSS/`
- `results/enhanced_evaluation_results.json` - Detailed scores per configuration
- `analysis/enhanced_analysis_report.json` - Summary statistics and top performers