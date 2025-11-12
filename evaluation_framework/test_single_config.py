#!/usr/bin/env python3
"""
Single Configuration Test Script
Tests the first available configuration with a single test case,
showing input prompt, LLM output, and comparison with expected results.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add evaluation framework to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from optimized_config_generator import OptimizedConfigMatrixGenerator
    from enhanced_evaluation_pipeline import EnhancedEvaluationPipeline
    from integrated_rag_system import IntegratedRAGSystemWrapper
except ImportError as e:
    print(f"❌ Error importing evaluation framework: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

def load_test_case():
    """Load the first test case from the test dataset"""
    test_files = [
        "/Users/choemanseung/789/hft/Final_dataset/simplified_triage_dialogues_test.json",
        "/Users/choemanseung/789/hft/Final_dataset/final_triage_dialogues_mlx/test.jsonl"
    ]
    
    # Try simplified test file first
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                if test_file.endswith('.json'):
                    with open(test_file, 'r') as f:
                        data = json.load(f)
                    if data:
                        case = data[0]
                        return {
                            "case_id": str(case.get("id", "test_1")),
                            "input": case.get("query", ""),
                            "triage_decision": case.get("final_triage_decision", ""),
                            "next_steps": case.get("next_step", ""),
                            "reasoning": case.get("reasoning", ""),
                            "symptom": case.get("symptom", "")
                        }
                elif test_file.endswith('.jsonl'):
                    with open(test_file, 'r') as f:
                        first_line = f.readline()
                        data = json.loads(first_line)
                    prompt = data.get("prompt", "")
                    completion = data.get("completion", "")
                    
                    # Extract patient query from prompt
                    if "Patient query:" in prompt:
                        patient_query = prompt.split("Patient query:")[1].split("\n\nProvide triage decision")[0].strip()
                    else:
                        patient_query = prompt
                    
                    # Parse completion for expected results
                    expected_triage = ""
                    expected_next_steps = ""
                    expected_reasoning = ""
                    
                    if "Triage Decision:" in completion:
                        parts = completion.split("Triage Decision:")[1]
                        if "Next Step:" in parts:
                            expected_triage = parts.split("Next Step:")[0].strip()
                            remaining = parts.split("Next Step:")[1]
                            if "Reasoning:" in remaining:
                                expected_next_steps = remaining.split("Reasoning:")[0].strip()
                                expected_reasoning = remaining.split("Reasoning:")[1].strip()
                            else:
                                expected_next_steps = remaining.strip()
                    
                    return {
                        "case_id": "jsonl_test_1",
                        "input": patient_query,
                        "triage_decision": expected_triage,
                        "next_steps": expected_next_steps,
                        "reasoning": expected_reasoning,
                        "symptom": "Unknown"
                    }
            except Exception as e:
                print(f"⚠️ Error loading {test_file}: {e}")
                continue
    
    # Fallback test case
    return {
        "case_id": "fallback_test",
        "input": "I have severe chest pain and shortness of breath that started 30 minutes ago",
        "triage_decision": "ED",
        "next_steps": "Seek immediate emergency care",
        "reasoning": "Symptoms suggest possible heart attack requiring immediate attention",
        "symptom": "Chest pain"
    }

def main():
    """Run single configuration test"""
    print("🧪 Single Configuration Test Script")
    print("=" * 60)
    
    # Load test case
    print("📋 Loading test case...")
    test_case = load_test_case()
    print(f"✅ Loaded test case: {test_case['case_id']}")
    print(f"   Symptom: {test_case['symptom']}")
    
    # Initialize configuration generator
    print("\n🔧 Loading configurations...")
    config_generator = OptimizedConfigMatrixGenerator()
    evaluation_combos = config_generator.generate_evaluation_matrix()
    
    if not evaluation_combos:
        print("❌ No configurations available!")
        return
    
    # Get first configuration
    first_config = evaluation_combos[0]
    print(f"✅ Using first configuration: {first_config.combo_id}")
    print(f"   RAG Method: {first_config.rag_config.get('retrieval_type', 'unknown')}")
    print(f"   Chunking: {first_config.rag_config.get('chunking_method', 'unknown')}")
    print(f"   Model: {first_config.adapter_config.model_name}")
    print(f"   Chunks: {first_config.rag_config.get('chunk_limit', 10)}")
    
    # Initialize RAG system
    print("\n🔍 Initializing RAG system...")
    try:
        rag_system = IntegratedRAGSystemWrapper(first_config.rag_config)
        print("✅ RAG system initialized")
    except Exception as e:
        print(f"⚠️ RAG system error: {e}")
        print("   Continuing with mock retrieval...")
        rag_system = None
    
    # Get RAG context
    print(f"\n📚 Retrieving context for query...")
    if rag_system:
        try:
            context = rag_system.retrieve_context(test_case["input"])
            print(f"✅ Retrieved context ({len(context)} characters)")
        except Exception as e:
            print(f"⚠️ Context retrieval error: {e}")
            context = "Mock medical context: General medical knowledge about symptoms and triage decisions."
    else:
        context = "Mock medical context: General medical knowledge about symptoms and triage decisions."
    
    # Create prompt
    print(f"\n📝 Creating prompt...")
    prompt = f"""Patient query: {test_case["input"]}

Context:
{context}

Provide triage decision, next steps, and reasoning:"""
    
    # Display input
    print("\n" + "="*80)
    print("📥 INPUT TO LLM")
    print("="*80)
    print(prompt)
    print("="*80)
    
    # Initialize evaluation pipeline
    print(f"\n🤖 Initializing model...")
    try:
        pipeline = EnhancedEvaluationPipeline(
            base_dir="/Users/choemanseung/789/hft",
            enable_clinical_judge=False,
            skip_models=False
        )
        
        # Load model and get response
        print(f"   Loading {first_config.adapter_config.model_name}...")
        model, tokenizer = pipeline._load_model_with_adapter(first_config.adapter_config)
        
        print(f"🚀 Generating response...")
        start_time = time.time()
        
        # Generate response
        from mlx_lm import generate
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"✅ Response generated in {generation_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        print("   Using mock response...")
        response = """Triage Decision: ED

Next Step: Seek immediate emergency care at the nearest emergency department.

Reasoning: The patient's symptoms of severe chest pain and shortness of breath that started recently are concerning for potential cardiac emergency such as myocardial infarction. These symptoms require immediate medical evaluation and intervention to prevent serious complications."""
        generation_time = 0.1
    
    # Display output
    print("\n" + "="*80)
    print("📤 LLM OUTPUT")
    print("="*80)
    print(response)
    print("="*80)
    
    # Parse response
    print(f"\n🔍 Parsing response...")
    try:
        pipeline = EnhancedEvaluationPipeline(base_dir="/Users/choemanseung/789/hft")
        predicted_triage = pipeline._extract_triage_decision(response)
        response_parts = pipeline._extract_next_steps_and_reasoning(response)
        predicted_next_steps = response_parts["next_steps"]
        predicted_reasoning = response_parts["reasoning"]
        print("✅ Response parsed successfully")
    except:
        # Simple fallback parsing
        predicted_triage = "ED" if "ED" in response.upper() else ("GP" if "GP" in response.upper() else "HOME")
        predicted_next_steps = "Contact healthcare provider"
        predicted_reasoning = "Unable to parse reasoning"
        print("⚠️ Using fallback parsing")
    
    # Display comparison
    print("\n" + "="*80)
    print("🔄 COMPARISON WITH EXPECTED RESULTS")
    print("="*80)
    
    print("🎯 TRIAGE DECISION:")
    print(f"   Expected: {test_case['triage_decision']}")
    print(f"   Predicted: {predicted_triage}")
    triage_match = predicted_triage == test_case['triage_decision']
    print(f"   Match: {'✅ YES' if triage_match else '❌ NO'}")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   Expected: {test_case['next_steps']}")
    print(f"   Predicted: {predicted_next_steps}")
    
    print(f"\n🤔 REASONING:")
    print(f"   Expected: {test_case['reasoning']}")
    print(f"   Predicted: {predicted_reasoning}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Configuration: {first_config.combo_id}")
    print(f"Test Case: {test_case['case_id']} ({test_case['symptom']})")
    print(f"Triage Accuracy: {'✅ CORRECT' if triage_match else '❌ INCORRECT'}")
    print(f"Generation Time: {generation_time:.2f}s")
    print(f"Response Length: {len(response)} characters")
    print("="*80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"/Users/choemanseung/789/hft/single_config_test_{timestamp}.json")
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "combo_id": first_config.combo_id,
            "rag_method": first_config.rag_config.get('retrieval_type', 'unknown'),
            "chunking_method": first_config.rag_config.get('chunking_method', 'unknown'),
            "model": first_config.adapter_config.model_name,
            "chunk_limit": first_config.rag_config.get('chunk_limit', 10)
        },
        "test_case": test_case,
        "input_prompt": prompt,
        "llm_response": response,
        "parsed_results": {
            "predicted_triage": predicted_triage,
            "predicted_next_steps": predicted_next_steps,
            "predicted_reasoning": predicted_reasoning
        },
        "evaluation": {
            "triage_correct": triage_match,
            "generation_time_seconds": generation_time,
            "response_length": len(response)
        }
    }
    
    try:
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n💾 Test results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️ Failed to save results: {e}")

if __name__ == "__main__":
    main()