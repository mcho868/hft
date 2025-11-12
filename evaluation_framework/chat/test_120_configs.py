#!/usr/bin/env python3
"""
Interactive Medical Triage Chat Interface
Allows users to select from the exact 120 evaluation configurations and test queries in real-time.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse
from comprehensive_triage_evaluator import ComprehensiveMedicalTriageEvaluator, EvaluationConfig
from evaluation_core import TriageInferenceEngine

# Add evaluation framework final to path
evaluation_final_path = Path(__file__).parent.parent.parent / "evaluation_framework_final"
sys.path.insert(0, str(evaluation_final_path))

@dataclass
class ConfigSelection:
    """User's configuration selection"""
    config_id: int
    evaluation_config: EvaluationConfig
    description: str

class InteractiveTRIAGEChat:
    """Interactive chat interface for testing 120 evaluation configurations"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.current_config = None
        self.current_engine = None
        self.available_configs = []
        
        print("🏥 Medical Triage Interactive Chat Interface - 120 Evaluation Configs")
        print("=" * 70)
        
        # Initialize evaluator to get exact configurations
        self.evaluator = ComprehensiveMedicalTriageEvaluator(str(base_dir))
        
        # Load the exact 120 configurations
        self._load_120_configurations()
        
    def _load_120_configurations(self):
        """Load the exact 120 evaluation configurations"""
        print("📊 Loading 120 evaluation configurations...")
        
        try:
            # Generate the exact evaluation matrix (120 configs)
            evaluation_configs = self.evaluator.create_evaluation_matrix()
            
            # Convert to user-friendly format
            for i, config in enumerate(evaluation_configs, 1):
                # Create meaningful description
                model_name = config.model_name.replace("_4bit", "").replace("_8bit", "")
                
                if config.adapter_path:
                    adapter_name = Path(config.adapter_path).name.split("_")[-2] if "_" in Path(config.adapter_path).name else "adapter"
                    model_desc = f"{model_name}+{adapter_name}"
                else:
                    model_desc = f"{model_name} (base)"
                
                if config.rag_config:
                    rag_desc = f"RAG: {config.rag_config['name']}"
                else:
                    rag_desc = "No RAG"
                
                description = f"{model_desc} | {rag_desc}"
                
                config_selection = ConfigSelection(
                    config_id=i,
                    evaluation_config=config,
                    description=description
                )
                
                self.available_configs.append(config_selection)
            
            print(f"✅ Loaded {len(self.available_configs)} evaluation configurations")

            # --- Manually add Tinfoil Agent configurations ---
            print("🔧 Manually adding Tinfoil Agent configurations for testing...")
            tinfoil_configs_to_add = []

            try:
                rag_configs = self.evaluator._define_top_rag_configs()
            except AttributeError:
                print("⚠️  Could not find _define_top_rag_configs on evaluator, using fallback RAG configs.")
                rag_configs = [
                    {'name': 'top1_structured_contextual_diverse', 'chunking_method': 'structured_agent_tinfoil_medical', 'retrieval_type': 'contextual_rag', 'bias_config': 'diverse'},
                    {'name': 'top2_structured_pure_diverse', 'chunking_method': 'structured_agent_tinfoil_medical', 'retrieval_type': 'pure_rag', 'bias_config': 'diverse'}
                ]

            # Config 1: Tinfoil Agent No RAG
            tinfoil_no_rag = EvaluationConfig(
                model_name="tinfoil_agent", model_path="N/A", adapter_path=None, rag_config=None,
                test_name="Tinfoil_Agent_NoRAG"
            )
            tinfoil_configs_to_add.append(tinfoil_no_rag)

            # Config 2: Tinfoil Agent with RAG (contextual)
            tinfoil_rag_contextual = EvaluationConfig(
                model_name="tinfoil_agent", model_path="N/A", adapter_path=None,
                rag_config=rag_configs[0],
                test_name="Tinfoil_Agent_RAG_Contextual"
            )
            tinfoil_configs_to_add.append(tinfoil_rag_contextual)

            # Config 3: Tinfoil Agent with RAG (pure)
            tinfoil_rag_pure = EvaluationConfig(
                model_name="tinfoil_agent", model_path="N/A", adapter_path=None,
                rag_config=rag_configs[1],
                test_name="Tinfoil_Agent_RAG_Pure"
            )
            tinfoil_configs_to_add.append(tinfoil_rag_pure)

            start_id = len(self.available_configs) + 1
            for i, config in enumerate(tinfoil_configs_to_add):
                description = f"{config.test_name} (Tinfoil API)"
                config_selection = ConfigSelection(
                    config_id=start_id + i,
                    evaluation_config=config,
                    description=description
                )
                self.available_configs.append(config_selection)

            print(f"✅ Added {len(tinfoil_configs_to_add)} Tinfoil configurations. Total available: {len(self.available_configs)}")
            # --- End of manual add ---
            
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def display_configuration_menu(self) -> ConfigSelection:
        """Display configuration selection menu"""
        print("\n🔧 Evaluation Configurations:")
        print("-" * 80)
        print("Format: CONFIG_ID. MODEL | RAG_CONFIG")
        print("-" * 80)
        
        # Display all 120 configurations in groups of 20
        for i in range(0, len(self.available_configs), 20):
            group_end = min(i + 20, len(self.available_configs))
            print(f"\n📊 CONFIGURATIONS {i+1}-{group_end}:")
            
            for config in self.available_configs[i:group_end]:
                print(f"  {config.config_id:3d}. {config.description}")
        
        # Get user selection
        while True:
            try:
                print(f"\n🎯 Select configuration (1-{len(self.available_configs)}) or 'q' to quit:")
                choice = input("➤ ").strip().lower()
                
                if choice == 'q':
                    print("👋 Goodbye!")
                    sys.exit(0)
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.available_configs):
                    return self.available_configs[choice_num - 1]
                else:
                    print(f"❌ Invalid choice. Please select 1-{len(self.available_configs)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
    
    def initialize_pipeline(self, config: ConfigSelection) -> bool:
        """Initialize the evaluation pipeline with selected configuration"""
        print(f"\n🚀 Initializing configuration {config.config_id}:")
        print(f"   Model: {config.evaluation_config.model_name}")
        print(f"   Adapter: {Path(config.evaluation_config.adapter_path).name if config.evaluation_config.adapter_path else 'None'}")
        print(f"   RAG Config: {config.evaluation_config.rag_config['name'] if config.evaluation_config.rag_config else 'None'}")
        
        try:
            # Initialize inference engine with retriever
            self.current_engine = TriageInferenceEngine(retriever=self.evaluator.retriever)
            
            # Store current configuration
            self.current_config = config
            
            print("✅ Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the selected configuration"""
        if not self.current_engine or not self.current_config:
            return {"error": "No configuration selected"}
        
        eval_config = self.current_config.evaluation_config
        if eval_config.model_name == "tinfoil_agent":
            return self.process_query_with_tinfoil(user_query, eval_config)

        print(f"\n🔍 Processing query with config {self.current_config.config_id}...")
        print(f"Query: {user_query}")
        
        try:
            # Create a test case
            test_case = {
                "case_id": f"interactive_{int(time.time())}",
                "query": user_query,
                "triage_decision": "UNKNOWN",  # We don't know the true answer
            }
            
            # Process using the evaluation core directly
            start_time = time.time()
            
            # Load model if needed
            eval_config = self.current_config.evaluation_config
            model, tokenizer = self.current_engine.load_model_if_needed(
                model_path=eval_config.model_path,
                adapter_path=eval_config.adapter_path
            )
            
            # Show detailed RAG and prompt information
            rag_context = ""
            rag_time = 0.0
            if eval_config.rag_config:
                print(f"\n📊 RAG Configuration: {eval_config.rag_config['name']}")
                print(f"   Chunking: {eval_config.rag_config['chunking_method']}")
                print(f"   Retrieval: {eval_config.rag_config['retrieval_type']}")
                print(f"   Bias: {eval_config.rag_config['bias_config']}")
                
                rag_context, rag_time = self.current_engine.get_rag_context(user_query, eval_config.rag_config)
                print(f"\n🔍 RAG Retrieval Results ({len(rag_context)} chars, {rag_time:.3f}s):")
                print("-" * 60)
                if rag_context:
                    # Show first few lines of context
                    context_lines = rag_context.split('\n')[:5]
                    for line in context_lines:
                        print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
                    if len(context_lines) > 5:
                        print(f"   ... ({len(context_lines) - 5} more lines)")
                else:
                    print("   No context retrieved!")
                print("-" * 60)
            else:
                print("\n📊 No RAG configured for this model")
            
            # Create and show prompt
            prompt = self.current_engine.create_prompt(user_query, rag_context)
            print(f"\n📝 Full Prompt ({len(prompt)} chars):")
            print("-" * 60)
            print(f"{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
            print("-" * 60)
            
            # Generate response
            response_start = time.time()
            response = self.current_engine.generate_response(model, tokenizer, prompt)
            inference_time = time.time() - response_start
            
            print(f"\n🤖 Raw Model Response ({len(response)} chars, {inference_time:.3f}s):")
            print("-" * 60)
            print(f"{response[:300]}{'...' if len(response) > 300 else ''}")
            print("-" * 60)
            
            # Extract triage decision
            predicted_triage = self.current_engine.extract_triage_decision(response)
            print(f"\n🎯 Extracted Triage Decision: {predicted_triage}")
            
            end_time = time.time()
            
            return {
                "case_id": test_case["case_id"],
                "predicted_triage": predicted_triage,
                "predicted_reasoning": response,
                "processing_time_ms": (end_time - start_time) * 1000,
                "inference_time_ms": inference_time * 1000,
                "rag_time_ms": rag_time * 1000,
                "context_length": len(rag_context),
                "prompt_length": len(prompt),
                "response_length": len(response),
                "config_id": self.current_config.config_id,
                "model_name": eval_config.model_name,
                "has_adapter": eval_config.adapter_path is not None,
                "has_rag": eval_config.rag_config is not None,
                "rag_name": eval_config.rag_config['name'] if eval_config.rag_config else None
            }
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Processing failed: {str(e)}"}
    
    def display_result(self, result: Dict[str, Any]):
        """Display query processing result"""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("\n" + "="*70)
        print("🏥 MEDICAL TRIAGE RESULT")
        print("="*70)
        print(f"🎯 Triage Decision: {result['predicted_triage']}")
        
        print("\n" + "-"*70)
        print("⚙️  CONFIGURATION & PERFORMANCE")
        print("-"*70)
        print(f"Config ID: {result.get('config_id', 'unknown')}")
        print(f"Model: {result.get('model_name', 'unknown')}")
        print(f"Has Adapter: {result.get('has_adapter', False)}")
        print(f"Has RAG: {result.get('has_rag', False)}")
        if result.get('rag_name'):
            print(f"RAG Config: {result['rag_name']}")
        
        print(f"\n📊 Timing Breakdown:")
        print(f"   Total Processing: {result.get('processing_time_ms', 0):.1f}ms")
        print(f"   Model Inference: {result.get('inference_time_ms', 0):.1f}ms")
        if result.get('rag_time_ms', 0) > 0:
            print(f"   RAG Retrieval: {result.get('rag_time_ms', 0):.1f}ms")
        
        print(f"\n📏 Content Lengths:")
        if result.get('context_length', 0) > 0:
            print(f"   RAG Context: {result.get('context_length', 0)} chars")
        print(f"   Prompt: {result.get('prompt_length', 0)} chars")
        print(f"   Response: {result.get('response_length', 0)} chars")
        
        print("="*70)

    def process_query_with_tinfoil(self, user_query: str, eval_config: EvaluationConfig) -> Dict[str, Any]:
        """Process user query using the Tinfoil Agent API."""
        print(f"\n🔍 Processing query with Tinfoil Agent...")
        try:
            start_time = time.time()
            from llm_quality_judge import TinfoilLLMClient
            tinfoil_client = TinfoilLLMClient()

            # Get RAG context if configured
            rag_context, rag_time = "", 0.0
            if eval_config.rag_config:
                print(f"\n📊 RAG Configuration: {eval_config.rag_config['name']}")
                rag_context, rag_time = self.current_engine.get_rag_context(user_query, eval_config.rag_config)
                print(f"\n🔍 RAG Retrieval Results ({len(rag_context)} chars, {rag_time:.3f}s):")
                print("-" * 60)
                if rag_context:
                    context_lines = rag_context.split('\n')
                    for line in context_lines[:5]:
                        print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
                    if len(context_lines) > 5:
                        print(f"   ... ({len(context_lines) - 5} more lines)")
                else:
                    print("   No context retrieved!")
                print("-" * 60)
            else:
                print("\n📊 No RAG configured for this model")

            # Create and show prompt
            if rag_context:
                prompt = f"""Patient query: {user_query}\n\nContext:\n{rag_context}\n\nProvide Triage decision, next steps and reasoing. answer in this exact format.\n\nTriage Decison: (ED, GP, HOME)\nNext Steps:\nReasoning:"""
            else:
                prompt = f"""Patient query: {user_query}\n\nProvide Triage decision, next steps and reasoing. answer in this exact format.\n\nTriage Decison: (ED, GP, HOME)\nNext Steps:\nReasoning:"""
            print(f"\n📝 Full Prompt ({len(prompt)} chars):")
            print("-" * 60)
            print(f"{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
            print("-" * 60)

            # Generate response
            response_start = time.time()
            response = tinfoil_client.complete(prompt)
            inference_time = time.time() - response_start

            print(f"\n🤖 Raw Model Response ({len(response)} chars, {inference_time:.3f}s):")
            print("-" * 60)
            print(f"{response}")
            print("-" * 60)

            # Extract triage decision
            predicted_triage = self.current_engine.extract_triage_decision(response)
            print(f"\n🎯 Extracted Triage Decision: {predicted_triage}")
            
            end_time = time.time()

            return {
                "case_id": f"interactive_{int(time.time())}",
                "predicted_triage": predicted_triage,
                "predicted_reasoning": response,
                "processing_time_ms": (end_time - start_time) * 1000,
                "inference_time_ms": inference_time * 1000,
                "rag_time_ms": rag_time * 1000,
                "context_length": len(rag_context),
                "prompt_length": len(prompt),
                "response_length": len(response),
                "config_id": self.current_config.config_id,
                "model_name": eval_config.model_name,
                "has_adapter": False,
                "has_rag": eval_config.rag_config is not None,
                "rag_name": eval_config.rag_config['name'] if eval_config.rag_config else None
            }

        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Processing failed: {str(e)}"}
    
    def format_query(self, age: str, gender: str, symptoms: str) -> str:
        """Format query in standard format: 'Kia ora, I'm a [age]-year-old [gender]. [symptoms]'"""
        return f"Kia ora, I'm a {age}-year-old {gender}. {symptoms}"

    def start_chat_session(self):
        """Start interactive chat session"""
        # Select configuration
        selected_config = self.display_configuration_menu()

        # Initialize pipeline
        if not self.initialize_pipeline(selected_config):
            print("❌ Failed to initialize pipeline. Exiting.")
            return

        print(f"\n💬 Chat session started with config {selected_config.config_id}")
        print("Enter patient information:")
        print("Format: age, gender, symptoms (e.g., '25, Female, I have chest pain')")
        print("Or type 'quit' to exit, 'config' to change configuration")
        print("-" * 80)

        while True:
            try:
                # Get user input
                user_input = input("\n🩺 Patient Info (age, gender, symptoms): ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Chat session ended. Thank you!")
                    break

                elif user_input.lower() == 'config':
                    # Change configuration
                    selected_config = self.display_configuration_menu()
                    if self.initialize_pipeline(selected_config):
                        print(f"✅ Switched to config {selected_config.config_id}")
                    continue

                elif not user_input:
                    print("Please enter patient information or 'quit' to exit")
                    continue

                # Parse input: age, gender, symptoms
                parts = [p.strip() for p in user_input.split(',', 2)]
                if len(parts) < 3:
                    print("❌ Invalid format. Use: age, gender, symptoms")
                    print("   Example: 25, Female, I have chest pain")
                    continue

                age, gender, symptoms = parts
                formatted_query = self.format_query(age, gender, symptoms)
                print(f"📝 Query: {formatted_query}")

                # Process query
                result = self.process_query(formatted_query)
                
                # Display result
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Interactive Medical Triage Chat - 120 Configs")
    parser.add_argument("--base-dir", default="/Users/choemanseung/789/hft",
                       help="Base directory for the evaluation framework")
    
    args = parser.parse_args()
    
    try:
        # Initialize and start chat interface
        chat_interface = InteractiveTRIAGEChat(args.base_dir)
        chat_interface.start_chat_session()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()