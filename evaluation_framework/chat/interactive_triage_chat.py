#!/usr/bin/env python3
"""
Interactive Medical Triage Chat Interface
Allows users to select from 600 optimized configurations and test queries in real-time.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from optimized_config_generator import OptimizedConfigMatrixGenerator, EvaluationCombo
    from integrated_rag_system import IntegratedRAGSystem
    from enhanced_evaluation_pipeline import EnhancedEvaluationPipeline
except ImportError as e:
    print(f"❌ Error importing evaluation framework: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

@dataclass
class ConfigSelection:
    """User's configuration selection"""
    combo_id: str
    rag_method: str
    chunking_method: str
    model_name: str
    adapter_path: str
    description: str

class InteractiveTRIAGEChat:
    """Interactive chat interface for testing medical triage configurations"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.current_config = None
        self.current_pipeline = None
        self.available_configs = []
        
        print("🏥 Medical Triage Interactive Chat Interface")
        print("=" * 60)
        
        # Initialize configuration generator
        self.config_generator = OptimizedConfigMatrixGenerator(str(base_dir))
        
        # Load available configurations
        self._load_configurations()
        
    def _load_configurations(self):
        """Load and parse all 600 optimized configurations"""
        print("📊 Loading optimized configurations...")
        
        try:
            # Generate the full configuration matrix
            evaluation_combos = self.config_generator.generate_evaluation_matrix()
            
            # Convert to user-friendly format
            for combo in evaluation_combos[:50]:  # Limit to first 50 for demo
                rag_config = combo.rag_config
                adapter_config = combo.adapter_config
                
                # Extract meaningful descriptions
                rag_method = f"{rag_config.get('retrieval_type', 'unknown')}"
                chunking_method = rag_config.get('chunking_method', 'unknown')
                model_name = getattr(adapter_config, 'model_name', 'unknown')
                adapter_path = getattr(adapter_config, 'adapter_path', 'unknown')
                
                # Create description
                description = f"{rag_method} + {chunking_method} + {model_name}"
                
                config_selection = ConfigSelection(
                    combo_id=combo.combo_id,
                    rag_method=rag_method,
                    chunking_method=chunking_method,
                    model_name=model_name,
                    adapter_path=adapter_path,
                    description=description
                )
                
                self.available_configs.append(config_selection)
            
            print(f"✅ Loaded {len(self.available_configs)} configurations for testing")
            
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            # Create some mock configurations for demo
            self._create_mock_configurations()
    
    def _create_mock_configurations(self):
        """Create mock configurations for demonstration"""
        print("📝 Creating demo configurations...")
        
        mock_configs = [
            ("structured_agent_tinfoil_medical", "sentence_transformers", "SmolLM2-360M", "demo_adapter_1"),
            ("hybrid_retrieval", "recursive_character", "SmolLM2-1.7B", "demo_adapter_2"),
            ("contextual_rag", "semantic_chunking", "Qwen2.5-3B", "demo_adapter_3"),
            ("multi_source", "fixed_size", "SmolLM2-360M", "demo_adapter_4"),
            ("bm25_semantic", "document_aware", "SmolLM2-1.7B", "demo_adapter_5")
        ]
        
        for i, (rag_method, chunking, model, adapter) in enumerate(mock_configs):
            combo_id = f"DEMO_{i+1}_{hash(f'{rag_method}{chunking}{model}') % 10000:04d}"
            description = f"{rag_method} + {chunking} + {model}"
            
            config = ConfigSelection(
                combo_id=combo_id,
                rag_method=rag_method,
                chunking_method=chunking,
                model_name=model,
                adapter_path=adapter,
                description=description
            )
            
            self.available_configs.append(config)
    
    def display_configuration_menu(self) -> ConfigSelection:
        """Display configuration selection menu"""
        print("\n🔧 Available Configurations:")
        print("-" * 80)
        
        # Group by RAG method for better organization
        rag_groups = {}
        for config in self.available_configs:
            rag_method = config.rag_method
            if rag_method not in rag_groups:
                rag_groups[rag_method] = []
            rag_groups[rag_method].append(config)
        
        # Display grouped configurations
        config_index = 1
        index_to_config = {}
        
        for rag_method, configs in rag_groups.items():
            print(f"\n📊 {rag_method.upper()} RAG METHOD:")
            for config in configs[:5]:  # Limit to 5 per group for readability
                print(f"  {config_index:2d}. {config.description}")
                print(f"      ID: {config.combo_id}")
                print(f"      Model: {config.model_name}")
                print(f"      Chunking: {config.chunking_method}")
                index_to_config[config_index] = config
                config_index += 1
        
        # Get user selection
        while True:
            try:
                print(f"\n🎯 Select configuration (1-{len(index_to_config)}) or 'q' to quit:")
                choice = input("➤ ").strip().lower()
                
                if choice == 'q':
                    print("👋 Goodbye!")
                    sys.exit(0)
                
                choice_num = int(choice)
                if choice_num in index_to_config:
                    return index_to_config[choice_num]
                else:
                    print(f"❌ Invalid choice. Please select 1-{len(index_to_config)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
    
    def initialize_pipeline(self, config: ConfigSelection) -> bool:
        """Initialize the evaluation pipeline with selected configuration"""
        print(f"\n🚀 Initializing pipeline with configuration: {config.combo_id}")
        print(f"   RAG Method: {config.rag_method}")
        print(f"   Chunking: {config.chunking_method}")  
        print(f"   Model: {config.model_name}")
        
        try:
            # Initialize enhanced evaluation pipeline
            self.current_pipeline = EnhancedEvaluationPipeline(
                base_dir=str(self.base_dir),
                batch_size=1,
                max_workers=1,
                enable_clinical_judge=False,  # Disable for interactive chat
                skip_models=False
            )
            
            # Store current configuration
            self.current_config = config
            
            print("✅ Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            return False
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the selected configuration"""
        if not self.current_pipeline or not self.current_config:
            return {"error": "No configuration selected"}
        
        print(f"\n🔍 Processing query with {self.current_config.combo_id}...")
        print(f"Query: {user_query}")
        
        try:
            # Create a test case for real processing
            test_case = {
                "case_id": f"interactive_{int(time.time())}",
                "input": user_query,  # Use "input" field as expected by pipeline
                "triage_decision": "UNKNOWN",  # We don't know the true answer
                "next_steps": "UNKNOWN",
                "reasoning": "UNKNOWN"
            }
            
            # Process through the real evaluation pipeline
            result = self._process_real_case(test_case)
            
            return result
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
    
    def _process_real_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Process case through real evaluation pipeline"""
        try:
            # Get the combination data for the selected configuration
            combo_data = self._create_combo_from_config(self.current_config)
            
            # Use the pipeline's single case evaluation
            start_time = time.time()
            result = self.current_pipeline.evaluate_combination_with_clinical_assessment(combo_data, single_case=case)
            end_time = time.time()
            
            # Extract relevant information from result
            return {
                "case_id": case["case_id"],
                "predicted_triage": result.detailed_metrics.get("predicted_triage", "GP") if result.detailed_metrics else "GP",
                "predicted_next_steps": result.detailed_metrics.get("predicted_next_steps", "Contact healthcare provider") if result.detailed_metrics else "Contact healthcare provider",
                "predicted_reasoning": result.detailed_metrics.get("predicted_reasoning", "Unable to generate reasoning") if result.detailed_metrics else "Unable to generate reasoning",
                "triage_accuracy": result.triage_accuracy,
                "f1_score": result.f1_score,
                "f2_score": result.f2_score,
                "next_step_quality": result.next_step_quality,
                "rag_method": self.current_config.rag_method,
                "model": self.current_config.model_name,
                "chunk_limit": getattr(self.current_config, 'chunk_limit', 10),
                "processing_time_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": result.memory_usage_mb,
                "inference_speed_tps": result.inference_speed_tps
            }
            
        except Exception as e:
            print(f"❌ Real processing failed: {e}")
            # Fallback to simplified processing
            return self._fallback_process_case(case)
    
    def _fallback_process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when real pipeline fails"""
        # Simple keyword-based processing as fallback
        query = case["input"].lower()
        
        if any(word in query for word in ["chest pain", "heart attack", "stroke", "emergency"]):
            triage = "ED"
            next_steps = "Immediate emergency department evaluation - call 911 or go to nearest emergency room"
            reasoning = "Symptoms suggest potential emergency requiring immediate medical attention"
        elif any(word in query for word in ["fever", "cough", "headache", "nausea"]):
            triage = "GP"
            next_steps = "Schedule appointment with your primary care physician within 24-48 hours"
            reasoning = "Symptoms suggest condition that can be managed by primary care"
        else:
            triage = "HOME"
            next_steps = "Monitor symptoms at home. Consider over-the-counter remedies and rest"
            reasoning = "Symptoms appear mild and suitable for home management"
        
        return {
            "case_id": case["case_id"],
            "predicted_triage": triage,
            "predicted_next_steps": next_steps,
            "predicted_reasoning": reasoning,
            "triage_accuracy": "N/A",
            "f1_score": "N/A",
            "f2_score": "N/A", 
            "next_step_quality": "N/A",
            "rag_method": self.current_config.rag_method,
            "model": self.current_config.model_name,
            "processing_time_ms": 100,
            "memory_usage_mb": 50,
            "note": "Fallback processing used"
        }
    
    def _create_combo_from_config(self, config: ConfigSelection) -> object:
        """Create evaluation combo object from config selection"""
        # Create mock combo object with same interface as evaluation pipeline expects
        class MockCombo:
            def __init__(self, config):
                self.combo_id = config.combo_id
                self.rag_config = {
                    'chunking_method': config.chunking_method,
                    'retrieval_type': config.rag_method,
                    'bias_config': 'diverse',
                    'chunk_limit': getattr(config, 'chunk_limit', 10)
                }
                self.adapter_config = type('AdapterConfig', (), {
                    'model_name': config.model_name,
                    'adapter_path': config.adapter_path
                })()
        
        return MockCombo(config)
    
    def display_result(self, result: Dict[str, Any]):
        """Display query processing result"""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("\n" + "="*70)
        print("🏥 MEDICAL TRIAGE RESULT")
        print("="*70)
        print(f"🎯 Triage Decision: {result['predicted_triage']}")
        print(f"📋 Next Steps: {result['predicted_next_steps']}")
        
        # Show reasoning if available
        if 'predicted_reasoning' in result and result['predicted_reasoning']:
            print(f"🤔 Reasoning: {result['predicted_reasoning']}")
        
        print("\n" + "-"*70)
        print("📊 PERFORMANCE METRICS")
        print("-"*70)
        
        # Show accuracy metrics if available
        if result.get('triage_accuracy') != "N/A":
            print(f"🎯 Triage Accuracy: {result.get('triage_accuracy', 0):.3f}")
        if result.get('f1_score') != "N/A":
            print(f"📈 F1 Score: {result.get('f1_score', 0):.3f}")
        if result.get('f2_score') != "N/A":
            print(f"🔥 F2 Score: {result.get('f2_score', 0):.3f} (Medical Priority)")
        if result.get('next_step_quality') != "N/A":
            print(f"⭐ Next Step Quality: {result.get('next_step_quality', 0):.1f}/10")
        
        print("\n" + "-"*70)
        print("⚙️  CONFIGURATION DETAILS")
        print("-"*70)
        print(f"RAG Method: {result.get('rag_method', 'unknown')}")
        print(f"Model: {result.get('model', 'unknown')}")
        if 'chunk_limit' in result:
            print(f"Chunks Retrieved: {result['chunk_limit']}")
        print(f"Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
        print(f"Memory Usage: {result.get('memory_usage_mb', 0):.1f}MB")
        if 'inference_speed_tps' in result:
            print(f"Inference Speed: {result.get('inference_speed_tps', 0):.1f} tokens/sec")
        
        # Show note if fallback was used
        if 'note' in result:
            print(f"\n⚠️  Note: {result['note']}")
        
        print("="*70)
    
    def start_chat_session(self):
        """Start interactive chat session"""
        # Select configuration
        selected_config = self.display_configuration_menu()
        
        # Initialize pipeline
        if not self.initialize_pipeline(selected_config):
            print("❌ Failed to initialize pipeline. Exiting.")
            return
        
        print(f"\n💬 Chat session started with {selected_config.combo_id}")
        print("Type your medical query or 'quit' to exit, 'config' to change configuration")
        print("-" * 80)
        
        while True:
            try:
                # Get user input
                user_query = input("\n🩺 Medical Query: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Chat session ended. Thank you!")
                    break
                    
                elif user_query.lower() == 'config':
                    # Change configuration
                    selected_config = self.display_configuration_menu()
                    if self.initialize_pipeline(selected_config):
                        print(f"✅ Switched to {selected_config.combo_id}")
                    continue
                    
                elif not user_query:
                    print("Please enter a medical query or 'quit' to exit")
                    continue
                
                # Process query
                result = self.process_query(user_query)
                
                # Display result
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Interactive Medical Triage Chat")
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