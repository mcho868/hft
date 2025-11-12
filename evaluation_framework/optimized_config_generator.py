#!/usr/bin/env python3
"""
Optimized Configuration Generator for Medical Triage Evaluation
Uses pre-validated top-performing RAG configurations from empirical testing.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import hashlib

class AdapterConfig:
    """Configuration for a trained adapter - flexible to handle any config fields"""
    def __init__(self, adapter_path: str = None, model_name: str = None, 
                 adapter_type: str = None, training_config: Dict[str, Any] = None, **kwargs):
        self.adapter_path = adapter_path
        self.model_name = model_name
        self.adapter_type = adapter_type
        self.training_config = training_config or {}
        
        # Handle any additional fields from actual adapter configs
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class EvaluationCombo:
    """Single evaluation combination of RAG config + adapter"""
    combo_id: str
    rag_config: Dict[str, Any]
    adapter_config: AdapterConfig

@dataclass
class ValidatedRAGConfig:
    """Pre-validated RAG configuration from empirical testing"""
    chunking_method: str
    retrieval_type: str
    bias_config: str
    pass_at_5: float
    pass_at_10: float
    avg_retrieval_time: float
    peak_memory_mb: float
    config_id: str

class OptimizedConfigMatrixGenerator:
    """Generates evaluation matrix using only top-performing RAG configurations"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.standard_adapters_dir = self.base_dir / "triage_adapters"
        self.safety_adapters_dir = self.base_dir / "safety_triage_adapters"
        
        # Load pre-validated RAG configurations from both pass@5 and pass@10 top performers
        self.pass5_results_file = self.base_dir / "final_retrieval_testing/analysis_output/top_pass_5_performers.csv"
        self.pass10_results_file = self.base_dir / "final_retrieval_testing/analysis_output/top_pass_10_performers.csv"
        self.validated_rag_configs = self._load_validated_rag_configs()
        
    def _load_validated_rag_configs(self) -> List[ValidatedRAGConfig]:
        """Load top-performing RAG configurations from both pass@5 and pass@10 CSVs"""
        configs = []
        config_signatures = set()  # To avoid duplicates
        
        # Load from top pass@5 performers (top 4 only)
        if self.pass5_results_file.exists():
            try:
                df5 = pd.read_csv(self.pass5_results_file)
                # Take only top 4 pass@5 performers
                df5_top4 = df5.head(4)
                for idx, row in df5_top4.iterrows():
                    signature = f"{row['chunking_method']}_{row['retrieval_type']}_{row['bias_config']}"
                    if signature not in config_signatures:
                        config = ValidatedRAGConfig(
                            chunking_method=row['chunking_method'],
                            retrieval_type=row['retrieval_type'],
                            bias_config=row['bias_config'],
                            pass_at_5=row['pass_at_5'],
                            pass_at_10=row['pass_at_10'],
                            avg_retrieval_time=row['avg_retrieval_time'],
                            peak_memory_mb=row['peak_memory_mb'],
                            config_id=f"pass5_{idx+1}_{row['chunking_method'][:8]}_{row['retrieval_type'][:6]}"
                        )
                        configs.append(config)
                        config_signatures.add(signature)
                print(f"✅ Loaded {len(configs)} configurations from top 4 pass@5 performers")
            except Exception as e:
                print(f"❌ Error loading pass@5 configurations: {e}")
        else:
            print(f"⚠️  Pass@5 results file not found: {self.pass5_results_file}")
        
        # Load from top pass@10 performers (top 4 only, excluding duplicates from pass@5)
        if self.pass10_results_file.exists():
            try:
                df10 = pd.read_csv(self.pass10_results_file)
                # Take only top 4 pass@10 performers
                df10_top4 = df10.head(4)
                added_from_pass10 = 0
                for idx, row in df10_top4.iterrows():
                    signature = f"{row['chunking_method']}_{row['retrieval_type']}_{row['bias_config']}"
                    if signature not in config_signatures:
                        config = ValidatedRAGConfig(
                            chunking_method=row['chunking_method'],
                            retrieval_type=row['retrieval_type'],
                            bias_config=row['bias_config'],
                            pass_at_5=row['pass_at_5'],
                            pass_at_10=row['pass_at_10'],
                            avg_retrieval_time=row['avg_retrieval_time'],
                            peak_memory_mb=row['peak_memory_mb'],
                            config_id=f"pass10_{idx+1}_{row['chunking_method'][:8]}_{row['retrieval_type'][:6]}"
                        )
                        configs.append(config)
                        config_signatures.add(signature)
                        added_from_pass10 += 1
                print(f"✅ Added {added_from_pass10} additional configurations from top 4 pass@10 performers")
            except Exception as e:
                print(f"❌ Error loading pass@10 configurations: {e}")
        else:
            print(f"⚠️  Pass@10 results file not found: {self.pass10_results_file}")
        
        if not configs:
            print("⚠️  No RAG configurations loaded, using fallback configurations...")
            return self._create_fallback_configs()
        
        print(f"🎯 Total validated RAG configurations: {len(configs)}")
        return configs
    
    def _create_fallback_configs(self) -> List[ValidatedRAGConfig]:
        """Create fallback configurations if CSV not available"""
        return [
            ValidatedRAGConfig(
                chunking_method="structured_agent_tinfoil_medical",
                retrieval_type="contextual_rag",
                bias_config="diverse",
                pass_at_5=0.595,
                pass_at_10=0.77,
                avg_retrieval_time=0.0321,
                peak_memory_mb=1.8,
                config_id="fallback_1_structured_contextual"
            ),
            ValidatedRAGConfig(
                chunking_method="structured_agent_tinfoil_medical", 
                retrieval_type="pure_rag",
                bias_config="diverse",
                pass_at_5=0.59,
                pass_at_10=0.73,
                avg_retrieval_time=0.0017,
                peak_memory_mb=30.3,
                config_id="fallback_2_structured_pure"
            )
        ]
    
    def scan_adapter_directories(self) -> List[AdapterConfig]:
        """Scan adapter directories and extract configuration information"""
        adapters = []
        
        # Scan standard adapters
        if self.standard_adapters_dir.exists():
            for adapter_dir in self.standard_adapters_dir.iterdir():
                if adapter_dir.is_dir() and (adapter_dir / "adapters.safetensors").exists():
                    config = self._extract_adapter_config(adapter_dir, "standard")
                    if config:
                        adapters.append(config)
        
        # Scan safety adapters
        if self.safety_adapters_dir.exists():
            for adapter_dir in self.safety_adapters_dir.iterdir():
                if adapter_dir.is_dir() and (adapter_dir / "adapters.safetensors").exists():
                    config = self._extract_adapter_config(adapter_dir, "safety")
                    if config:
                        adapters.append(config)
        
        return adapters
    
    def _extract_adapter_config(self, adapter_dir: Path, adapter_type: str) -> AdapterConfig:
        """Extract configuration from adapter directory"""
        try:
            # Try to load adapter config from config.json
            config_file = adapter_dir / "adapter_config.json"
            base_config = {}
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    base_config = json.load(f)
            
            # Extract model name from directory structure or config
            model_name = self._extract_model_name(adapter_dir.name, base_config)
            
            return AdapterConfig(
                adapter_path=str(adapter_dir),
                model_name=model_name,
                adapter_type=adapter_type,
                base_config=base_config
            )
        except Exception as e:
            print(f"Error extracting config from {adapter_dir}: {e}")
            return None
    
    def _extract_model_name(self, dir_name: str, config: Dict) -> str:
        """Extract model name from directory name or config"""
        # Try to extract from config first
        if "base_model" in config:
            return config["base_model"]
        
        # Extract from directory name patterns
        if "SmolLM2-360M" in dir_name:
            return "SmolLM2-360M"
        elif "SmolLM2-135M" in dir_name:
            return "SmolLM2-135M"
        elif "Gemma-270M" in dir_name:
            return "Gemma-270M"
        else:
            # Fallback: use directory name
            return dir_name.split('_')[0] if '_' in dir_name else dir_name
    
    def create_optimized_evaluation_matrix(self) -> List[EvaluationCombo]:
        """Create evaluation matrix using only validated RAG configurations"""
        adapter_configs = self.scan_adapter_directories()
        
        evaluation_combos = []
        
        # Test both top 5 and top 10 chunk configurations
        chunk_limits = [5, 10]
        
        for rag_config in self.validated_rag_configs:
            for adapter_config in adapter_configs:
                for chunk_limit in chunk_limits:
                    # Generate unique ID for this combination including chunk limit
                    combo_id = self._generate_combo_id(rag_config, adapter_config, chunk_limit)
                    
                    # Convert ValidatedRAGConfig to dictionary for EvaluationCombo
                    rag_config_dict = {
                        'chunking_method': rag_config.chunking_method,
                        'retrieval_type': rag_config.retrieval_type,
                        'bias_config': rag_config.bias_config,
                        'chunk_limit': chunk_limit,  # Add chunk limit to config
                        'pass_at_5': rag_config.pass_at_5,
                        'pass_at_10': rag_config.pass_at_10,
                        'avg_retrieval_time': rag_config.avg_retrieval_time,
                        'peak_memory_mb': rag_config.peak_memory_mb,
                        'config_id': rag_config.config_id
                    }
                    
                    # Create evaluation combo with correct structure
                    combo = type('EvaluationCombo', (), {
                        'rag_config': rag_config_dict,
                        'adapter_config': adapter_config,
                        'combo_id': combo_id
                    })()
                    evaluation_combos.append(combo)
        
        return evaluation_combos
    
    def _generate_combo_id(self, rag_config: ValidatedRAGConfig, adapter_config: AdapterConfig, chunk_limit: int = 10) -> str:
        """Generate unique ID for configuration combination"""
        # Create hash from configuration parameters including chunk limit
        config_str = f"{rag_config.config_id}_{adapter_config.adapter_path}_{chunk_limit}"
        combo_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create readable ID with chunk limit indicator
        model_short = adapter_config.model_name.replace("SmolLM2-", "S").replace("Gemma-", "G").replace("M", "")
        rag_short = rag_config.config_id.split('_')[1] if '_' in rag_config.config_id else rag_config.config_id[:8]
        chunk_indicator = f"C{chunk_limit}"  # C5 or C10
        
        return f"{model_short}_{rag_short}_{chunk_indicator}_{combo_hash}"
    
    def save_matrix_to_file(self, evaluation_combos: List[EvaluationCombo], 
                           output_file: str = "optimized_evaluation_matrix.json"):
        """Save optimized evaluation matrix to JSON file"""
        matrix_data = []
        
        for combo in evaluation_combos:
            combo_data = {
                "combo_id": combo.combo_id,
                "rag_config": combo.rag_config,  # Already a dict
                "adapter_config": {
                    "adapter_path": combo.adapter_config.adapter_path,
                    "model_name": combo.adapter_config.model_name,
                    "adapter_type": combo.adapter_config.adapter_type,
                    "base_config": combo.adapter_config.base_config
                }
            }
            matrix_data.append(combo_data)
        
        output_path = self.base_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(matrix_data, f, indent=2)
        
        return output_path
    
    def print_optimized_summary(self, evaluation_combos: List[EvaluationCombo]):
        """Print summary of optimized evaluation matrix"""
        print("\n" + "="*70)
        print("OPTIMIZED EVALUATION MATRIX SUMMARY")
        print("="*70)
        
        # Count by adapter type
        standard_count = sum(1 for c in evaluation_combos if c.adapter_config.adapter_type == "standard")
        safety_count = sum(1 for c in evaluation_combos if c.adapter_config.adapter_type == "safety")
        
        # Count by model
        model_counts = {}
        for combo in evaluation_combos:
            model = combo.adapter_config.model_name
            model_counts[model] = model_counts.get(model, 0) + 1
        
        # RAG configuration performance summary
        rag_performance = {}
        for combo in evaluation_combos:
            rag_id = combo.rag_config['config_id']
            if rag_id not in rag_performance:
                rag_performance[rag_id] = {
                    'pass_at_5': combo.rag_config['pass_at_5'],
                    'pass_at_10': combo.rag_config['pass_at_10'],
                    'retrieval_time': combo.rag_config['avg_retrieval_time'],
                    'memory_mb': combo.rag_config['peak_memory_mb']
                }
        
        print(f"Total Evaluation Combinations: {len(evaluation_combos)}")
        print(f"Validated RAG Configurations: {len(self.validated_rag_configs)}")
        print(f"Standard Adapters: {standard_count}")
        print(f"Safety Adapters: {safety_count}")
        
        print("\nModel Distribution:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count} combinations")
        
        print(f"\nTop RAG Configuration Performance:")
        sorted_rag = sorted(rag_performance.items(), key=lambda x: x[1]['pass_at_5'], reverse=True)
        for i, (rag_id, perf) in enumerate(sorted_rag[:5], 1):
            print(f"  {i}. {rag_id}")
            print(f"     Pass@5: {perf['pass_at_5']:.3f}, Pass@10: {perf['pass_at_10']:.3f}")
            print(f"     Time: {perf['retrieval_time']:.4f}s, Memory: {perf['memory_mb']:.1f}MB")
        
        # Time savings calculation
        original_combinations = 243 * len([c for c in evaluation_combos if c.adapter_config.adapter_type in ["standard", "safety"]])
        original_combinations = original_combinations // len(evaluation_combos) * 243  # Rough estimate
        time_savings = (original_combinations - len(evaluation_combos)) / original_combinations * 100 if original_combinations > 0 else 0
        
        print(f"\n🚀 OPTIMIZATION IMPACT:")
        print(f"  Original space: ~14,580 combinations")
        print(f"  Optimized space: {len(evaluation_combos)} combinations")
        print(f"  Reduction: {100 - (len(evaluation_combos)/14580*100):.1f}%")
        print(f"  Estimated time savings: {time_savings:.1f}%")

def main():
    """Test optimized configuration generator"""
    print("🚀 Testing Optimized Configuration Generator")
    
    generator = OptimizedConfigMatrixGenerator()
    
    print("\n📊 Loading validated RAG configurations...")
    print(f"Found {len(generator.validated_rag_configs)} validated RAG configs")
    
    print("\n🔍 Scanning adapter directories...")
    adapters = generator.scan_adapter_directories()
    print(f"Found {len(adapters)} trained adapters")
    
    print("\n🔗 Creating optimized evaluation matrix...")
    evaluation_combos = generator.create_optimized_evaluation_matrix()
    
    # Print summary
    generator.print_optimized_summary(evaluation_combos)
    
    # Save matrix to file
    print(f"\n💾 Saving optimized evaluation matrix...")
    output_file = generator.save_matrix_to_file(evaluation_combos)
    print(f"Matrix saved to: {output_file}")
    
    print(f"\n✅ Optimized configuration generation complete!")
    print(f"   Ready to evaluate {len(evaluation_combos)} combinations instead of 14,580!")
    
    return evaluation_combos, output_file

if __name__ == "__main__":
    main()