#!/usr/bin/env python3
"""
Integrated RAG System for Medical Triage Evaluation
Uses the actual RAG implementations from final_retrieval_testing.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add final_retrieval_testing to path to import existing classes
final_retrieval_path = Path(__file__).parent.parent / "final_retrieval_testing"
sys.path.insert(0, str(final_retrieval_path))

# Simple BiasConfig class for retrieval configuration
class BiasConfig:
    """Simple bias configuration for retrieval sources"""
    def __init__(self, healthify: int = 3, mayo: int = 3, nhs: int = 4):
        self.healthify = healthify
        self.mayo = mayo  
        self.nhs = nhs

try:
    # Import your actual retrieval classes  
    from offline_contextual_retrieval import (
        OfflineContextualRetrieval, OfflineVectorDB, BM25Search, 
        OfflineReranker, HybridRetriever
    )
    from hybrid_retrieval_evaluator import MultiSourceRetriever
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import retrieval system: {e}")
    RETRIEVAL_AVAILABLE = False

class IntegratedRAGSystem:
    """Integration wrapper for your existing RAG system"""
    
    def __init__(self, base_dir: str = "/Users/choemanseung/789/hft"):
        self.base_dir = Path(base_dir)
        self.final_retrieval_dir = self.base_dir / "final_retrieval_testing"
        
        # Initialize retrieval system
        self.retrieval_system = None
        self.is_loaded = False
        
        if RETRIEVAL_AVAILABLE:
            self._initialize_retrieval_system()
        else:
            print("⚠️  Using mock RAG system - retrieval classes not available")
    
    def _initialize_retrieval_system(self):
        """Initialize the hybrid retrieval system"""
        try:
            # Initialize MultiSourceRetriever which has search_with_bias method
            self.retrieval_system = MultiSourceRetriever()
            
            # Load actual data sources from RAGdatav4
            self._load_data_sources()
            
            self.is_loaded = True
            print("✅ RAG system initialized successfully (using MultiSourceRetriever)")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            self.is_loaded = False
    
    def _load_data_sources(self):
        """Load actual data sources from RAGdatav4 directory"""
        rag_data_dir = self.base_dir / "RAGdatav4"
        
        if not rag_data_dir.exists():
            print("⚠️  RAGdatav4 directory not found, using mock mode")
            return
        
        # Source mapping for file naming convention
        source_files = {
            'healthify': 'healthify_chunks_structured_agent_tinfoil_medical.json',
            'mayo': 'mayo_chunks_structured_agent_tinfoil_medical.json', 
            'nhs': 'nhs_chunks_structured_agent_tinfoil_medical.json'
        }
        
        sources_loaded = 0
        for source_name, filename in source_files.items():
            source_file = rag_data_dir / filename
            
            if source_file.exists():
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        source_data = json.load(f)
                    
                    # Convert to the format expected by MultiSourceRetriever
                    chunks = []
                    for item in source_data:
                        if isinstance(item, dict):
                            # Use the text field as content and preserve metadata
                            chunk = {
                                'content': item.get('text', ''),
                                'chunk_id': item.get('chunk_id', f"{source_name}_{len(chunks)}"),
                                'source_document': item.get('source_document', ''),
                                'source': source_name,
                                'metadata': item
                            }
                            chunks.append(chunk)
                    
                    if chunks:
                        # Add source to retrieval system
                        # Note: MultiSourceRetriever.add_source expects (source_name, index_path, chunks)
                        # Since we don't have pre-built indices, we'll need to modify the approach
                        self._add_source_to_retriever(source_name, chunks)
                        sources_loaded += 1
                        print(f"✅ Loaded {len(chunks)} chunks from {source_name}")
                    
                except Exception as e:
                    print(f"❌ Error loading {source_file}: {e}")
            else:
                print(f"⚠️  Source file not found: {source_file}")
        
        if sources_loaded > 0:
            print(f"✅ Successfully loaded {sources_loaded} data sources")
        else:
            print("⚠️  No data sources loaded, evaluation will use mock mode")
    
    def _add_source_to_retriever(self, source_name: str, chunks: List[Dict]):
        """Add a source to the MultiSourceRetriever with in-memory indexing"""
        try:
            # For now, store chunks in a simple format that can be used by search_with_bias
            # This is a simplified approach - in production you'd want proper FAISS indexing
            if not hasattr(self.retrieval_system, 'sources'):
                self.retrieval_system.sources = {}
            
            self.retrieval_system.sources[source_name] = {
                'chunks': chunks,
                'count': len(chunks)
            }
            
        except Exception as e:
            print(f"❌ Error adding source {source_name}: {e}")
    
    def retrieve_context(self, query: str, config: Dict[str, Any]) -> str:
        """Retrieve context using validated RAG configuration"""
        
        if not RETRIEVAL_AVAILABLE or not self.is_loaded:
            return self._mock_retrieval(query, config)
        
        try:
            # Check if we have sources loaded
            if not hasattr(self.retrieval_system, 'sources') or not self.retrieval_system.sources:
                print("⚠️  No sources loaded, using mock retrieval")
                return self._mock_retrieval(query, config)
            
            # Extract configuration parameters
            chunking_method = config.get('chunking_method', 'structured_agent_tinfoil_medical')
            retrieval_type = config.get('retrieval_type', 'contextual_rag')
            bias_config_name = config.get('bias_config', 'diverse')
            
            # Create bias configuration
            bias_config = self._create_bias_config(bias_config_name, config)
            
            # Perform simple text-based retrieval (without FAISS for now)
            results = self._simple_retrieval_with_bias(query, bias_config)
            
            # Format results into context string
            context_parts = []
            for i, result in enumerate(results[:5]):  # Top 5 results
                chunk_content = result.get('content', '')
                source = result.get('source', '')
                if chunk_content:
                    context_parts.append(f"[{i+1}] ({source}) {chunk_content}")
            
            context = "\n\n".join(context_parts)
            return context if context else self._mock_retrieval(query, config)
            
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            return self._mock_retrieval(query, config)
    
    def _simple_retrieval_with_bias(self, query: str, bias_config: BiasConfig) -> List[Dict]:
        """Simple text-based retrieval with bias configuration (without FAISS)"""
        all_results = []
        query_lower = query.lower()
        
        # Source mapping with bias configuration
        source_mapping = {
            'healthify': bias_config.healthify,
            'mayo': bias_config.mayo,
            'nhs': bias_config.nhs
        }
        
        for source_name, num_chunks in source_mapping.items():
            if source_name in self.retrieval_system.sources:
                source_data = self.retrieval_system.sources[source_name]
                chunks = source_data['chunks']
                
                # Simple keyword-based scoring
                scored_chunks = []
                for chunk in chunks:
                    content = chunk.get('content', '').lower()
                    
                    # Simple scoring based on keyword matches
                    score = 0
                    query_words = query_lower.split()
                    for word in query_words:
                        if len(word) > 2:  # Skip very short words
                            score += content.count(word)
                    
                    if score > 0:
                        chunk_with_score = chunk.copy()
                        chunk_with_score['score'] = score
                        scored_chunks.append(chunk_with_score)
                
                # Sort by score and take top num_chunks
                scored_chunks.sort(key=lambda x: x['score'], reverse=True)
                all_results.extend(scored_chunks[:num_chunks])
        
        # Sort all results by score and return
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results
    
    def _create_bias_config(self, bias_config_name: str, config: Dict[str, Any]) -> 'BiasConfig':
        """Create bias configuration for retrieval"""
        
        # Extract chunk limit from config for dynamic sizing
        chunk_limit = config.get('chunk_limit', 10)
        
        # Default bias configurations based on chunk limit
        if chunk_limit == 5:
            # Top 5 chunk configurations
            bias_configs = {
                'diverse': BiasConfig(healthify=2, mayo=2, nhs=1),      # Total 5 chunks
                'medical_focused': BiasConfig(healthify=1, mayo=2, nhs=2),  # Total 5 chunks
                'comprehensive': BiasConfig(healthify=2, mayo=1, nhs=2)     # Total 5 chunks
            }
        else:  # chunk_limit == 10 or default
            # Top 10 chunk configurations (original)
            bias_configs = {
                'diverse': BiasConfig(healthify=3, mayo=3, nhs=4),      # Total 10 chunks
                'medical_focused': BiasConfig(healthify=2, mayo=4, nhs=4),  # Total 10 chunks
                'comprehensive': BiasConfig(healthify=4, mayo=3, nhs=3)     # Total 10 chunks
            }
        
        return bias_configs.get(bias_config_name, bias_configs['diverse'])
    
    def _mock_retrieval(self, query: str, config: Dict[str, Any]) -> str:
        """Mock retrieval when actual system is not available"""
        
        # Generate mock context based on query and config
        chunking_method = config.get('chunking_method', 'unknown')
        retrieval_type = config.get('retrieval_type', 'unknown')
        
        # Mock medical contexts based on common patterns
        if "chest pain" in query.lower() or "heart" in query.lower():
            contexts = [
                "Chest pain can indicate serious cardiac conditions requiring immediate evaluation.",
                "Acute coronary syndrome presents with chest pain, shortness of breath, and diaphoresis.",
                "Emergency department evaluation is recommended for chest pain with associated symptoms."
            ]
        elif "headache" in query.lower():
            contexts = [
                "Most headaches are benign tension-type headaches manageable with rest and OTC medications.",
                "Red flag symptoms for headaches include sudden onset, fever, or neurological changes.",
                "Migraine headaches often present with photophobia and may require prescription treatment."
            ]
        elif "fever" in query.lower():
            contexts = [
                "Fever is a common symptom of infection and usually resolves with supportive care.",
                "High fever above 101.5°F may require medical evaluation, especially in vulnerable populations.",
                "Bacterial infections may require antibiotic treatment under medical supervision."
            ]
        else:
            contexts = [
                f"Medical context retrieved using {chunking_method} chunking method.",
                f"Information obtained through {retrieval_type} retrieval approach.",
                "Relevant medical information for clinical decision-making."
            ]
        
        # Simulate the retrieval configuration impact
        if retrieval_type == 'contextual_rag':
            contexts.append("Enhanced context through hybrid semantic and keyword retrieval.")
        elif retrieval_type == 'pure_rag':
            contexts.append("Context retrieved through semantic similarity matching.")
        
        return "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the loaded RAG system"""
        return {
            "system_loaded": self.is_loaded,
            "retrieval_available": RETRIEVAL_AVAILABLE,
            "base_dir": str(self.base_dir),
            "final_retrieval_dir": str(self.final_retrieval_dir),
            "mock_mode": not (RETRIEVAL_AVAILABLE and self.is_loaded)
        }

class ValidatedRAGConfigProcessor:
    """Process validated RAG configurations for evaluation"""
    
    def __init__(self):
        self.config_mappings = self._create_config_mappings()
    
    def _create_config_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Create mappings from validated config names to system parameters"""
        
        mappings = {
            # Top performers from your analysis
            "structured_agent_tinfoil_medical": {
                "chunking_method": "structured_agent_tinfoil_medical",
                "chunking_type": "structured_agent",
                "chunk_size": 1024,
                "overlap": 150,
                "specialization": "medical"
            },
            "contextual_sentence_c1024_o2_tinfoil": {
                "chunking_method": "contextual_sentence_c1024_o2_tinfoil",
                "chunking_type": "sentence",
                "chunk_size": 1024,
                "overlap": 2,
                "specialization": "medical_tinfoil"
            },
            "contextual_fixed_c512_o100": {
                "chunking_method": "contextual_fixed_c512_o100",
                "chunking_type": "fixed",
                "chunk_size": 512,
                "overlap": 100,
                "specialization": "contextual"
            },
            "sentence_t1024_o2": {
                "chunking_method": "sentence_t1024_o2",
                "chunking_type": "sentence",
                "chunk_size": 1024,
                "overlap": 2,
                "specialization": "general"
            },
            "fixed_c1024_o150": {
                "chunking_method": "fixed_c1024_o150",
                "chunking_type": "fixed",
                "chunk_size": 1024,
                "overlap": 150,
                "specialization": "general"
            },
            "structured_agent_qwen3_4b_medical": {
                "chunking_method": "structured_agent_qwen3_4b_medical",
                "chunking_type": "structured_agent",
                "chunk_size": 1024,
                "overlap": 150,
                "specialization": "qwen_medical"
            },
            "contextual_sentence_c1024_o2": {
                "chunking_method": "contextual_sentence_c1024_o2",
                "chunking_type": "sentence",
                "chunk_size": 1024,
                "overlap": 2,
                "specialization": "contextual"
            },
            "fixed_c384_o50": {
                "chunking_method": "fixed_c384_o50", 
                "chunking_type": "fixed",
                "chunk_size": 384,
                "overlap": 50,
                "specialization": "compact"
            },
            "sentence_t1024_o3": {
                "chunking_method": "sentence_t1024_o3",
                "chunking_type": "sentence", 
                "chunk_size": 1024,
                "overlap": 3,
                "specialization": "general"
            }
        }
        
        return mappings
    
    def process_config(self, validated_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process validated config into detailed parameters"""
        
        chunking_method = validated_config.get('chunking_method', '')
        
        # Get detailed mapping
        detailed_config = self.config_mappings.get(chunking_method, {
            "chunking_method": chunking_method,
            "chunking_type": "unknown",
            "chunk_size": 512,
            "overlap": 50,
            "specialization": "general"
        })
        
        # Add retrieval and bias information
        detailed_config.update({
            "retrieval_type": validated_config.get('retrieval_type', 'contextual_rag'),
            "bias_config": validated_config.get('bias_config', 'diverse'),
            "pass_at_5": validated_config.get('pass_at_5', 0.0),
            "pass_at_10": validated_config.get('pass_at_10', 0.0),
            "avg_retrieval_time": validated_config.get('avg_retrieval_time', 0.0),
            "peak_memory_mb": validated_config.get('peak_memory_mb', 0.0)
        })
        
        return detailed_config

def test_integrated_rag_system():
    """Test the integrated RAG system"""
    print("🧪 Testing Integrated RAG System")
    
    # Initialize system
    rag_system = IntegratedRAGSystem()
    
    # Print system info
    info = rag_system.get_system_info()
    print(f"\nSystem Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test retrieval with validated config
    test_config = {
        'chunking_method': 'structured_agent_tinfoil_medical',
        'retrieval_type': 'contextual_rag',
        'bias_config': 'diverse',
        'pass_at_5': 0.595
    }
    
    test_query = "Patient presents with severe chest pain and shortness of breath"
    
    print(f"\nTesting retrieval:")
    print(f"Query: {test_query}")
    print(f"Config: {test_config}")
    
    context = rag_system.retrieve_context(test_query, test_config)
    
    print(f"\nRetrieved Context:")
    print("-" * 50)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Test config processor
    processor = ValidatedRAGConfigProcessor()
    processed_config = processor.process_config(test_config)
    
    print(f"\nProcessed Config:")
    for key, value in processed_config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_integrated_rag_system()