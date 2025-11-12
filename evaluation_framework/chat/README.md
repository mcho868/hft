# Interactive Medical Triage Chat Interface

This directory contains an interactive chat interface that allows you to test the 600 optimized medical triage configurations in real-time.

## Features

🏥 **Interactive Testing**: Test any of the 600 optimized configurations with your own medical queries  
🔧 **Configuration Selection**: Browse and select RAG methods, chunking strategies, and model/adapter combinations  
💬 **Real-time Chat**: Get immediate triage decisions and next step recommendations  
📊 **Performance Metrics**: View processing time, memory usage, and confidence scores  
🎯 **Live Switching**: Change configurations mid-session to compare results  

## Quick Start

```bash
# Run the interactive chat interface
cd /Users/choemanseung/789/hft/evaluation_framework/chat
python3 interactive_triage_chat.py
```

## Usage Example

```
🏥 Medical Triage Interactive Chat Interface
============================================================
📊 Loading optimized configurations...
✅ Loaded 50 configurations for testing

🔧 Available Configurations:
--------------------------------------------------------------------------------

📊 STRUCTURED_AGENT_TINFOIL_MEDICAL RAG METHOD:
   1. structured_agent_tinfoil_medical + sentence_transformers + SmolLM2-360M
      ID: S360_1_18f3c334
      Model: SmolLM2-360M
      Chunking: sentence_transformers

🎯 Select configuration (1-25) or 'q' to quit:
➤ 1

🚀 Initializing pipeline with configuration: S360_1_18f3c334
   RAG Method: structured_agent_tinfoil_medical
   Chunking: sentence_transformers
   Model: SmolLM2-360M
✅ Pipeline initialized successfully!

💬 Chat session started with S360_1_18f3c334
Type your medical query or 'quit' to exit, 'config' to change configuration
--------------------------------------------------------------------------------

🩺 Medical Query: I have severe chest pain and shortness of breath

🔍 Processing query with S360_1_18f3c334...
Query: I have severe chest pain and shortness of breath

============================================================
🏥 MEDICAL TRIAGE RESULT
============================================================
🎯 Triage Decision: ED
📋 Next Steps: Immediate emergency department evaluation - call 911 or go to nearest emergency room
🎲 Confidence: 95.00%

------------------------------------------------------------
⚙️  TECHNICAL DETAILS
------------------------------------------------------------
RAG Method: structured_agent_tinfoil_medical
Model: SmolLM2-360M
Processing Time: 500ms
Memory Usage: 150MB
============================================================
```

## Available Commands

- **Medical Query**: Type any medical symptoms or concerns
- **`config`**: Change to a different configuration
- **`quit`** / **`exit`** / **`q`**: Exit the chat session
- **Ctrl+C**: Emergency exit

## Configuration Categories

The interface organizes the 600 configurations by:

### RAG Methods
- `structured_agent_tinfoil_medical`: Advanced contextual retrieval
- `hybrid_retrieval`: BM25 + semantic hybrid approach  
- `contextual_rag`: Context-aware chunking and retrieval
- `multi_source`: Multiple medical data source integration
- `bm25_semantic`: BM25 + semantic reranking

### Chunking Methods
- `sentence_transformers`: Semantic sentence-based chunking
- `recursive_character`: Recursive text splitting
- `semantic_chunking`: Meaning-preserving chunks
- `fixed_size`: Fixed character-length chunks
- `document_aware`: Document structure-aware chunking

### Models & Adapters
- **SmolLM2-360M**: Lightweight, fast inference
- **SmolLM2-1.7B**: Balanced performance and speed
- **Qwen2.5-3B**: Higher accuracy, more parameters

## Sample Medical Queries

Try these example queries to test different configurations:

### Emergency Cases
- "I have severe chest pain and shortness of breath"
- "I think I'm having a stroke - slurred speech and weakness"
- "Severe abdominal pain, vomiting blood"

### Primary Care Cases  
- "I've had a fever and cough for 3 days"
- "Persistent headaches for a week"
- "Ankle pain after a fall yesterday"

### Home Care Cases
- "Minor cut that won't stop bleeding"
- "Mild sore throat and runny nose"
- "Muscle soreness after exercise"

## Technical Integration

The chat interface integrates with:

- **OptimizedConfigMatrixGenerator**: Loads the 600 optimized configurations
- **EnhancedEvaluationPipeline**: Processes queries through the selected configuration
- **IntegratedRAGSystem**: Handles retrieval and context generation
- **Model/Adapter Loading**: Dynamically loads the selected model and adapter

## Limitations

- Currently shows first 50 configurations (demo mode)
- Mock processing for demonstration (replace with actual pipeline integration)
- Simplified confidence scoring
- Single-query processing (not conversational)

## Next Steps

To integrate with the full evaluation pipeline:

1. Replace `_mock_process_case()` with actual pipeline integration
2. Add full 600 configuration loading
3. Implement real-time RAG retrieval
4. Add conversation history tracking
5. Include clinical appropriateness scoring