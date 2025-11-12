import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Use the same constants and model as the build script.
FAISS_INDEX_PATH = ["RAGdatav2/nhs_vector_db.index", "RAGdatav2/mayo_vector_db.index", "RAGdatav2/healthify_vector_db.index"]
CHUNKS_DATA_PATH = ["RAGdatav2/nhs_chunks.json", "RAGdatav2/mayo_chunks.json", "RAGdatav2/healthify_chunks.json"]

class RagRetriever:
    """
    A class to handle loading the RAG database and efficiently retrieving context.
    Compatible with both old format (list of strings) and new format (list of objects with metadata).
    """
    def __init__(self, model_name, index_path, chunks_path, source_name="Unknown"):
        """
        Initializes the retriever by loading the model, FAISS index, and chunks.
        
        Args:
            model_name (str): Name of the SentenceTransformer model
            index_path (str): Path to the FAISS index file
            chunks_path (str): Path to the JSON file containing text chunks
            source_name (str): Name of the data source (e.g., "NHS", "Mayo")
        """
        print(f"Initializing RagRetriever for {source_name}...")
        self.source_name = source_name
        self.model = SentenceTransformer(model_name)
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Loading text chunks from {chunks_path}...")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            
        # Handle both old format (list of strings) and new format (list of objects with metadata)
        if chunks_data and isinstance(chunks_data[0], dict):
            # New format: extract text from objects
            self.chunks = [chunk['text'] for chunk in chunks_data]
            self.chunk_metadata = chunks_data  # Store full metadata for reference
            print(f"Using new format with metadata (chunk objects)")
        else:
            # Old format: list of strings
            self.chunks = chunks_data
            self.chunk_metadata = None
            print(f"Using old format (plain text chunks)")
            
        print(f"{source_name} retriever is ready with {len(self.chunks)} chunks.")

    def fetch_context(self, query, k=5):
        """
        Fetches the top k most relevant text chunks for a given query.

        Args:
            query (str): The user's input query.
            k (int): The number of relevant chunks to retrieve.

        Returns:
            dict: Contains the context string, distances, source info, and metadata if available.
        """
        # 1. Generate the embedding for the user's query
        query_embedding = self.model.encode([query]).astype('float32')

        # 2. Perform the search using the FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # 3. Retrieve the original text chunks using the indices
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        
        # 4. Concatenate the chunks into a single context string
        context = f"\n--- {self.source_name} Medical Information ---\n".join(retrieved_chunks)
        
        # 5. Get metadata if available
        retrieved_metadata = None
        if self.chunk_metadata:
            retrieved_metadata = [self.chunk_metadata[i] for i in indices[0]]
        
        return {
            'context': context,
            'distances': distances[0].tolist(),
            'source': self.source_name,
            'num_chunks': len(retrieved_chunks),
            'metadata': retrieved_metadata
        }


class MultiSourceRagRetriever:
    """
    A class to handle multiple RAG databases and retrieve context from all sources.
    Compatible with both old and new chunk formats.
    """
    def __init__(self, model_name, index_paths, chunks_paths, source_names):
        """
        Initializes multiple retrievers for different data sources.
        
        Args:
            model_name (str): Name of the SentenceTransformer model
            index_paths (list): List of paths to FAISS index files
            chunks_paths (list): List of paths to JSON chunk files
            source_names (list): List of source names
        """
        print("Initializing Multi-Source RAG Retriever...")
        self.retrievers = {}
        
        for i, source_name in enumerate(source_names):
            self.retrievers[source_name] = RagRetriever(
                model_name=model_name,
                index_path=index_paths[i],
                chunks_path=chunks_paths[i],
                source_name=source_name
            )
        
        print(f"Multi-source retriever ready with {len(self.retrievers)} sources: {list(self.retrievers.keys())}")

    def fetch_context_from_all_sources(self, query, k=3):
        """
        Fetches context from all available sources and combines them.

        Args:
            query (str): The user's input query
            k (int): Number of chunks to retrieve from each source

        Returns:
            dict: Combined results from all sources
        """
        print(f"Fetching context from all {len(self.retrievers)} sources...")
        individual_results = {}
        all_contexts = []
        
        for source_name, retriever in self.retrievers.items():
            print(f"  Querying {source_name}...")
            result = retriever.fetch_context(query, k)
            individual_results[source_name] = result
            all_contexts.append(result['context'])
        
        # Combine all contexts
        combined_context = "\n\n".join(all_contexts)
        
        return {
            'combined_context': combined_context,
            'individual_results': individual_results,
            'total_chunks': sum([result['num_chunks'] for result in individual_results.values()])
        }

    def fetch_context_from_source(self, query, source_name, k=4):
        """
        Fetches context from a specific source.

        Args:
            query (str): The user's input query
            source_name (str): Name of the source to query
            k (int): Number of chunks to retrieve

        Returns:
            dict: Results from the specified source
        """
        if source_name not in self.retrievers:
            return {'error': f'Source "{source_name}" not found. Available sources: {list(self.retrievers.keys())}'}
        
        print(f"Fetching context from {source_name}...")
        result = self.retrievers[source_name].fetch_context(query, k)
        return result