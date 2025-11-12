import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Use the same constants and model as the build script.
FAISS_INDEX_PATH = ["RAGdata/nhs_vector_db.index", "RAGdata/mayo_vector_db.index"]
CHUNKS_DATA_PATH = ["RAGdata/nhs_chunks.json", "RAGdata/mayo_chunks.json"]
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


class RagRetriever:
    """
    A class to handle loading the RAG database and efficiently retrieving context.
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
            self.chunks = json.load(f)
        print(f"{source_name} retriever is ready with {len(self.chunks)} chunks.")

    def fetch_context(self, query, k=5):
        """
        Fetches the top k most relevant text chunks for a given query.

        Args:
            query (str): The user's input query.
            k (int): The number of relevant chunks to retrieve.

        Returns:
            dict: Contains the context string, distances, and source info.
        """
        # 1. Generate the embedding for the user's query
        query_embedding = self.model.encode([query]).astype('float32')

        # 2. Perform the search using the FAISS index
        # The `search` method returns two numpy arrays:
        # D: Distances (L2 distance in our case) to the k nearest neighbors.
        # I: Indices of the k nearest neighbors.
        distances, indices = self.index.search(query_embedding, k)

        # 3. Retrieve the original text chunks using the indices
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        
        # 4. Concatenate the chunks into a single context string
        context = f"\n--- {self.source_name} Medical Information ---\n".join(retrieved_chunks)
        
        return {
            'context': context,
            'distances': distances[0].tolist(),
            'source': self.source_name,
            'num_chunks': len(retrieved_chunks)
        }


class MultiSourceRagRetriever:
    """
    A class to handle multiple RAG databases and retrieve context from all sources.
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
            dict: Combined context and metadata from all sources
        """
        all_results = {}
        combined_context = []
        
        for source_name, retriever in self.retrievers.items():
            result = retriever.fetch_context(query, k=k)
            all_results[source_name] = result
            
            # Add source header and context
            source_context = f"\n=== {source_name.upper()} INFORMATION ===\n{result['context']}\n"
            combined_context.append(source_context)
        
        return {
            'combined_context': '\n'.join(combined_context),
            'individual_results': all_results,
            'total_sources': len(self.retrievers),
            'query': query
        }

    def fetch_context_from_source(self, query, source_name, k=5):
        """
        Fetches context from a specific source only.
        
        Args:
            query (str): The user's input query
            source_name (str): Name of the source to search
            k (int): Number of chunks to retrieve
            
        Returns:
            dict: Context and metadata from the specified source
        """
        if source_name not in self.retrievers:
            return {
                'error': f"Source '{source_name}' not available. Available sources: {list(self.retrievers.keys())}"
            }
        
        return self.retrievers[source_name].fetch_context(query, k=k)


# --- Simple Example Usage ---

if __name__ == "__main__":
    # Simple example of how to use the retriever
    multi_retriever = MultiSourceRagRetriever(
        model_name=EMBEDDING_MODEL_NAME,
        index_paths=FAISS_INDEX_PATH,
        chunks_paths=CHUNKS_DATA_PATH,
        source_names=["NHS", "Mayo"]
    )
    
    print("\n" + "="*50)
    print("BASIC RETRIEVAL EXAMPLE")
    print("="*50)
    
    # Example query
    query = "What are the symptoms of diabetes?"
    print(f"Query: '{query}'")
    
    # Get results from all sources
    result = multi_retriever.fetch_context_from_all_sources(query, k=2)
    print("\nRetrieved Context:")
    print(result['combined_context'][:600] + "..." if len(result['combined_context']) > 600 else result['combined_context'])
    
    print(f"\nTotal sources searched: {result['total_sources']}")
    print("="*50)
    print("For comprehensive testing, run: python retrieval_test.py")