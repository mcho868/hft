








import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

def create_faiss_index(db_path: str, output_dir: str):
    """
    Generates embeddings from a SQLite database and creates a FAISS index.

    Args:
        db_path (str): Path to the SQLite database containing the chunks.
        output_dir (str): Directory to save the 'faiss.index' and 'ids.bin' files.
    """
    
    faiss_index_path = os.path.join(output_dir, "faiss.index")
    ids_path = os.path.join(output_dir, "ids.bin")

    if os.path.exists(faiss_index_path) and os.path.exists(ids_path):
        print(f"✅ FAISS index and ID file already exist. Skipping generation.")
        return

    print(f"Connecting to database: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch chunk IDs and text
        cursor.execute("SELECT id, text FROM chunks ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("❌ No data found in the 'chunks' table. Aborting.")
            return

        chunk_ids, chunk_texts = zip(*rows)
        print(f"Retrieved {len(chunk_texts)} text chunks from the database.")

        # Load the sentence transformer model
        print("Loading sentence-transformer model: all-MiniLM-L6-v2...")
        # Use a specific cache folder to avoid permission issues in some environments
        cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "sentence_transformers")
        os.makedirs(cache_folder, exist_ok=True)
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_folder)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {embedding_dim}")

        # Generate embeddings for all texts
        print("Generating embeddings for all chunks... (This may take a while)")
        embeddings = model.encode(chunk_texts, show_progress_bar=True)
        print(f"Generated {len(embeddings)} embeddings.")

        # Normalize embeddings for cosine similarity search with IndexFlatIP
        print("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)

        # Create the FAISS index
        print(f"Creating FAISS index (IndexFlatIP) with dimension {embedding_dim}...")
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Add vectors to the index
        index.add(embeddings.astype('float32'))
        print(f"Added {index.ntotal} vectors to the index.")

        # Save the FAISS index
        print(f"Saving FAISS index to: {faiss_index_path}")
        faiss.write_index(index, faiss_index_path)

        # Save the corresponding chunk IDs
        print(f"Saving chunk IDs to: {ids_path}")
        with open(ids_path, 'wb') as f:
            f.write(np.array(chunk_ids, dtype=np.uint32).tobytes())

        print("\n✅ FAISS index and ID mapping created successfully.")

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        # Provide more detailed error for sentence-transformers issues
        if "sentence-transformers" in str(e):
            print("Hint: This might be a network issue or a problem with the model cache.")
            print("Please ensure you have an internet connection and try again.")


if __name__ == "__main__":
    # Define the base directory
    base_dir = Path("/Users/choemanseung/789/hft")
    
    # Define the input database path and output directory
    demo_dir = base_dir / "iosDemo"
    db_file_path = demo_dir / "chunks.sqlite"

    if not db_file_path.exists():
        print(f"❌ Error: Database file not found at {db_file_path}")
        print("Please run 'create_chunk_database.py' first.")
    else:
        # Before running, ensure necessary libraries are installed
        try:
            import faiss
            import sentence_transformers
        except ImportError:
            print("Installing required libraries: faiss-cpu, sentence-transformers, numpy")
            os.system("pip install faiss-cpu sentence-transformers numpy")
        
        create_faiss_index(str(db_file_path), str(demo_dir))
