
import sqlite3
import json
import os
from pathlib import Path

def create_database(db_path: str, chunk_files: dict):
    """
    Creates a SQLite database from a list of JSON chunk files.

    Args:
        db_path (str): The full path to the SQLite database file to be created.
        chunk_files (dict): A dictionary where keys are source names ('healthify', 'mayo', 'nhs')
                            and values are paths to the corresponding JSON chunk files.
    """
    print(f"Creating database at: {db_path}")

    # Delete existing database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create the main 'chunks' table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            url TEXT,
            headings_path TEXT
        )
        ''')
        print("Created 'chunks' table.")

        # Create the FTS5 virtual table for full-text search
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        )
        ''')
        print("Created 'chunks_fts' virtual table.")

        # Create a trigger to keep the FTS table in sync with the chunks table
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        ''')
        print("Created 'chunks_after_insert' trigger.")

        total_chunks_processed = 0

        # Process each JSON file
        for source, file_path in chunk_files.items():
            print(f"\nProcessing source: {source} from {file_path}")
            if not os.path.exists(file_path):
                print(f"⚠️  Warning: File not found, skipping: {file_path}")
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            chunks_to_insert = []
            for item in data:
                # The 'text' field already contains the structured summary
                chunk_text = item.get("text", "")
                chunk_id = item.get("chunk_id", "")
                
                # Extract title from the structured data, fallback to source document
                title = item.get("structured_data", {}).get("condition")
                if not title:
                    title = item.get("source_document", "Unknown")

                # Ensure required fields are not empty
                if not chunk_text or not chunk_id:
                    print(f"Skipping chunk with missing text or id: {item}")
                    continue

                chunks_to_insert.append((
                    chunk_id,
                    source,
                    title,
                    chunk_text,
                    "",  # url - not available
                    ""   # headings_path - not available
                ))
            
            # Bulk insert the chunks for the current source
            cursor.executemany(
                "INSERT INTO chunks (chunk_id, source, title, text, url, headings_path) VALUES (?, ?, ?, ?, ?, ?)",
                chunks_to_insert
            )
            print(f"Inserted {len(chunks_to_insert)} chunks from {source}.")
            total_chunks_processed += len(chunks_to_insert)

        # Rebuild the FTS index
        print("\nRebuilding FTS index...")
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        
        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        print(f"\n✅ Database creation complete.")
        print(f"   Total chunks processed: {total_chunks_processed}")
        print(f"   Database saved to: {db_path}")

    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from a file: {e}")
    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Define the base directory
    base_dir = Path("/Users/choemanseung/789/hft")
    
    # Define the output path for the SQLite database
    output_dir = base_dir / "iosDemo"
    output_dir.mkdir(exist_ok=True)
    db_file_path = output_dir / "chunks.sqlite"

    # Define the paths to the pre-computed chunk files
    chunk_files = {
        "healthify": base_dir / "RAGdatav4/healthify_chunks_structured_agent_tinfoil_medical.json",
        "mayo": base_dir / "RAGdatav4/mayo_chunks_structured_agent_tinfoil_medical.json",
        "nhs": base_dir / "RAGdatav4/nhs_chunks_structured_agent_tinfoil_medical.json"
    }

    # Run the database creation process
    create_database(str(db_file_path), chunk_files)
