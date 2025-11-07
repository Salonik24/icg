

from google import genai

import os
import json
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
import pickle
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache directory for embeddings and index
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache file paths
CACHE_INDEX_FILE = CACHE_DIR / "faiss_index.bin"
CACHE_EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"
CACHE_CHUNKS_FILE = CACHE_DIR / "chunks.pkl"
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"
CACHE_MODEL_FILE = CACHE_DIR / "model_name.txt"

# ---------------------------------------------------------------------
# Cache Management Functions
# ---------------------------------------------------------------------

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file for change detection."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return None

def get_file_metadata(file_paths):
    """Get metadata (hash and mtime) for all files."""
    metadata = {}
    for file_path in file_paths:
        path = Path(file_path).resolve()
        if path.exists():
            metadata[str(path)] = {
                "hash": compute_file_hash(path),
                "mtime": path.stat().st_mtime,
                "size": path.stat().st_size
            }
    return metadata

def load_cache_metadata():
    """Load cache metadata if it exists."""
    if CACHE_METADATA_FILE.exists():
        try:
            with open(CACHE_METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
    return None

def save_cache_metadata(file_paths):
    """Save metadata for files that were used to create the cache."""
    metadata = {
        "file_metadata": get_file_metadata(file_paths),
        "created_at": datetime.now().isoformat(),
        "model_name": None  # Will be set when saving
    }
    try:
        with open(CACHE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Cache metadata saved successfully")
    except Exception as e:
        logger.error(f"Error saving cache metadata: {e}")

def files_have_changed(file_paths, cached_metadata):
    """Check if any source files have changed since cache was created."""
    if not cached_metadata or "file_metadata" not in cached_metadata:
        logger.info("No cached metadata found")
        return True
    
    current_metadata = get_file_metadata(file_paths)
    cached_file_metadata = cached_metadata["file_metadata"]
    
    # Normalize cached paths for comparison
    cached_paths_normalized = {str(Path(k).resolve()): v for k, v in cached_file_metadata.items()}
    
    # Check if file list has changed
    if set(current_metadata.keys()) != set(cached_paths_normalized.keys()):
        logger.info("File list has changed")
        return True
    
    # Check if any file has changed
    for file_path, current_info in current_metadata.items():
        if file_path not in cached_paths_normalized:
            logger.info(f"New file detected: {file_path}")
            return True
        
        cached_info = cached_paths_normalized[file_path]
        if current_info["hash"] != cached_info.get("hash"):
            logger.info(f"File changed (hash mismatch): {file_path}")
            return True
    
    logger.info("All files unchanged, can use cache")
    return False

def save_embeddings_and_index(index, embeddings, chunks, model_name, file_paths):
    """Save embeddings, FAISS index, chunks, and metadata to disk."""
    try:
        logger.info("Saving embeddings and index to cache...")
        
        # Save FAISS index
        faiss.write_index(index, str(CACHE_INDEX_FILE))
        logger.info(f"FAISS index saved to {CACHE_INDEX_FILE}")
        
        # Save embeddings
        np.save(str(CACHE_EMBEDDINGS_FILE), embeddings)
        logger.info(f"Embeddings saved to {CACHE_EMBEDDINGS_FILE}")
        
        # Save chunks
        with open(CACHE_CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Chunks saved to {CACHE_CHUNKS_FILE}")
        
        # Save model name
        with open(CACHE_MODEL_FILE, "w") as f:
            f.write(model_name)
        logger.info(f"Model name saved: {model_name}")
        
        # Save metadata
        metadata = {
            "file_metadata": get_file_metadata(file_paths),
            "created_at": datetime.now().isoformat(),
            "model_name": model_name,
            "num_vectors": index.ntotal,
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        }
        with open(CACHE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Cache metadata saved successfully")
        
        logger.info("✅ All cache files saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        raise

def load_embeddings_and_index():
    """Load embeddings, FAISS index, and chunks from disk."""
    try:
        logger.info("Loading embeddings and index from cache...")
        
        # Check if all cache files exist
        if not all([
            CACHE_INDEX_FILE.exists(),
            CACHE_EMBEDDINGS_FILE.exists(),
            CACHE_CHUNKS_FILE.exists(),
            CACHE_MODEL_FILE.exists()
        ]):
            logger.info("Cache files not found, need to regenerate")
            return None, None, None, None
        
        # Load FAISS index
        index = faiss.read_index(str(CACHE_INDEX_FILE))
        logger.info(f"FAISS index loaded: {index.ntotal} vectors")
        
        # Load embeddings
        embeddings = np.load(str(CACHE_EMBEDDINGS_FILE))
        logger.info(f"Embeddings loaded: shape {embeddings.shape}")
        
        # Load chunks
        with open(CACHE_CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Chunks loaded: {len(chunks)} chunks")
        
        # Load model name
        with open(CACHE_MODEL_FILE, "r") as f:
            model_name = f.read().strip()
        logger.info(f"Model name: {model_name}")
        
        logger.info("✅ All cache files loaded successfully")
        return index, embeddings, chunks, model_name
        
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None, None, None, None

# ---------------------------------------------------------------------
# Utility: Load all JSON files and normalize into tables
# ---------------------------------------------------------------------

def load_tables_from_files(file_paths):
    """
    Loads structured or list-style JSON files and converts them into DataFrames.

    Each file may contain:
      - Dict with keys 'title', 'description', 'table'
      - Or just a list of row dictionaries (no metadata)

    Args:
        file_paths (list[str]): Paths to JSON files.

    Returns:
        list[dict]: Each dict contains title, description, dataframe, and metadata.
    """
    logger.info(f"Step 1: Loading tables from {len(file_paths)} files...")
    all_tables = []

    for path in file_paths:
        path = Path(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both dict-style and list-style JSONs
            if isinstance(data, list):
                df = pd.DataFrame(data)
                title = path.stem
                description = f"Data extracted from {path.name}"
            elif isinstance(data, dict) and "table" in data:
                df = pd.DataFrame(data["table"])
                title = data.get("title", path.stem)
                description = data.get("description", "")
            else:
                raise ValueError("Unsupported JSON structure")

            if df.empty:
                logger.warning(f"'{path.name}' contained an empty table, skipped.")
                continue

            all_tables.append({
                "title": title,
                "description": description,
                "dataframe": df,
                "source_file": path.name
            })
            logger.info(f"✅ Loaded '{path.name}' ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Error loading '{path.name}': {e}")

    if not all_tables:
        logger.warning("No valid tables loaded. Check file paths or formats.")
    return all_tables


# ---------------------------------------------------------------------
# Step 2: Create Chunks
# ---------------------------------------------------------------------

def create_chunks(tables):
    """
    Converts each row of each table into a serialized textual chunk with metadata.
    """
    logger.info("Step 2: Creating row-based chunks with metadata...")
    chunks = []

    for table in tables:
        df = table["dataframe"]
        for row_idx, row in df.iterrows():
            serialized_text = "; ".join(f"{col}: {val}" for col, val in row.items())

            metadata = {
                "source_file": table["source_file"],
                "table_title": table["title"],
                "description": table["description"],
                "row_index": row_idx,
                "columns": df.columns.tolist(),
                "row_data": row.to_dict(),
            }

            chunks.append({"serialized_text": serialized_text, "metadata": metadata})

    logger.info(f"✅ Created {len(chunks)} total chunks from {len(tables)} files.")
    return chunks


# ---------------------------------------------------------------------
# Step 3: Embedding & Indexing
# ---------------------------------------------------------------------

def embed_and_index(chunks, model_name='all-MiniLM-L6-v2', file_paths=None, use_cache=True):
    """
    Embeds all text chunks and builds a FAISS index for semantic retrieval.
    Uses cache if available and files haven't changed.
    
    Args:
        chunks: List of chunks to embed
        model_name: Name of the sentence transformer model
        file_paths: List of source file paths (for cache validation)
        use_cache: Whether to use cache if available
    
    Returns:
        tuple: (index, model, embeddings, chunks)
    """
    logger.info("Step 3: Embedding chunks and building FAISS index...")

    if not chunks:
        raise ValueError("No chunks provided to embed_and_index().")

    # Try to load from cache if enabled
    if use_cache and file_paths:
        cached_metadata = load_cache_metadata()
        
        if cached_metadata and not files_have_changed(file_paths, cached_metadata):
            # Check if model name matches
            if cached_metadata.get("model_name") == model_name:
                logger.info("Loading embeddings and index from cache...")
                index, embeddings, cached_chunks, cached_model_name = load_embeddings_and_index()
                
                if index is not None and embeddings is not None and cached_chunks is not None:
                    # Verify model name matches
                    if cached_model_name == model_name:
                        logger.info("✅ Using cached embeddings and index")
                        model = SentenceTransformer(model_name)
                        return index, model, embeddings, cached_chunks
                    else:
                        logger.warning(f"Model mismatch: cached={cached_model_name}, requested={model_name}. Regenerating...")
                else:
                    logger.info("Cache load failed, will regenerate embeddings")
            else:
                logger.info(f"Model name changed: cached={cached_metadata.get('model_name')}, requested={model_name}. Regenerating...")
        else:
            logger.info("Files have changed or cache not found, will regenerate embeddings")

    # Generate new embeddings
    logger.info(f"Generating embeddings using model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [chunk["serialized_text"] for chunk in chunks]
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    logger.info(f"✅ FAISS index built with {index.ntotal} vectors (dimension: {embeddings.shape[1]})")
    
    # Save to cache if file_paths provided
    if file_paths:
        try:
            save_embeddings_and_index(index, embeddings, chunks, model_name, file_paths)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    return index, model, embeddings, chunks


# ---------------------------------------------------------------------
# Step 4: Retrieval
# ---------------------------------------------------------------------

def retrieve_results(query, index, model, chunks, top_k=3):
    """
    Retrieves top-k relevant chunks for the given user query.
    """
    logger.info(f"Step 4: Retrieving top {top_k} results for: '{query}'")

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Run embed_and_index() first.")

    logger.debug(f"Encoding query: {query}")
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    retrieved = [chunks[i] for i in indices[0]]
    logger.info(f"Retrieved {len(retrieved)} chunks:")
    for idx, chunk in enumerate(retrieved):
        distance = distances[0][idx] if len(distances) > 0 and len(distances[0]) > idx else "N/A"
        logger.info(f"  🔹 [{idx+1}] From {chunk['metadata']['source_file']} | Row {chunk['metadata']['row_index']} | Distance: {distance:.4f}")

    return retrieved


# ---------------------------------------------------------------------
# Step 5: Prompt Assembly
# ---------------------------------------------------------------------

def generate_llm_prompt(retrieved_chunks, query):
    """
    Builds a human-readable prompt combining retrieved context and the user query.
    """
    logger.info("Step 5: Generating final LLM prompt...")

    if not retrieved_chunks:
        logger.warning("No chunks retrieved for prompt generation")
        return f"User Question: {query}\n\nNo relevant context found."

    grouped_context = ""
    for chunk in retrieved_chunks:
        m = chunk["metadata"]
        grouped_context += (
            f"\nFrom '{m['source_file']}' — Table: '{m['table_title']}':\n"
            + "; ".join(f"{k}: {v}" for k, v in m["row_data"].items()) + "\n"
        )

    prompt = f"""
You are a factual AI assistant. Use the context below to answer the user's question.

--- CONTEXT ---
{grouped_context}
--- QUESTION ---
{query}

Answer:
"""
    logger.debug(f"Generated prompt length: {len(prompt)} characters")
    return prompt.strip()


# ---------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------



# Option 1: Pass API key directly to the function
def get_llm_answer(prompt, api_key=None, model="gemini-2.0-flash"):
    """
    Gets answer from LLM (Gemini) for the given prompt.
    """
    logger.info(f"Step 6: Getting answer from LLM (model: {model})...")
    try:
        client = genai.Client(api_key=api_key)
        logger.debug(f"Sending prompt to LLM (length: {len(prompt)} characters)")
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        logger.info("✅ Received response from LLM")
        return response.text
    except Exception as e:
        logger.error(f"Error getting LLM answer: {e}")
        raise

# In main execution:
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting LLM RAG Pipeline")
    logger.info("=" * 60)
    
    # Define your API key
    GEMINI_API_KEY = "AIzaSyA7oR7AJEDwQYDMMQGjHc1oLve_BUkC-h4"
    
    FILES_TO_PROCESS = [
        "punjab.json",
        "all_india.json",
        "andhra_pradesh.json",
        "bihar.json",
        "UP.json",
        "MP.json"
    ]

    USER_QUERY = "what percentage of children are enrolled in government primary schools in india in 2024"

    logger.info(f"Processing {len(FILES_TO_PROCESS)} files")
    tables = load_tables_from_files(FILES_TO_PROCESS)
    logger.info(f"Loaded {len(tables)} tables")
    
    if tables:
        chunks = create_chunks(tables)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Use cache-aware embedding function
        index, model, embeddings, chunks = embed_and_index(
            chunks, 
            model_name='all-MiniLM-L6-v2',
            file_paths=FILES_TO_PROCESS,
            use_cache=True
        )
        
        retrieved = retrieve_results(USER_QUERY, index, model, chunks)
        final_prompt = generate_llm_prompt(retrieved, USER_QUERY)
        
        logger.info("Calling LLM for answer...")
        answer = get_llm_answer(final_prompt, api_key=GEMINI_API_KEY)
        
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        print("\n--- FINAL LLM PROMPT ---")
        print(final_prompt)
        print("\n--- GEMINI RESPONSE ---")
        print(answer)
        logger.info("Pipeline completed successfully")
