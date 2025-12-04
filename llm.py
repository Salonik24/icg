import google.generativeai as genai
import os
import json
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
from dotenv import load_dotenv, find_dotenv  # ✅ FIXED: Added find_dotenv
import hashlib
from datetime import datetime
import time

# Load environment variables with auto-detection
load_dotenv(find_dotenv(), override=True)

# Verify API key before proceeding
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ GOOGLE_API_KEY not found!\n"
        "Please ensure your .env file contains:\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
print(f"✓ API Key loaded successfully (ends with: ...{GOOGLE_API_KEY[-8:]})")

# Configure logging - Set to WARNING to reduce terminal clutter
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8',
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

# =====================================================================
# Cache Management Functions
# =====================================================================

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

def save_cache_metadata(file_paths, model_name):
    """Save metadata for files that were used to create the cache."""
    metadata = {
        "file_metadata": get_file_metadata(file_paths),
        "created_at": datetime.now().isoformat(),
        "model_name": model_name
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
    
    cached_paths_normalized = {str(Path(k).resolve()): v for k, v in cached_file_metadata.items()}
    
    if set(current_metadata.keys()) != set(cached_paths_normalized.keys()):
        logger.info("File list has changed")
        return True
    
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
        
        faiss.write_index(index, str(CACHE_INDEX_FILE))
        logger.info(f"FAISS index saved to {CACHE_INDEX_FILE}")
        
        np.save(str(CACHE_EMBEDDINGS_FILE), embeddings)
        logger.info(f"Embeddings saved to {CACHE_EMBEDDINGS_FILE}")
        
        with open(CACHE_CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Chunks saved to {CACHE_CHUNKS_FILE}")
        
        with open(CACHE_MODEL_FILE, "w") as f:
            f.write(model_name)
        logger.info(f"Model name saved: {model_name}")
        
        metadata = {
            "file_metadata": get_file_metadata(file_paths),
            "created_at": datetime.now().isoformat(),
            "model_name": model_name,
            "num_vectors": index.ntotal,
            "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        }
        with open(CACHE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("All cache files saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        raise

def load_embeddings_and_index():
    """Load embeddings, FAISS index, and chunks from disk."""
    try:
        logger.info("Loading embeddings and index from cache...")
        
        if not all([
            CACHE_INDEX_FILE.exists(),
            CACHE_EMBEDDINGS_FILE.exists(),
            CACHE_CHUNKS_FILE.exists(),
            CACHE_MODEL_FILE.exists()
        ]):
            logger.info("Cache files not found, need to regenerate")
            return None, None, None, None
        
        index = faiss.read_index(str(CACHE_INDEX_FILE))
        logger.info(f"FAISS index loaded: {index.ntotal} vectors")
        
        embeddings = np.load(str(CACHE_EMBEDDINGS_FILE))
        logger.info(f"Embeddings loaded: shape {embeddings.shape}")
        
        with open(CACHE_CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Chunks loaded: {len(chunks)} chunks")
        
        with open(CACHE_MODEL_FILE, "r") as f:
            model_name = f.read().strip()
        logger.info(f"Model name: {model_name}")
        
        logger.info("All cache files loaded successfully")
        return index, embeddings, chunks, model_name
        
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None, None, None, None

# =====================================================================
# Utility: Load all JSON files and normalize into tables
# =====================================================================

def load_tables_from_files(file_paths):
    """
    Loads structured or list-style JSON files and converts them into DataFrames.
    
    Each file may contain:
      - Dict with keys 'title', 'description', 'table'
      - Or just a list of row dictionaries (no metadata)
    
    Args:
        file_paths (list[str]): Paths to JSON files.
    
    Returns:
        list[dict]: Each dict contains title, description, dataframe, state, and metadata.
    """
    print(f"Loading tables from {len(file_paths)} files...")
    all_tables = []

    for path in file_paths:
        path = Path(path)
        try:
            if not path.exists():
                logger.error(f"File not found: {path}")
                continue
                
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            state_name = path.stem

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
                "source_file": path.name,
                "state": state_name
            })
            print(f"✓ Loaded '{path.name}' ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Error loading '{path.name}': {e}")

    if not all_tables:
        logger.warning("No valid tables loaded. Check file paths or formats.")
    return all_tables

# =====================================================================
# Step 2: Create Chunks
# =====================================================================

def create_chunks(tables):
    """
    Converts each row of each table into a serialized textual chunk with metadata.
    Enhanced to include state name and description in the serialized text.
    """
    print("Creating chunks...")
    chunks = []

    for table in tables:
        df = table["dataframe"]
        state_name = table["state"]
        description = table["description"]
        
        for row_idx, row in df.iterrows():
            row_data = "; ".join(f"{col}: {val}" for col, val in row.items())
            serialized_text = f"The given table gives the data for {state_name}. The description for the table is: {description}. {row_data}"

            metadata = {
                "source_file": table["source_file"],
                "table_title": table["title"],
                "description": table["description"],
                "state": table["state"],
                "row_index": row_idx,
                "columns": df.columns.tolist(),
                "row_data": row.to_dict(),
            }

            chunks.append({"serialized_text": serialized_text, "metadata": metadata})

    print(f"✓ Created {len(chunks)} chunks from {len(tables)} tables")
    return chunks

# =====================================================================
# Step 3: Embedding & Indexing with Gemini API
# =====================================================================

def embed_and_index(chunks, model_name='models/text-embedding-004', batch_size=100, file_paths=None, use_cache=True):
    """
    Embeds chunks using Gemini API and builds a FAISS index.
    Uses cache if available and files haven't changed.
    
    Args:
        chunks: List of chunks to embed
        model_name: Gemini embedding model name
        batch_size: Batch size for API calls
        file_paths: List of source file paths (for cache validation)
        use_cache: Whether to use cache if available
    
    Returns:
        tuple: (index, model_name, embeddings, chunks)
    """
    print("Embedding chunks and building FAISS index...")

    if not chunks:
        raise ValueError("No chunks provided to embed_and_index().")

    # Check cache
    if use_cache and file_paths:
        cached_metadata = load_cache_metadata()
        
        if cached_metadata and not files_have_changed(file_paths, cached_metadata):
            if cached_metadata.get("model_name") == model_name:
                print("Loading from cache...")
                index, embeddings, cached_chunks, cached_model_name = load_embeddings_and_index()
                
                if index is not None and embeddings is not None and cached_chunks is not None:
                    if cached_model_name == model_name:
                        print(f"✓ Using cached embeddings ({index.ntotal} vectors)")
                        return index, model_name, embeddings, cached_chunks
                    else:
                        logger.warning(f"Model mismatch: cached={cached_model_name}, requested={model_name}. Regenerating...")
                else:
                    print("Cache load failed, regenerating...")
            else:
                print(f"Model changed, regenerating...")
        else:
            print("Files changed or no cache, regenerating...")

    # Generate embeddings using Gemini API
    print(f"Generating embeddings with {model_name}...")
    texts = [chunk["serialized_text"] for chunk in chunks]
    all_embeddings = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Processing {len(texts)} texts in {total_batches} batches...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            print(f"  Batch {batch_num}/{total_batches}...", end=" ")
            response = genai.embed_content(
                model=model_name,
                content=batch,
                task_type="retrieval_document",
                title="Table Data"
            )
            batch_embeddings = response['embedding']
            all_embeddings.extend(batch_embeddings)
            print("✓")
            
            # Optional: minimal sleep to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error embedding batch {batch_num}: {e}")
            raise e

    # Convert to numpy array
    embeddings_np = np.array(all_embeddings).astype('float32')

    # Initialize FAISS
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    print(f"✓ FAISS index built: {index.ntotal} vectors (dim: {dimension})")
    
    # Save to cache
    if file_paths:
        try:
            print("Saving to cache...")
            save_embeddings_and_index(index, embeddings_np, chunks, model_name, file_paths)
            print("✓ Cache saved")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    return index, model_name, embeddings_np, chunks

# =====================================================================
# Step 4: Retrieval with Gemini Embeddings
# =====================================================================

def retrieve_results(query, index, model_name, chunks, top_k=3):
    """
    Retrieves top-k relevant chunks for the given user query using Gemini Embeddings.
    """
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Run embed_and_index() first.")

    try:
        response = genai.embed_content(
            model=model_name,
            content=query,
            task_type="retrieval_query"
        )
        query_emb = np.array([response['embedding']]).astype('float32')
        
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        return []

    # Search FAISS
    distances, indices = index.search(query_emb, top_k)

    retrieved = []
    for idx_pos, idx in enumerate(indices[0]):
        if idx != -1:  # FAISS returns -1 if fewer than k results found
            chunk = chunks[idx]
            retrieved.append(chunk)

    return retrieved

# =====================================================================
# Step 5: Prompt Assembly
# =====================================================================

def generate_llm_prompt(retrieved_chunks, query):
    """
    Builds a human-readable prompt combining retrieved context and the user query.
    """
    if not retrieved_chunks:
        logger.warning("No chunks retrieved for prompt generation")
        return f"User Question: {query}\n\nNo relevant context found."

    grouped_context = ""
    for chunk in retrieved_chunks:
        m = chunk["metadata"]
        grouped_context += (
            f"\nFrom '{m['source_file']}' — Table: '{m['table_title']}' — State: '{m['state']}':\n"
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
    return prompt.strip()

# =====================================================================
# Step 6: LLM Answer Generation (NEW - MISSING FUNCTION ADDED)
# =====================================================================

def get_llm_answer(prompt, model_name="gemini-2.5-flash"):
    """
    Generates an answer from the LLM given a prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM
        model_name (str): The Gemini model to use
        
    Returns:
        str: The generated response text
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return f"Error generating response: {str(e)}"

# =====================================================================
# Step 7: Query Complexity Detection
# =====================================================================

def is_complex_query(query):
    """
    Determines if a query is complex and needs decomposition.
    Complex queries typically involve comparisons, multiple conditions, or aggregations.
    """
    complexity_keywords = [
        'compare', 'comparison', 'difference', 'versus', 'vs', 'between',
        'both', 'and', 'contrast', 'how has', 'trend', 'change',
        'multiple', 'each', 'all', 'different', 'various'
    ]
    
    query_lower = query.lower()
    is_complex = any(keyword in query_lower for keyword in complexity_keywords)
    
    return is_complex

def decompose_query(query, model_answer):
    """
    Uses LLM to decompose a complex query into simpler sub-queries.
    Returns a list of sub-queries.
    """
    print("Decomposing complex query...")
    
    decomposition_prompt = f"""
You are a query decomposition expert. Break down the following complex query into 2-5 simple, independent sub-queries that can be answered individually.

Rules:
1. Each sub-query should be self-contained and answerable from a single data source
2. Sub-queries should be specific and focused
3. Return ONLY a JSON array of strings (the sub-queries), nothing else
4. Do not include explanations or markdown formatting

Complex Query: {query}

Output format example:
["sub-query 1", "sub-query 2", "sub-query 3"]
"""
    
    try:
        response = model_answer.generate_content(decomposition_prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[len('```json'):]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        sub_queries = json.loads(response_text.strip())
        
        print(f"✓ Decomposed into {len(sub_queries)} sub-queries")
        
        return sub_queries
        
    except Exception as e:
        logger.error(f"Error in query decomposition: {e}")
        print("Falling back to simple query processing")
        return [query]

def answer_sub_query(sub_query, index, model_name, chunks, model_answer):
    """
    Processes a single sub-query and returns the answer.
    """
    retrieved = retrieve_results(sub_query, index, model_name, chunks, top_k=3)
    prompt = generate_llm_prompt(retrieved, sub_query)
    
    response = model_answer.generate_content(prompt)
    answer = response.text.strip()
    
    return {
        "sub_query": sub_query,
        "answer": answer,
        "retrieved_chunks": retrieved
    }

def combine_answers(original_query, sub_query_results, model_answer):
    """
    Combines answers from multiple sub-queries into a final coherent answer.
    """
    print("Combining sub-answers...")
    
    # Build context from all sub-query results
    combined_context = ""
    for i, result in enumerate(sub_query_results, 1):
        combined_context += f"\nSub-question {i}: {result['sub_query']}\n"
        combined_context += f"Answer: {result['answer']}\n"
    
    combiner_prompt = f"""
You are an expert at synthesizing information. Given the original complex question and answers to its sub-questions, provide a comprehensive, coherent answer to the original question.

Original Question: {original_query}

Sub-questions and their answers:
{combined_context}

Instructions:
1. Synthesize the sub-answers into a single coherent response
2. Directly address the original question
3. Highlight comparisons, trends, or patterns if relevant
4. Be concise but complete
5. If sub-answers conflict, acknowledge the discrepancy

Final Answer:
"""
    
    response = model_answer.generate_content(combiner_prompt)
    final_answer = response.text.strip()
    
    return final_answer

# =====================================================================
# Step 8: Main Query Processing
# =====================================================================

def process_query(query, index, model_name, chunks, model_answer):
    """
    Main query processing function that handles both simple and complex queries.
    """
    # Check if query is complex
    if is_complex_query(query):
        print(f"Processing complex query: {query[:50]}...")
        # Decompose into sub-queries
        sub_queries = decompose_query(query, model_answer)
        
        # Process each sub-query
        sub_query_results = []
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"  Sub-query {i}/{len(sub_queries)}: {sub_query[:50]}...")
            result = answer_sub_query(sub_query, index, model_name, chunks, model_answer)
            sub_query_results.append(result)
        
        # Combine answers
        final_answer = combine_answers(query, sub_query_results, model_answer)
        
        return {
            "query": query,
            "is_complex": True,
            "sub_queries": sub_queries,
            "sub_query_results": sub_query_results,
            "final_answer": final_answer
        }
    else:
        print(f"Processing simple query: {query[:50]}...")
        retrieved = retrieve_results(query, index, model_name, chunks, top_k=3)
        prompt = generate_llm_prompt(retrieved, query)
        response = model_answer.generate_content(prompt)
        
        return {
            "query": query,
            "is_complex": False,
            "final_answer": response.text.strip(),
            "retrieved_chunks": retrieved
        }

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Starting LLM RAG Pipeline with Gemini Embeddings")
    print("=" * 80)

    # Configure Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model_answer = genai.GenerativeModel("gemini-2.5-flash")

    # File paths
    file_paths = [
        "andhra_pradesh .json",
        "bihar .json",
        "Madhya_pradesh.json",
        "punjab .json",
        "all_india .json",
        "odisha.json",
        "Rajasthan.json",
        "Sikkim.json",
        "tamil_nadu.json",
        "telangana.json",
        "tripura.json",
        "Uttarakhand.json",
        "Uttar_Pradesh.json",
        "Arunachal_pradesh.json",
        "Assam.json",
        "Chhattisgarh.json",
        "Gujarat.json",
        "Harayan.json",
        "West Bengal_Tables.json",
        "Himachal_pradesh.json",
        "Jammu_Kashmir.json",
        "Jharkhand.json",
        "Karnataka.json",
        "Kerela.json",
        "Maharashtra.json",
        "Meghalaya.json",
        "Mizoram.json",
        "Nagaland.json"


    ]

    # Load tables
    tables = load_tables_from_files(file_paths)

    if tables:
        # Create chunks
        chunks = create_chunks(tables)

        # Embed and index with cache support
        index, model_name, embeddings, chunks = embed_and_index(
            chunks,
            model_name='models/text-embedding-004',
            batch_size=100,
            file_paths=file_paths,
            use_cache=True
        )

        # Example queries
        queries = [
            "What are the major schemes in Andhra Pradesh?",
            "Compare the reading levels of Standard III students between Andhra Pradesh and Uttar Pradesh in 2024",
            "What percentage of children are enrolled in government schools in 2024?"
        ]

        # Process each query
        for query in queries:
            result = process_query(query, index, model_name, chunks, model_answer)
            
            print("\n" + "=" * 80)
            print("FINAL RESULT")
            print("=" * 80)
            print(f"Query: {result['query']}")
            print(f"Complex Query: {result['is_complex']}")
            
            if result['is_complex']:
                print(f"\nSub-queries processed: {len(result['sub_queries'])}")
                for i, sq in enumerate(result['sub_queries'], 1):
                    print(f"  {i}. {sq}")
            
            print(f"\nFinal Answer:\n{result['final_answer']}")
            print("=" * 80 + "\n")

        print("Pipeline completed successfully")
    else:
        logger.error("No tables were loaded. Halting execution.")
