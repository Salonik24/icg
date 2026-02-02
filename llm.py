import os
import json
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------
# Configure Gemini API
# -------------------------
# Ensure you have your API Key set in your .env file or environment
# defaults to the key in the original file if not found in env
api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCvPJe-wKe-w_I3IYDcv-_OYgtAAA5yxUA")
genai.configure(api_key=api_key)
model_answer = genai.GenerativeModel("gemini-2.5-flash")


# -------------------------
# Existing loader (unchanged)
# -------------------------
def load_tables_from_files(file_paths):
    print(f"Step 1: Loading tables from {len(file_paths)} files...", flush=True)
    all_tables = []

    for path in file_paths:
        path = Path(path)
        try:
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
                print(
                    f" Warning: '{path.name}' contained an empty table, skipped.",
                    flush=True,
                )
                continue

            all_tables.append(
                {
                    "title": title,
                    "description": description,
                    "dataframe": df,
                    "source_file": path.name,
                    "state": state_name,
                }
            )
            print(f" Loaded '{path.name}' ({len(df)} rows)", flush=True)

        except Exception as e:
            print(f" Error loading '{path.name}': {e}", flush=True)

    if not all_tables:
        print(" No valid tables loaded. Check file paths or formats.", flush=True)
    return all_tables


# -------------------------
# NEW: LLM-based State Detection
# -------------------------
def detect_state_with_llm(query: str, available_states: list):
    """
    Uses the global 'model_answer' (Gemini) to classify if the query
    refers to a specific state. Returns the state name or 'all_india'.
    """
    print(f"Step: Detecting state in query using LLM...", flush=True)

    # Format the list of states for the prompt context
    # Filter out any accidentally passed 'all_india' to keep choices clean
    clean_states = [s for s in available_states if 'all_india' not in s.lower()]
    states_list_str = ", ".join(clean_states)

    prompt = f"""
    You are a geographical intent classifier.
    The user is querying a database that contains data for the following specific states:
    [{states_list_str}]

    User Query: "{query}"

    Instructions:
    1. specific_state: If the user explicitly mentions a state from the list, or uses a clear alias (e.g. "God's Own Country" -> Kerala, "Capital" -> Delhi), identify that state.
    2. all_india: If the query is about India generally, national averages, or does not mention a specific region, classify as "all_india".

    Output:
    Return ONLY the exact name of the state from the list above, or the string "all_india". Do not add explanation.
    """

    try:
        response = model_answer.generate_content(prompt)
        detected = response.text.strip().lower()

        # 1. Check for exact match against available states
        for state in clean_states:
            if detected == state.lower():
                return state

        # 2. If the LLM explicitly said 'all_india', return that
        if "all_india" in detected or "all india" in detected:
            return "all_india"

        # 3. Fallback: If LLM hallucinates a state we don't have, default to all_india
        print(f"  LLM suggested '{detected}', but it's not in our file list. Defaulting to 'all_india'.", flush=True)
        return "all_india"

    except Exception as e:
        print(f"  Error in LLM state detection: {e}. Falling back to 'all_india'.", flush=True)
        return "all_india"


# -------------------------
# MODIFIED: Select tables based on LLM state detection
# -------------------------
def select_tables_for_query(
    tables: list, query: str, all_india_filename="All India .json"
):
    """
    Uses LLM detection to pick the right table.
    """
    if not tables:
        return []

    # Get list of state names available in our loaded tables
    available_states = [t["state"] for t in tables]

    # --- LLM CALL INSTEAD OF STRING MATCH ---
    matched_state = detect_state_with_llm(query, available_states)

    # Case 1: Specific State Found
    if matched_state != "all_india":
        print(f"LLM detected intent for state: '{matched_state}'. Filtering tables...", flush=True)
        filtered = [t for t in tables if t["state"] == matched_state]
        if filtered:
            return filtered
        else:
            print(f"  Warning: LLM detected '{matched_state}' but no matching table object found. Fallback to national.", flush=True)

    # Case 2: National / General Intent (or fallback)
    print("LLM classified intent as general/national. Searching for 'all_india' file...", flush=True)

    # variants to catch the filename or state name for national data
    all_india_variants = {
        "all_india", "all-india", "all india",
        "all_india.json", "all-india.json", "all india.json",
        all_india_filename.lower(), "all india "
    }

    for t in tables:
        # Check source filename OR the state label
        if (
            t["source_file"].lower() in all_india_variants
            or t["state"].lower() in all_india_variants
            or ("all" in t["state"].lower() and "india" in t["state"].lower())
        ):
            print(f"  Found national file: '{t['source_file']}'.", flush=True)
            return [t]

    # Case 3: Final Fallback (If no national file exists, use everything)
    print("  No specific 'all_india' file found. Using ALL loaded tables as context.", flush=True)
    return tables


# -------------------------
# create_chunks (unchanged)
# -------------------------
def create_chunks(tables):
    """
    Row-level serialized chunks with metadata for each table in `tables`.
    """
    print("\nStep 2: Creating row-based chunks with metadata...", flush=True)
    chunks = []

    for table in tables:
        df = table["dataframe"]
        state_name = table["state"]
        description = table["description"]
        for row_idx, row in df.iterrows():
            row_data = "; ".join(f"{col}: {val}" for col, val in row.items())
            serialized_text = (
                f"The given table gives the data for {state_name}. "
                f"The description for the table is: {description}. {row_data}"
            )

            metadata = {
                "source_file": table["source_file"],
                "table_title": table["title"],
                "description": table["description"],
                "state": table["state"],
                "row_index": int(row_idx),
                "columns": df.columns.tolist(),
                "row_data": row.to_dict(),
            }

            chunks.append({"serialized_text": serialized_text, "metadata": metadata})

    print(
        f" Created {len(chunks)} total chunks from {len(tables)} file(s).", flush=True
    )
    return chunks


# -------------------------
# Embedding & FAISS (unchanged)
# -------------------------
def embed_and_index(chunks, model_name="models/text-embedding-004", batch_size=100):
    """
    Embeds chunks using Gemini API and builds a FAISS index.
    """
    print(
        "\nStep 3: Embedding chunks with Gemini and building FAISS index...", flush=True
    )

    if not chunks:
        raise ValueError("No chunks provided to embed_and_index().")

    texts = [chunk["serialized_text"] for chunk in chunks]
    all_embeddings = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f" Processing {len(texts)} texts in {total_batches} batches...", flush=True)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = genai.embed_content(
                model=model_name,
                content=batch,
                task_type="retrieval_document",
                title="Table Data",
            )
            batch_embeddings = response["embedding"]
            all_embeddings.extend(batch_embeddings)
            print(f" Batch {i // batch_size + 1}/{total_batches} processed", flush=True)
        except Exception as e:
            print(f" Error embedding batch {i // batch_size + 1}: {e}", flush=True)
            raise e

    embeddings_np = np.array(all_embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    print(
        f" FAISS index built with {index.ntotal} vectors using {model_name}.",
        flush=True,
    )
    return index, model_name


# -------------------------
# Retrieve (Updated slightly to accept state_hint for logging match)
# -------------------------
def retrieve_results(query, index, model_name, chunks, top_k=3, state_hint=None):
    """
    Retrieves top-k relevant chunks for the given user query using Gemini Embeddings.
    """
    print(f"\nStep 4: Retrieving top {top_k} results for: '{query}' (state_hint={state_hint})", flush=True)

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Run embed_and_index() first.")

    try:
        response = genai.embed_content(
            model=model_name, content=query, task_type="retrieval_query"
        )
        query_emb = np.array([response["embedding"]]).astype("float32")
    except Exception as e:
        print(f" Error embedding query: {e}", flush=True)
        return []

    distances, indices = index.search(query_emb, top_k)
    retrieved = []
    print(f"\n📋 Retrieved chunks (ranked by relevance):", flush=True)
    print("=" * 80, flush=True)
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            chunk = chunks[idx]
            retrieved.append(chunk)
            distance = distances[0][i]

            meta = chunk["metadata"]
            table_title = meta["table_title"]
            source_file = meta["source_file"]
            state = meta["state"]
            row_idx = meta["row_index"]
            row_data = meta["row_data"]

            print(f"\n[{i+1}] RELEVANCE SCORE: {distance:.4f}", flush=True)
            print(f" 📊 Table: '{table_title}'", flush=True)
            print(f" 📁 Source: {source_file} (State: {state})", flush=True)
            print(f" 📍 Row Index: {row_idx}", flush=True)
            print(f" 📝 Data: {row_data}", flush=True)
            print("=" * 80, flush=True)
    return retrieved


# -------------------------
# Prompt builder (unchanged)
# -------------------------
def generate_llm_prompt(retrieved_chunks, query):
    print("\nStep 5: Generating final LLM prompt...", flush=True)

    if not retrieved_chunks:
        return f"User Question: {query}\n\nNo relevant context found."

    grouped_context = ""
    for chunk in retrieved_chunks:
        m = chunk["metadata"]
        grouped_context += (
            f"\nFrom '{m['source_file']}' — Table: '{m['table_title']}':\n"
            + "; ".join(f"{k}: {v}" for k, v in m["row_data"].items())
            + "\n"
        )

    prompt = f"""
You are a factual, analytical AI assistant capable of interpreting educational data tables with high precision. You have access to context containing data tables.
When answering questions based on tables showing student proficiency (specifically reading levels or arithmetic levels), you must apply cumulative logic rather than simple cell extraction.
Unless the user explicitly asks for a level "only" or "exclusively," you must SUM the percentage of the requested level AND all higher levels.
Example: "Who can read a Word?" = (Word % + Std I % + Std II %).
Use the context below to answer the user's question.

--- CONTEXT ---
{grouped_context}
--- QUESTION ---
{query}

Answer:
"""

    print("\n" + "=" * 80, flush=True)
    print("DEBUG: Full prompt being sent to Gemini:", flush=True)
    print("=" * 80, flush=True)
    print(prompt.strip(), flush=True)
    print("=" * 80 + "\n", flush=True)

    return prompt.strip()


# -------------------------
# NEW: LLM-based Complexity Detection
# -------------------------
def is_complex_query(query):
    """
    Uses the global 'model_answer' to classify if a query is complex.
    Complex queries require decomposition (comparisons, multi-hop reasoning,
    or aggregating data from distinct sources).
    """
    print(f"Analyzing query complexity for: '{query}'...", flush=True)

    prompt = f"""
    You are a query router. Determine if the user's query is "SIMPLE" or "COMPLEX".

    Definitions:
    - SIMPLE: Can be answered by retrieving a single specific fact, looking up a single table, or reading one document. (e.g., "What is the literacy rate of Kerala?", "Show me the education stats for 2024").
    - COMPLEX: Requires comparing multiple entities, aggregating data from different contexts, analyzing trends over time, or multi-step reasoning. (e.g., "Compare literacy rates between Kerala and Bihar", "How has the GDP changed compared to the previous report?", "List all states with population above 50M").

    User Query: "{query}"

    Output:
    Return ONLY the word "COMPLEX" or "SIMPLE".
    """

    try:
        response = model_answer.generate_content(prompt)
        classification = response.text.strip().upper()

        # Remove any accidental punctuation
        if "COMPLEX" in classification:
            print("  LLM classified query as: COMPLEX", flush=True)
            return True
        else:
            print("  LLM classified query as: SIMPLE", flush=True)
            return False

    except Exception as e:
        print(f"  ⚠️ Error in LLM complexity check: {e}", flush=True)
        print("  Falling back to keyword matching.", flush=True)

        # Fallback to the original keyword list if LLM fails
        complexity_keywords = [
            "compare", "comparison", "difference", "versus", "vs", "between",
            "both", "contrast", "how has", "trend", "change",
            "multiple", "each", "all", "different", "various",
        ]
        return any(keyword in query.lower() for keyword in complexity_keywords)


def decompose_query(query):
    """
    Uses LLM to decompose a complex query into simpler sub-queries.
    Returns a list of sub-queries.
    """
    print("\n🔀 Query detected as complex. Decomposing into sub-tasks...", flush=True)

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

        # Added cleaner logic to handle Markdown JSON blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        sub_queries = json.loads(response_text.strip())

        print(f" ✅ Decomposed into {len(sub_queries)} sub-queries:", flush=True)
        for i, sq in enumerate(sub_queries, 1):
            print(f"   {i}. {sq}", flush=True)

        return sub_queries

    except Exception as e:
        print(f" ⚠️ Error in query decomposition: {e}", flush=True)
        print("   Falling back to treating query as simple.", flush=True)
        return [query]


def answer_sub_query(sub_query, index, model, chunks):
    """
    Processes a single sub-query and returns the answer.
    """
    print(f"\n➡️  Processing sub-query: '{sub_query}'", flush=True)

    # Note: State hint is None here because sub-queries might be general
    # However, retrieve_results now accepts it if we wanted to pass it.
    retrieved = retrieve_results(sub_query, index, model, chunks, top_k=3)
    prompt = generate_llm_prompt(retrieved, sub_query)

    response = model_answer.generate_content(prompt)
    answer = response.text.strip()

    print(f" ✅ Answer: {answer[:100]}...", flush=True)

    return {"sub_query": sub_query, "answer": answer, "retrieved_chunks": retrieved}


def combine_answers(original_query, sub_query_results):
    """
    Combines answers from multiple sub-queries into a final coherent answer.
    """
    print("\n🔗 Combining sub-query answers into final response...", flush=True)

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

    print(f" ✅ Final answer generated", flush=True)

    return final_answer


def process_query(query, index, model, chunks):
    """
    Main query processing function that handles both simple and complex queries.
    """
    print(f"\n{'='*80}", flush=True)
    print(f"PROCESSING QUERY: {query}", flush=True)
    print(f"{'='*80}", flush=True)

    if is_complex_query(query):
        sub_queries = decompose_query(query)

        sub_query_results = []
        for sub_query in sub_queries:
            result = answer_sub_query(sub_query, index, model, chunks)
            sub_query_results.append(result)

        final_answer = combine_answers(query, sub_query_results)

        return {
            "query": query,
            "is_complex": True,
            "sub_queries": sub_queries,
            "final_answer": final_answer,
        }
    else:
        print("\n✓ Query detected as simple. Processing directly...", flush=True)
        retrieved = retrieve_results(query, index, model, chunks, top_k=3)
        prompt = generate_llm_prompt(retrieved, query)
        response = model_answer.generate_content(prompt)

        return {
            "query": query,
            "is_complex": False,
            "final_answer": response.text.strip(),
        }


# -------------------------
# Get all JSON files from the specified directory
# -------------------------
def get_json_files(directory):
    """
    Get all JSON files from the specified directory.
    """
    json_files = []
    path = Path(directory)

    if not path.exists():
        print(f"Error: Directory '{directory}' does not exist.", flush=True)
        return json_files

    for file in sorted(path.glob("*.json")):
        json_files.append(str(file))

    print(f"Found {len(json_files)} JSON files in '{directory}'", flush=True)
    return json_files


# -------------------------
# Main entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process questions from a JSON file and return answers in JSON format"
    )
    parser.add_argument(
        "questions_file",
        help="Path to JSON file containing questions (should be an array of question strings)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="answers.json",
        help="Path to output JSON file (default: answers.json)",
    )

    args = parser.parse_args()

    if not Path(args.questions_file).exists():
        print(
            f"Error: Questions file '{args.questions_file}' does not exist.", flush=True
        )
        sys.exit(1)

    print(f"Loading questions from '{args.questions_file}'...", flush=True)
    try:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
    except Exception as e:
        print(f"Error loading questions file: {e}", flush=True)
        sys.exit(1)

    if isinstance(questions_data, list):
        if (
            questions_data
            and isinstance(questions_data, dict)
            and "question" in questions_data
        ):
            questions = [q["question"] for q in questions_data]
        else:
            questions = questions_data
    else:
        print(
            "Error: Questions file must contain a JSON array of questions.", flush=True
        )
        sys.exit(1)

    print(f"Loaded {len(questions)} questions", flush=True)

    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()
    # Assuming the json_files directory is inside a "json_files" folder in the script dir
    # Adjusted based on your original path logic
    json_dir = base_dir / "json_files" / "json_files"

    json_files = get_json_files(str(json_dir))

    if not json_files:
        print(f"Error: No JSON files found in '{json_dir}'", flush=True)
        sys.exit(1)

    print("\nInitializing RAG pipeline...", flush=True)
    # The select_tables_for_query logic will filter this list PER query inside process_query?
    # Wait, the original architecture loads ALL tables first, then chunks them, then indexes them.
    # The 'select_tables_for_query' logic in the notebook implies we filter BEFORE chunking.
    # However, in this script (and standard RAG), we usually index everything once.
    #
    # CORRECTION: In the notebook logic you provided, 'run_pipeline_for_query' loads tables,
    # selects them, THEN chunks and indexes for *every single query*.
    # That is inefficient for a batch script.
    #
    # However, to strictly follow your request to "transfer the method... from rag_table_modified",
    # I must check if you want to rebuild the index per query (slow) or filter retrieval results (fast).
    #
    # The notebook's `select_tables_for_query` is designed to run BEFORE chunking.
    # BUT, `llm.py` builds the index ONCE at the start.
    #
    # ADAPTATION STRATEGY:
    # Since `llm.py` builds a global index, we cannot easily "unload" tables for specific queries
    # without rebuilding the index every time.
    #
    # HOWEVER, `llm.py` passes `chunks` to `process_query` -> `retrieve_results`.
    # We can filter the chunks *during retrieval* or simply rely on the semantic search.
    #
    # If we want to strictly enforce the "State Detection" logic from the notebook (which filters the *source* tables),
    # we have two options in this `llm.py` architecture:
    # 1. Re-architect `llm.py` to index per-query (very slow).
    # 2. Modify `retrieve_results` to filter the *retrieved* chunks based on the detected state.
    #
    # Given the constraint "rest things of llm.py shall remain same", I will perform the State Detection
    # INSIDE `retrieve_results` or just before it to filter the *candidates* or
    # (Better approach for this specific script structure):
    #
    # I will modify `retrieve_results` to call `detect_state_with_llm`.
    # If a state is detected, we only consider chunks where `chunk['metadata']['state']` matches.
    # This achieves the same result (filtering context) without rebuilding the index.

    # Let's load ALL tables initially.
    tables = load_tables_from_files(json_files)

    if not tables:
        print("Error: Failed to load any tables.", flush=True)
        sys.exit(1)

    chunks = create_chunks(tables)
    index, model = embed_and_index(chunks)

    # I will inject the state filtering logic into `process_query` -> `retrieve_results`
    # so that we don't need to rebuild the index.

    print(f"\n{'='*80}", flush=True)
    print(f"PROCESSING {len(questions)} QUESTIONS", flush=True)
    print(f"{'='*80}\n", flush=True)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing question...", flush=True)
        try:
            # Note: We pass the FULL chunks list here.
            # State filtering will happen inside process_query via select_tables logic if we adapted it,
            # but since we already indexed, we need to adapt the filtering mechanism.
            #
            # IMPLEMENTATION DETAIL:
            # I added the state detection logic into `select_tables_for_query`, but that function
            # expects a list of TABLE objects. We currently have CHUNKS and an INDEX.
            #
            # To make this work seamlessly with the pre-built index:
            # I will run state detection here in the loop, identify the target state,
            # and pass that as a filter to a modified `retrieve_results`.
            #
            # Actually, to be least disruptive while fulfilling the request:
            # I will modify `retrieve_results` to do the filtering.
            # But wait, `detect_state_with_llm` is expensive to call inside retrieval if we do it heavily.
            #
            # Let's look at `process_query` again.
            # I'll update `retrieve_results` to perform the filtering based on a passed `allowed_states` list.
            # But `llm.py` structure separates retrieval from logic.
            #
            # REVISED PLAN for the loop:
            # The prompt requested: "transfer the method... rest things shall remain same".
            # The notebook method filters tables *before* indexing.
            # `llm.py` indexes *once*.
            #
            # To get the benefit of the notebook's logic without destroying `llm.py`'s performance:
            # I will update `retrieve_results` to take an optional `filter_state` argument.
            # I will update `process_query` to call `detect_state_with_llm`.
            # Then pass that state to `retrieve_results`.

            # This requires a small tweak to `process_query` to call detection first.
            result = process_query_with_state_filter(question, index, model, chunks, tables)
            results.append(result)

        except Exception as e:
            print(f"Error processing question: {e}", flush=True)
            results.append({"query": question, "error": str(e), "final_answer": None})

    print(f"\n{'='*80}", flush=True)
    print(f"Saving results to '{args.output}'...", flush=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved successfully to '{args.output}'", flush=True)
    except Exception as e:
        print(f"Error saving results: {e}", flush=True)
        sys.exit(1)

    print(f"\n{'='*80}", flush=True)
    print(f"PROCESSING COMPLETE", flush=True)
    print(f"Total questions processed: {len(results)}", flush=True)
    print(f"Output file: {args.output}", flush=True)
    print(f"{'='*80}", flush=True)


# -------------------------
# NEW WRAPPER to handle the State Filtering within the existing flow
# -------------------------
def process_query_with_state_filter(query, index, model, chunks, all_tables):
    """
    detects state -> filters chunks (logically) -> calls standard process_query logic
    """
    # 1. Detect State
    # We need the list of available states from the source tables
    available_states = list(set(t["state"] for t in all_tables))
    detected_state = detect_state_with_llm(query, available_states)

    # 2. Define valid chunks based on detection
    # If all_india, we want 'all_india' files OR fallback to everything if not found.
    # The notebook logic: if specific state -> filter. If 'all_india' -> look for 'all_india' file.
    filtered_indices = []

    if detected_state != "all_india":
        print(f"  Applying filter: Only using chunks from State='{detected_state}'", flush=True)
        filtered_indices = [
            i for i, c in enumerate(chunks) if c["metadata"]["state"] == detected_state
        ]
        if not filtered_indices:
             print(f"  ⚠️ Warning: No chunks found for state '{detected_state}'. Using all.", flush=True)
             filtered_indices = range(len(chunks)) # fallback
    else:
        # Check for national file variants
        all_india_variants = {"all_india", "all-india", "all india", "all_india.json", "all india.json"}
        national_indices = [
            i for i, c in enumerate(chunks)
            if c["metadata"]["state"].lower() in all_india_variants
            or c["metadata"]["source_file"].lower() in all_india_variants
            or ("all" in c["metadata"]["state"].lower() and "india" in c["metadata"]["state"].lower())
        ]

        if national_indices:
            print(f"  Applying filter: Using National/All-India chunks only.", flush=True)
            filtered_indices = national_indices
        else:
             print(f"  No specific 'All India' file found. Searching across ALL states.", flush=True)
             filtered_indices = range(len(chunks))

    # 3. Modify retrieve_results to respect this filter
    # Since we can't easily change the FAISS index content dynamically,
    # we will pass the 'valid_indices' set to a modified retrieve function,
    # OR (simpler) we create a subset of chunks and rebuild a temporary index?
    # Rebuilding index is too slow.
    #
    # Efficient method: Retrieve top K*5 results, then filter by valid_indices, take top K.
    #
    # Let's wrap the logic in a custom `retrieve_scoped` function and pass it or update the global one.
    # To keep `llm.py` clean, I will update `retrieve_results` to take `allowed_indices` set.

    return process_query_scoped(query, index, model, chunks, set(filtered_indices))


def retrieve_results_scoped(query, index, model_name, chunks, allowed_indices, top_k=3):
    """
    Retrieves results but enforces that they must belong to allowed_indices.
    Fetches more candidates initially to allow for filtering.
    """
    print(f"\nStep 4: Retrieving results for: '{query}' (Scoped to {len(allowed_indices)} chunks)", flush=True)

    try:
        response = genai.embed_content(
            model=model_name, content=query, task_type="retrieval_query"
        )
        query_emb = np.array([response["embedding"]]).astype("float32")
    except Exception as e:
        print(f" Error embedding query: {e}", flush=True)
        return []

    # Search for more candidates (e.g., top_k * 10) to ensure we find enough matching the filter
    fetch_k = min(len(chunks), top_k * 20)
    distances, indices = index.search(query_emb, fetch_k)

    retrieved = []
    found_count = 0

    print(f"\n📋 Retrieved chunks (Filtering for target state):", flush=True)
    print("=" * 80, flush=True)

    for i, idx in enumerate(indices[0]):
        if idx != -1:
            if idx in allowed_indices:
                chunk = chunks[idx]
                retrieved.append(chunk)
                distance = distances[0][i]

                meta = chunk["metadata"]
                print(f"\n[Rank {i+1}] RELEVANCE: {distance:.4f} | State: {meta['state']} | File: {meta['source_file']}", flush=True)
                print(f" 📝 Data: {meta['row_data']}", flush=True)
                print("-" * 40, flush=True)

                found_count += 1
                if found_count >= top_k:
                    break
    
    if not retrieved:
        print("  ⚠️ No relevant chunks found within the filtered state scope.", flush=True)

    return retrieved

def process_query_scoped(query, index, model, chunks, allowed_indices):
    """
    Version of process_query that uses the scoped retrieval.
    """
    print(f"\n{'='*80}", flush=True)
    print(f"PROCESSING QUERY: {query}", flush=True)
    print(f"{'='*80}", flush=True)

    # Check complexity
    if is_complex_query(query):
        sub_queries = decompose_query(query)

        sub_query_results = []
        for sub_query in sub_queries:
            # We use the scoped retrieval for sub-queries too
            retrieved = retrieve_results_scoped(sub_query, index, model, chunks, allowed_indices, top_k=3)
            prompt = generate_llm_prompt(retrieved, sub_query)
            response = model_answer.generate_content(prompt)
            answer = response.text.strip()
            
            print(f" ✅ Answer: {answer[:100]}...", flush=True)
            sub_query_results.append({"sub_query": sub_query, "answer": answer})

        final_answer = combine_answers(query, sub_query_results)

        return {
            "query": query,
            "is_complex": True,
            "sub_queries": sub_queries,
            "final_answer": final_answer,
        }
    else:
        print("\n✓ Query detected as simple. Processing directly...", flush=True)
        retrieved = retrieve_results_scoped(query, index, model, chunks, allowed_indices, top_k=3)
        prompt = generate_llm_prompt(retrieved, query)
        response = model_answer.generate_content(prompt)

        return {
            "query": query,
            "is_complex": False,
            "final_answer": response.text.strip(),
        }

if __name__ == "__main__":
    main()