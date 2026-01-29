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
genai.configure(api_key=os.getenv("AIzaSyCvPJe-wKe-w_I3IYDcv-_OYgtAAA5yxUA"))
model_answer = genai.GenerativeModel("gemini-2.5-flash")


# -------------------------
# Existing loader (unchanged, minor tidy)
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
# Helper: detect state mention in a user query
# -------------------------
def detect_state_in_query(query: str, available_states: list):
    """
    Heuristic: check whether any available state names appear in the query.
    Returns the matched state name (as in available_states) or None.
    """
    q = query.lower()
    normalized = {s.lower(): s for s in available_states}

    for low_state, orig_state in normalized.items():
        alt = low_state.replace("_", " ").replace("-", " ")
        if (f" {low_state} " in f" {q} ") or (f" {alt} " in f" {q} "):
            return orig_state

    for low_state, orig_state in normalized.items():
        if low_state in q or low_state.replace("_", " ") in q:
            return orig_state

    return None


# -------------------------
# Function: choose which tables to use for a given query
# -------------------------
def select_tables_for_query(
    tables: list, query: str, all_india_filename="All India .json"
):
    """
    If query mentions a state (based on tables' 'state' field), return only that table.
    If no state mentioned -> try to find a table whose source_file matches all_india_filename (or state == 'all_india').
    If all_india isn't available, fallback to returning all tables.
    """
    if not tables:
        return []

    available_states = [t["state"] for t in tables]
    matched_state = detect_state_in_query(query, available_states)

    if matched_state:
        print(
            f"Detected state in query: '{matched_state}'. Using that table only.",
            flush=True,
        )
        return [t for t in tables if t["state"] == matched_state]

    all_india_variants = {
        "all_india",
        "all-india",
        "all india",
        "all_india.json",
        "all-india.json",
        "all india.json",
        "all india ",
        "all india.json",
    }
    for t in tables:
        src_lower = t["source_file"].lower()
        state_lower = t["state"].lower()
        if (
            src_lower in all_india_variants
            or state_lower in all_india_variants
            or ("all" in state_lower and "india" in state_lower)
        ):
            print(
                f"No state in query — using national file '{t['source_file']}' (state='{t['state']}').",
                flush=True,
            )
            return [t]

    print(
        "No state mentioned and no 'all_india' file found — falling back to using all loaded tables.",
        flush=True,
    )
    return tables


# -------------------------
# create_chunks
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
# Embedding & FAISS
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
# Retrieve
# -------------------------
def retrieve_results(query, index, model_name, chunks, top_k=3, state_hint=None):
    """
    Retrieves top-k relevant chunks for the given user query using Gemini Embeddings.
    """
    print(f"\nStep 4: Retrieving top {top_k} results for: '{query}'", flush=True)

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
# Prompt builder
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
# Query complexity detection and processing
# -------------------------
def is_complex_query(query):
    """
    Determines if a query is complex and needs decomposition.
    Complex queries typically involve comparisons, multiple conditions, or aggregations.
    """
    complexity_keywords = [
        "compare",
        "comparison",
        "difference",
        "versus",
        "vs",
        "between",
        "both",
        "contrast",
        "how has",
        "trend",
        "change",
        "multiple",
        "each",
        "all",
        "different",
        "various",
    ]

    query_lower = query.lower()
    return any(keyword in query_lower for keyword in complexity_keywords)


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
    json_dir = base_dir / "json_files" / "json_files"

    json_files = get_json_files(str(json_dir))

    if not json_files:
        print(f"Error: No JSON files found in '{json_dir}'", flush=True)
        sys.exit(1)

    print("\nInitializing RAG pipeline...", flush=True)
    tables = load_tables_from_files(json_files)

    if not tables:
        print("Error: Failed to load any tables.", flush=True)
        sys.exit(1)

    chunks = create_chunks(tables)
    index, model = embed_and_index(chunks)

    print(f"\n{'='*80}", flush=True)
    print(f"PROCESSING {len(questions)} QUESTIONS", flush=True)
    print(f"{'='*80}\n", flush=True)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing question...", flush=True)
        try:
            result = process_query(question, index, model, chunks)
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


if __name__ == "__main__":
    main()
