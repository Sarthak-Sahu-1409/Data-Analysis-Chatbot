import pandas as pd
import logging
import os
import shutil
import ollama
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend import config
from backend.embed_store import EmbedStore
from backend.file_ingest import FileIngestor
from backend.chartGenerator import ChartGenerator
import numpy as np

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

app = FastAPI()
embed_store = EmbedStore(
    model_name=config.EMBEDDING_MODEL,
    index_path=config.FAISS_INDEX_PATH,
    meta_path=config.META_PATH,
)
file_ingestor = FileIngestor(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)
chart_generator = ChartGenerator(chart_dir=config.CHART_DIR)

async def _assess_data_sufficiency(context: str, query: str) -> tuple[bool, str]:
    """
    Analyzes whether the retrieved data is sufficient to answer the query.
    
    Args:
        context (str): The current context data
        query (str): The original query
        
    Returns:
        tuple[bool, str]: (is_sufficient, missing_information)
    """
    assessment_prompt = f"""Analyze if the following data is sufficient to answer the question completely.
If information is missing, specify exactly what additional data would be needed.

Question: {query}

Available Data:
{context}

Respond in the following format:
SUFFICIENT: true/false
MISSING: <list specific missing information or 'none' if sufficient>
"""
    
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": assessment_prompt}]
        )
        result = response['message']['content']
        
        # Parse the response
        lines = result.strip().split('\n')
        is_sufficient = 'SUFFICIENT: true' in result.lower()
        missing_info = ''
        
        for line in lines:
            if line.startswith('MISSING:'):
                missing_info = line[8:].strip()
                break
        
        return is_sufficient, missing_info
    except Exception as e:
        logging.error(f"Data sufficiency assessment failed: {e}")
        return False, "Error in assessing data sufficiency"

async def _generate_refined_query(original_query: str, missing_info: str) -> str:
    """
    Generates a refined query to find specific missing information.
    
    Args:
        original_query (str): The original user query
        missing_info (str): Description of what information is missing
        
    Returns:
        str: A new, more targeted query
    """
    refinement_prompt = f"""Generate a specific, focused query to find the following missing information:
Original Question: {original_query}
Missing Information: {missing_info}

Generate a simple, direct query that will help find this specific information.
Output ONLY the refined query, nothing else."""
    
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": refinement_prompt}]
        )
        refined_query = response['message']['content'].strip()
        logging.info(f"Generated refined query: {refined_query}")
        return refined_query
    except Exception as e:
        logging.error(f"Query refinement failed: {e}")
        return original_query  # Fallback to original query

async def _reflective_retrieval(query: str, max_iterations: int = 3) -> tuple[str, List[Dict[str, Any]]]:
    """
    Performs iterative self-correcting retrieval until sufficient data is found.
    
    Args:
        query (str): The original query
        max_iterations (int): Maximum number of refinement iterations
        
    Returns:
        tuple[str, List[Dict]]: (final_context, all_retrieved_chunks)
    """
    logging.info("Starting reflective retrieval process")
    current_query = query
    all_chunks = []
    iteration = 0
    
    # Start with query decomposition
    sub_queries = await _decompose_query(query)
    logging.info(f"Decomposed main query into {len(sub_queries)} sub-queries")
    
    while iteration < max_iterations:
        # Process each sub-query first
        for sub_query in sub_queries:
            current_chunks = embed_store.search(sub_query, top_k=3)  # Reduced to 3 per sub-query
            
            # Add new unique chunks
            existing_contents = {chunk["chunk"] for chunk in all_chunks if "chunk" in chunk}
            for chunk in current_chunks:
                if chunk.get("chunk") not in existing_contents:
                    all_chunks.append(chunk)
        
        # Then process the current refined query if it's different from the original
        if current_query != query:
            refined_chunks = embed_store.search(current_query, top_k=3)
            for chunk in refined_chunks:
                if chunk.get("chunk") not in existing_contents:
                    all_chunks.append(chunk)
        
        # Construct current context
        current_context = "\n---\n".join([c["chunk"] for c in all_chunks if "chunk" in c])
        
        # Assess if we have sufficient data
        is_sufficient, missing_info = await _assess_data_sufficiency(current_context, query)
        
        if is_sufficient or missing_info.lower() == "none":
            logging.info(f"Sufficient data found after {iteration + 1} iterations")
            break
        
        # Generate refined query for missing information
        current_query = await _generate_refined_query(query, missing_info)
        logging.info(f"Iteration {iteration + 1}: Retrieving additional data for: {current_query}")
        
        iteration += 1
    
    # Re-rank all collected chunks for final context
    if all_chunks:
        all_chunks = _rerank_chunks(all_chunks, query)
        final_context = "\n---\n".join([c["chunk"] for c in all_chunks[:5] if "chunk" in c])
    else:
        final_context = ""
    
    return final_context, all_chunks

async def _decompose_query(query: str) -> List[str]:
    """
    Breaks down a complex query into simpler sub-queries using the LLM.
    
    Args:
        query (str): The original complex query
        
    Returns:
        List[str]: List of sub-queries
    """
    decomposition_prompt = f"""Break down the following complex query into simple, focused sub-queries.
Output ONLY the list of sub-queries, one per line. If the query is already simple, output it as is.

Complex Query: "{query}"

Sub-queries:"""

    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": decomposition_prompt}]
        )
        sub_queries = [q.strip() for q in response['message']['content'].split('\n') if q.strip()]
        logging.info(f"Decomposed query into {len(sub_queries)} sub-queries")
        return sub_queries
    except Exception as e:
        logging.error(f"Query decomposition failed: {e}")
        return [query]  # Fallback to original query

def _rerank_chunks(chunks: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
    """
    Re-ranks chunks based on their relevance to the original query using a scoring system.
    
    Args:
        chunks (List[Dict]): List of retrieved chunks with their metadata
        original_query (str): The original user query
        
    Returns:
        List[Dict]: Re-ranked list of chunks
    """
    try:
        # Create relevance prompt for the LLM
        relevance_scores = []
        for chunk in chunks:
            chunk_content = chunk.get("chunk", "")
            relevance_prompt = f"""Rate the relevance of the following data to answering this question.
Consider exact matches, numeric values, and contextual relevance.
Rate from 0 to 10, output ONLY the number.

Question: {original_query}

Data:
{chunk_content}

Rating (0-10):"""
            
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": relevance_prompt}]
            )
            try:
                score = float(response['message']['content'].strip())
                relevance_scores.append(score)
            except ValueError:
                relevance_scores.append(0)
        
        # Create numpy array for efficient sorting
        scores = np.array(relevance_scores)
        ranked_indices = np.argsort(-scores)  # Sort in descending order
        
        # Return re-ranked chunks
        return [chunks[i] for i in ranked_indices]
    except Exception as e:
        logging.error(f"Chunk re-ranking failed: {e}")
        return chunks  # Fallback to original order

# Mount static files directory for serving charts
app.mount("/charts", StaticFiles(directory=config.CHART_DIR), name="charts")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    """Returns the current status of the backend services."""
    return {
        "status": "ok",
        "ollama_model": config.OLLAMA_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "index_status": {
            "indexed": embed_store.index is not None,
            "vector_count": embed_store.index.ntotal if embed_store.index else 0,
        }
    }

@app.post("/upload", 
         description="Upload a CSV or Excel file for analysis",
         response_description="File upload status and metadata")
async def upload_file(file: UploadFile = File(..., description="The file to upload (CSV or Excel)")):
    """
    Upload and process a data file for analysis.
    
    The file will be:
    1. Saved to disk
    2. Processed into chunks
    3. Indexed for semantic search
    
    Returns:
        dict: Upload status and file metadata
    """
    file_location = os.path.join(config.UPLOAD_DIR, file.filename)
    try:
        logging.info(f"Received file upload request: {file.filename}")
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File saved to {file_location}")

        # Ingest and chunk the file
        logging.info(f"Starting ingestion and chunking for: {file_location}")
        chunks, metadata = file_ingestor.ingest_file(file_location)
        if chunks is None:
            logging.error(f"File ingestion failed for '{file.filename}'. Error: {metadata.get('error')}")
            return {"filename": file.filename, "status": "error", "details": metadata.get("error")}
        logging.info(f"File chunked into {len(chunks)} chunks")

        # Build FAISS index
        logging.info(f"Clearing previous FAISS index (if any)")
        embed_store.clear_index()
        chunk_meta = [{"chunk": c, **metadata} for c in chunks]
        logging.info(f"Building FAISS index for file: {file.filename}")
        embed_store.build_index(chunks, chunk_meta)
        logging.info(f"FAISS index built for file '{file.filename}'")

        return {
            "filename": file.filename,
            "status": "uploaded",
            "metadata": metadata
        }
    except Exception as e:
        logging.error(f"Upload failed for file '{file.filename}': {e}")
        return {"filename": file.filename, "status": "error", "details": str(e)}

@app.post("/query",
         description="Ask a question about the uploaded data",
         response_description="Answer and optional chart")
async def query_data(query: str = Form(..., description="The question to ask about the data")):
    """
    Process a natural language query about the uploaded data.
    
    The query will be:
    1. Used to retrieve relevant context from the indexed data
    2. Processed by the LLM to generate an answer
    3. Analyzed for chart generation if visualization is requested
    
    Returns:
        dict: Contains the answer text and optional chart path
    """
    try:
        logging.info(f"Received query: {query}")
        if not hasattr(embed_store, 'search') or embed_store is None:
            logging.warning("No file has been processed/uploaded yet. embed_store is not ready.")
            return JSONResponse({"answer": "No file has been processed/uploaded yet.", "chart": None})
        else:
            logging.info("File processing is ready. embed_store is initialized.")

        # Check if this is a metadata query about rows/columns
        metadata_keywords = {
            'rows': ['rows', 'records', 'entries', 'data points'],
            'columns': ['columns', 'fields', 'variables', 'features'],
            'structure': ['structure', 'shape', 'size', 'dimensions']
        }
        
        query_lower = query.lower()
        is_metadata_query = any(
            any(keyword in query_lower for keyword in keywords)
            for keywords in metadata_keywords.values()
        )

        if is_metadata_query:
            # Get the first chunk which should be our summary
            summary_chunk = embed_store.search(query, top_k=1)[0]
            if summary_chunk and 'total_rows' in summary_chunk:
                answer = (
                    f"The dataset contains {summary_chunk['total_rows']} rows and "
                    f"{len(summary_chunk['columns'])} columns.\n\n"
                    f"The columns are: {', '.join(summary_chunk['columns'])}"
                )
                return JSONResponse({"answer": answer, "chart": None})

        # For non-metadata queries, use reflective RAG process
        logging.info("Starting reflective RAG process...")
        
        # Get initial context through reflective retrieval
        context, all_chunks = await _reflective_retrieval(query)
        
        if not context:
            return JSONResponse({
                "answer": "I couldn't find enough relevant information to answer your question.",
                "chart": None
            })
            
        logging.info("Retrieved and validated context through reflective RAG")

        # Call Ollama with the RAG prompt
        prompt = f"""You are a data analyst. Use ONLY the following context to answer the question. 
If the answer cannot be found in the context, say so. Do not make up information.

Context:
---
{context}
---

Question: {query}

Answer concisely and professionally."""

        try:
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response['message']['content']
            logging.info("Received response from LLM")
        except Exception as e:
            logging.error(f"Failed to get response from Ollama: {e}")
            answer = "I apologize, but I encountered an error while processing your query."

        # Chart generation (keyword-based)
        chart_img = None
        chart_keywords = ["plot", "chart", "graph", "visualize"]
        if any(kw in query.lower() for kw in chart_keywords):
            try:
                # Load the file again for charting
                file_name = all_chunks[0].get("file_name") if all_chunks else None
                if file_name:
                    file_path = os.path.join(config.UPLOAD_DIR, file_name)
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == ".csv":
                        # Try multiple encodings in order of preference
                        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding)
                                logging.info(f"Successfully loaded CSV with {encoding} encoding")
                                break
                            except UnicodeDecodeError:
                                if encoding == encodings[-1]:
                                    raise  # If all encodings fail, raise the error
                                continue
                    else:
                        df = pd.read_excel(file_path)
                    
                    # Add data info to the answer
                    data_info = f"\nDataset Info:\n- Rows: {len(df)}\n- Columns: {len(df.columns)}"
                    answer += data_info
                    
                    chart_path = chart_generator.generate_chart(df, query)
                    # Construct absolute URL for the chart
                    chart_url = f"{config.BASE_URL}{chart_path}"
                    answer += f"\nChart generated and available at: {chart_url}"
                    chart_img = chart_url
            except Exception as e:
                logging.error(f"Chart generation failed: {e}")
                answer += f"\nChart generation error: {e}"

        return JSONResponse({
            "answer": answer,
            "chart": chart_img
        })
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return JSONResponse({
            "answer": f"Error processing query",
            "chart": None,
            "details": str(e)
        })

