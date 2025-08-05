from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import ollama
from backend.embed_store import EmbedStore
import re

from backend.chartGenerator import plot_and_encode
import pandas as pd
import matplotlib.pyplot as plt


embed_store = EmbedStore()
app = FastAPI()

def call_llm(prompt):
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

#Allow frontend (e.g., Streamlit/Electron) to call API
app.add_middleware(
    CORSMiddleware, # Cross Origin Resource Sharing (CORS) middleware
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_methods=["*"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from backend.file_ingest import ingest_and_chunk
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)): #UploadFile is the type and File(...) -> FastAPI's File() function to define this input is a file upload from the client, ...means no default value(required)
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copy(file.file, buffer)
        print(f"[INFO] File saved to {file_location}")
        print(f"[INFO] File '{file.filename}' uploaded successfully.")
    
        #Ingest and chunk the file
        chunks, metadata = ingest_and_chunk(file_location)
        if chunks is None:
            print(f"[ERROR] File ingestion failed for '{file.filename}'.")
            return {"filename": file.filename, "status": "error", "details": metadata.get("error")}
        print(f"[INFO] File chunked into {len(chunks)} chunks")

        # Build FAISS index
        embed_store.clear_index()
        # Store both chunk text and metadata for retrieval
        chunk_meta = [{"chunk": c, **metadata} for c in chunks]
        embed_store.build_index(chunks, chunk_meta)
        print(f"[INFO] FAISS index built for file '{file.filename}'.")

        return {
            "filename": file.filename,
            "status": "uploaded",
            "metadata": metadata
        }
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return {"filename": file.filename, "status": "error", "details": str(e)}

@app.post("/query")
async def query_data(query: str = Form(...)):
    try:
        print(f"[INFO] Received query: {query}")
        if not hasattr(embed_store, 'search') or embed_store is None:
            print("[WARN] No file has been processed/uploaded yet. embed_store is not ready.")
        else:
            print("[INFO] File processing is ready. embed_store is initialized.")
        #Retrieve top-3 relevant chunks from FAISS
        top_chunks = embed_store.search(query, top_k = 3)
        print (f"[INFO] Retrieved {len(top_chunks)} relevant chunks")

        #Construct a simple RAG prompt (for now, just return the chunks)
        context = "\n---\n".join([c["chunk"] for c in top_chunks if "chunk" in c]) 

        prompt = f"""You are a data analyst. Use ONLY the data below to answer the question.
---
{context}
---
Question: {query}

If a chart is needed, first write a brief answer in plain English.
Then, on a new line, provide ONLY the Python code (using matplotlib and pandas) to generate the chart from a DataFrame called df.
Enclose the code in triple backticks like this:

```python
# your code here
```

If no chart is needed, just answer in plain English.
"""
        llm_response = call_llm(prompt)
        print(f"[INFO] LLM response: {llm_response}")

        chart_img = None
        answer = None
        #will extract code if present using this
        code = None
        #Try to extract code from triple backticks
        code_block = re.search(r"```(?:python)?(.*?)```", llm_response, re.DOTALL)
        if code_block:
            code = code_block.group(1).strip()
            # Remove code block from answer
            answer = llm_response.replace(code_block.group(0), "").strip()
        elif "import matplotlib" in llm_response or "plt." in llm_response:
            #Fallback: try to extract lines that look like code
            lines = llm_response.splitlines()
            code_lines = [line for line in lines if line.strip().startswith(("import", "plt.", "df."))]
            code = "\n".join(code_lines)
            #Remove code lines from answer
            answer = "\n".join([line for line in lines if line not in code_lines]).strip()
        else:
            # Only text, no code
            answer = llm_response.strip()

        # --- Execute code if present ---
        if code and len(code.strip()) > 0:
            try:
                file_path = os.path.join(UPLOAD_DIR, top_chunks[0].get("file_name"))
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                local_vars = {"df": df, "plt": plt}
                exec(code, {}, local_vars)
                fig = plt.gcf()
                chart_img = plot_and_encode(fig)
                if not answer:
                    answer = "Chart generated from LLM code."
            except Exception as e:
                print(f"[ERROR] Chart code execution failed: {e}")
                answer = f"LLM code execution error: {e}"

        return JSONResponse({
            "answer": answer,
            "chart": chart_img
        })
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return JSONResponse({
            "answer": f"Error processing query",
            "chart": None,
            "details": str(e)
        })

