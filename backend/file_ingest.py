import pandas as pd
import os


def ingest_and_chunk(file_path: str, chunk_size: int = 100):
    """
    Loads a CSV/XLSX file, chunks it, and returns a list of stringified chunks and metadata.
    returns chuks(list) and metadata(dictionary)
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"[ERROR] Unsupported file type: {ext}")
            return None, {"error": "Unsupported file type"}
        

        # Chunk the DataFrame
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk  = df.iloc[i: i + chunk_size]
            chunk_str = chunk.to_csv(index=False) #can do to_str as well, but to_csv is better for our context
            chunks.append(chunk_str)
        print(f"[DEBUG] Created {len(chunks)} chunks of size {chunk_size}")

        metadata = {
            "file_name": os.path.basename(file_path),
            "columns": df.columns.tolist(),
            "num_rows": len(df),
            "num_chunks": len(chunks)
        }
        return chunks, metadata

    except Exception as e:
        print(f"[ERROR] Failed to ingest file: {e}")
        return None, {"error": str(e)}