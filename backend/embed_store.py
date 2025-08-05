from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

MODEL_NAME = "all-MiniLM-L6-v2" # This is the model used for embedding, this is small, fast and good for prototyping
INDEX_PATH =  "vector_store/faiss.index"
META_PATH = "vector_store/meta.pkl"

os.makedirs("vector_store", exist_ok=True)

class EmbedStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None #find out what this is used for
        self.metadata = []

    def embed_chunks(self, chunks):
        print("[INFO] Embedding chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings).astype("float32")
    
    def build_index(self, chunks, meta):
        embeddings = self.embed_chunks(chunks) #n chunks each constituting 384 dimensional vector
        dimensions = embeddings.shape[1] # if n chunks of vectors of 384 dimensions -> (n, 384) -> 384
        self.index = faiss.IndexFlatL2(dimensions)
        self.index.add(embeddings)
        self.metadata = meta
        self.save_index()
        print(f"[INFO] FAISS index built with {len(chunks)} vectors.")

    def save_index(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            print("[INFO] FAISS index loaded.")
        else:
            print("[WARN] No FAISS index found.")
    
    def clear_index(self):
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        self.index = None
        self.metadata = []
        print("[INFO] FAISS index cleared")
    
    def search(self, query, top_k = 3):
        if self.index is None:
            self.load_index()
        query_emb = self.embed_chunks([query])
        D, I = self.index.search(query_emb, top_k) # D -> distances, I -> indices of best matches in the index
        results = []
        for idx in I[0]: # I is a 2D array shape(1, top_k) -> [[12, 4, 0]] -> I[0] is the list of top match indices - [12, 4, 0]
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results
