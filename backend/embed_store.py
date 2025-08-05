import logging
import os
import pickle
from typing import List, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# It's better to configure logging once at the application entry point.
# For this module, we'll just get the logger instance.
logger = logging.getLogger(__name__)

class EmbedStore:
    """
    Manages the creation, storage, and retrieval of vector embeddings using
    SentenceTransformers and FAISS for efficient similarity search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "vector_store/faiss.index",
        meta_path: str = "vector_store/meta.pkl",
    ):
        """
        Initializes the EmbedStore.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
            index_path (str): The file path to save/load the FAISS index.
            meta_path (str): The file path to save/load the metadata.
        """
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = meta_path
        self.vector_store_dir = os.path.dirname(index_path)

        os.makedirs(self.vector_store_dir, exist_ok=True)

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}")
            raise

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Any] = []
        self.load_index()

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Encodes a list of text chunks into vector embeddings.

        Args:
            chunks (List[str]): A list of text strings to embed.

        Returns:
            np.ndarray: A numpy array of embeddings with dtype 'float32'.
        """
        logger.info(f"Embedding {len(chunks)} chunks...")
        try:
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            logger.info("Embedding complete.")
            return np.array(embeddings).astype("float32")
        except Exception as e:
            logger.error(f"An error occurred during chunk embedding: {e}")
            # Depending on desired behavior, you might return an empty array
            # or re-raise the exception.
            raise

    def build_index(self, chunks: List[str], meta: List[Any]):
        """
        Builds a new FAISS index from text chunks and their metadata.

        Args:
            chunks (List[str]): The text chunks to index.
            meta (List[Any]): The corresponding metadata for each chunk.
        """
        if len(chunks) != len(meta):
            raise ValueError("The number of chunks and metadata items must be the same.")

        logger.info("Building new FAISS index...")
        embeddings = self.embed_chunks(chunks)
        if embeddings.shape[0] == 0:
            logger.warning("No embeddings were generated. Index will not be built.")
            return

        dimensions = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimensions)
        self.index.add(embeddings)
        self.metadata = meta
        self.save_index()
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def save_index(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("Attempted to save an empty index. Aborting.")
            return

        logger.info("Saving FAISS index and metadata to disk...")
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Index saved to '{self.index_path}' and metadata to '{self.meta_path}'.")
        except IOError as e:
            logger.error(f"Failed to write index or metadata to disk: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving the index: {e}")

    def load_index(self):
        """Loads the FAISS index and metadata from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            logger.info("Loading FAISS index and metadata from disk...")
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"FAISS index with {self.index.ntotal} vectors and metadata loaded.")
            except (IOError, pickle.UnpicklingError) as e:
                logger.error(f"Failed to load index or metadata from disk: {e}. Starting fresh.")
                self.clear_index() # Clear potentially corrupt files
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading the index: {e}. Starting fresh.")
                self.clear_index()
        else:
            logger.warning("No FAISS index found on disk. Index is empty.")
            self.index = None
            self.metadata = []

    def clear_index(self):
        """Clears the index from memory and deletes the corresponding files from disk."""
        logger.info("Clearing FAISS index from memory and disk...")
        self.index = None
        self.metadata = []
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            logger.info("FAISS index and metadata files deleted.")
        except OSError as e:
            logger.error(f"Error deleting index files: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Any]:
        """
        Searches the index for the most similar chunks to a given query.

        Args:
            query (str): The search query string.
            top_k (int): The number of top results to return.

        Returns:
            List[Any]: A list of metadata corresponding to the top matching chunks.
        """
        logger.info(f"Searching FAISS index for query: '{query}' (top_k={top_k})")
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search attempted on an empty or non-existent index.")
            return []

        # Ensure top_k is not greater than the number of items in the index
        actual_k = min(top_k, self.index.ntotal)
        if actual_k == 0:
            return []

        try:
            query_emb = self.embed_chunks([query])
            distances, indices = self.index.search(query_emb, actual_k)

            results = []
            for idx in indices[0]:
                # FAISS returns -1 for indices it can't find
                if idx != -1 and idx < len(self.metadata):
                    results.append(self.metadata[idx])
            logger.info(f"Search complete. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"An error occurred during search: {e}")
            return []