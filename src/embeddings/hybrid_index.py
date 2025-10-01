# clinical_agent/src/embeddings/hybrid_index.py
import os
import pickle
import faiss
import pandas as pd
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridIndex:
    def __init__(self, dense_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(dense_model)
        self.faiss = None
        self.texts = []
        self.bm25 = None

        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.csv_path = os.path.join(self.root, "clinical_data.csv")
        self.faiss_path = os.path.join(self.root, "src", "embeddings", "faiss_index.idx")
        self.map_path = os.path.join(self.root, "src", "embeddings", "text_mapping.pkl")
        self.bm25_path = os.path.join(self.root, "src", "embeddings", "bm25.pkl")

    def build(self):
        df = pd.read_csv(self.csv_path)
        if "visit_notes" not in df.columns:
            raise ValueError(f"'visit_notes' column not found in {self.csv_path}. Columns: {df.columns.tolist()}")
        self.texts = df["visit_notes"].fillna("").astype(str).tolist()

        # dense
        embs = self.model.encode(self.texts, show_progress_bar=True)
        embs = normalize(embs)
        dim = embs.shape[1]
        self.faiss = faiss.IndexFlatIP(dim)
        self.faiss.add(embs)
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(self.faiss, self.faiss_path)
        with open(self.map_path, "wb") as f:
            pickle.dump(self.texts, f)

        # sparse
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self):
        self.faiss = faiss.read_index(self.faiss_path)
        with open(self.map_path, "rb") as f:
            self.texts = pickle.load(f)
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, k: int = 6):
        # dense
        q = self.model.encode([query])
        q = normalize(q)
        D, I = self.faiss.search(q, k)
        dense = [(self.texts[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

        # sparse
        toks = query.lower().split()
        s_scores = self.bm25.get_scores(toks)
        sparse_idx = sorted(range(len(s_scores)), key=lambda i: s_scores[i], reverse=True)[:k]
        sparse = [(self.texts[i], float(s_scores[i])) for i in sparse_idx]

        # merge
        merged = {}
        for t, s in dense + sparse:
            merged[t] = max(s, merged.get(t, s))
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]
