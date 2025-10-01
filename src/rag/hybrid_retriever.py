# clinical_agent/src/rag/hybrid_retriever.py
from typing import List
from Clinical_agent.src.embeddings.hybrid_index import HybridIndex

class HybridRetriever:
    def __init__(self, use_reranker: bool = False):
        self.index = HybridIndex()
        self.index.load()
        self.use_reranker = use_reranker
        self.reranker = None
        if self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder("BAAI/bge-reranker-large")
            except Exception:
                self.reranker = None

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        cands = self.index.search(query, k=max(k, 10))
        docs = [t for t, _ in cands]
        if self.reranker:
            pairs = [(query, d) for d in docs]
            scores = self.reranker.predict(pairs)
            reranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
            return reranked[:k]
        return docs[:k]
