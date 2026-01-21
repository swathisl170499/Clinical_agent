# clinical_agent/src/rag/rag_pipeline.py
import re
from typing import Tuple, List
from Clinical_agent.src.llms.codestral_llm import CodeStralClient
from Clinical_agent.src.rag.hybrid_retriever import HybridRetriever

QA_PROMPT = """You are a clinical research assistant. Use ONLY the provided context.
If the context is insufficient, say you do not know.

Question: {question}

Context:
{context}

Answer in 3–5 concise, fact-based sentences suitable for a clinical trial team.
"""

SUMMARY_PROMPT = """You are a clinical research assistant. Summarize ONLY the provided context.
If the context is insufficient, say you do not know.

Request: {question}

Context:
{context}

Provide a 4–6 sentence summary for a clinical trial team. Include key facts and avoid speculation.
"""

class ClinicalRAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever(use_reranker=False)
        self.llm = CodeStralClient(
            project_id="clinical-copilot",
            region="us-central1",
            model_name="mistralai/codestral-2501@001",
            temperature=0.35,
            max_output_tokens=500
        )

    def _build_prompt(self, q: str, docs: List[str], mode: str) -> str:
        ctx = "\n\n---\n\n".join(d.strip() for d in docs if d and d.strip())
        if len(ctx) > 7000:
            ctx = ctx[:7000] + " ..."
        if mode == "summary":
            return SUMMARY_PROMPT.format(question=q.strip(), context=ctx)
        return QA_PROMPT.format(question=q.strip(), context=ctx)

    def _fallback(self, docs: List[str], q: str, mode: str) -> str:
        text = " ".join(docs[:3])
        if not text:
            return "I don’t know based on the available knowledge."
        if mode == "summary":
            snippet = " ".join([d[:240] for d in docs[:2]]).strip()
            return f"Summary based on available notes: {snippet}" if snippet else "I don’t know based on the available knowledge."
        if "symptom" in q.lower():
            hits = re.findall(r"(?:presents with|reports|complains of)\s+([a-z ,/-]{3,80})", text, flags=re.I)
            uniq = []
            for h in hits:
                for p in re.split(r",| and ", h):
                    p = p.strip().lower()
                    if 2 < len(p) < 40 and p not in uniq:
                        uniq.append(p)
            if uniq:
                return "Commonly noted symptoms include: " + ", ".join(uniq[:6]) + "."
        return " ".join([d[:300] for d in docs[:2]])

    def retrieve(self, question: str, k: int = 5) -> List[str]:
        return self.retriever.retrieve(question, k=k)

    def generate(self, question: str, docs: List[str], mode: str = "answer") -> str:
        prompt = self._build_prompt(question, docs, mode=mode)
        try:
            out = self.llm.generate([prompt])[0]
            return (out or "").strip()
        except Exception:
            return self._fallback(docs, question, mode=mode)

    def query(self, question: str) -> Tuple[str, List[str]]:
        docs = self.retrieve(question, k=5)
        return self.generate(question, docs, mode="answer"), docs
