# clinical_agent/src/agents/multi_agent_orchestrator.py
from typing import Dict, Any
from Clinical_agent.src.rag.rag_pipeline import ClinicalRAGPipeline

class MultiAgentOrchestrator:
    def __init__(self):
        self.rag = ClinicalRAGPipeline()

    def _plan(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["symptom", "drug", "medication", "adverse", "dose", "protocol"]):
            return "retrieve_then_answer"
        return "retrieve_then_answer"

    def run(self, query: str) -> Dict[str, Any]:
        plan = self._plan(query)
        if plan == "retrieve_then_answer":
            answer, sources = self.rag.query(query)
            return {"plan": plan, "answer": answer, "sources": sources}
        return {"plan": plan, "answer": "I’m not sure how to handle this request yet.", "sources": []}
