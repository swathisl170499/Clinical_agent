# clinical_agent/src/agents/multi_agent_orchestrator.py
from typing import Dict, Any, List

from Clinical_agent.src.agents.agent_base import AgentContext
from Clinical_agent.src.agents.clinical_agents import (
    ClinicalAnswerAgent,
    ClinicalSummaryAgent,
    HybridRetrievalAgent,
)
from Clinical_agent.src.agents.registry import AgentRegistry
from Clinical_agent.src.rag.rag_pipeline import ClinicalRAGPipeline

class MultiAgentOrchestrator:
    def __init__(self):
        self.rag = ClinicalRAGPipeline()
        self.registry = AgentRegistry(
            agents=[
                HybridRetrievalAgent(self.rag.retriever, k=5),
                ClinicalAnswerAgent(self.rag),
                ClinicalSummaryAgent(self.rag),
            ]
        )

    def _plan(self, query: str) -> List[str]:
        q = query.lower()
        if any(k in q for k in ["summarize", "summary", "overview", "background"]):
            return ["hybrid_retriever", "clinical_summary"]
        return ["hybrid_retriever", "clinical_answer"]

    def run(self, query: str) -> Dict[str, Any]:
        steps = self._plan(query)
        context = AgentContext(question=query)
        for step in steps:
            agent = self.registry.get(step)
            agent.run(context)
        plan = " -> ".join(steps)
        return {"plan": plan, "answer": context.answer, "sources": context.documents}
