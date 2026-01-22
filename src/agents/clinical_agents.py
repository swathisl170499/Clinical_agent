# clinical_agent/src/agents/clinical_agents.py
from __future__ import annotations

from typing import List

from Clinical_agent.src.agents.agent_base import AgentContext, AgentResult, BaseAgent
from Clinical_agent.src.rag.hybrid_retriever import HybridRetriever
from Clinical_agent.src.rag.rag_pipeline import ClinicalRAGPipeline


class HybridRetrievalAgent(BaseAgent):
    name = "hybrid_retriever"
    capabilities = ["retrieve", "hybrid_search"]

    def __init__(self, retriever: HybridRetriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def run(self, context: AgentContext) -> AgentResult:
        documents = self.retriever.retrieve(context.question, k=self.k)
        context.documents = documents
        return AgentResult(output={"documents": documents}, message=f"Retrieved {len(documents)} documents.")


class ClinicalAnswerAgent(BaseAgent):
    name = "clinical_answer"
    capabilities = ["answer", "rag_generation"]

    def __init__(self, pipeline: ClinicalRAGPipeline):
        self.pipeline = pipeline

    def run(self, context: AgentContext) -> AgentResult:
        answer = self.pipeline.generate(context.question, context.documents, mode="answer")
        context.answer = answer
        return AgentResult(output={"answer": answer}, message="Generated clinical answer.")


class ClinicalSummaryAgent(BaseAgent):
    name = "clinical_summary"
    capabilities = ["summarize", "rag_generation"]

    def __init__(self, pipeline: ClinicalRAGPipeline):
        self.pipeline = pipeline

    def run(self, context: AgentContext) -> AgentResult:
        answer = self.pipeline.generate(context.question, context.documents, mode="summary")
        context.answer = answer
        return AgentResult(output={"summary": answer}, message="Generated clinical summary.")
