# clinical_agent/src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from Clinical_agent.src.agents.multi_agent_orchestrator import MultiAgentOrchestrator

app = FastAPI(title="Clinical Agent API")
orchestrator = MultiAgentOrchestrator()

class QueryIn(BaseModel):
    question: str

class QueryOut(BaseModel):
    plan: str
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryOut)
def query(inb: QueryIn):
    result = orchestrator.run(inb.question)
    return QueryOut(**result)
