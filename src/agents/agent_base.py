# clinical_agent/src/agents/agent_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentContext:
    question: str
    documents: List[str] = field(default_factory=list)
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    output: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class BaseAgent(ABC):
    name: str
    capabilities: List[str]

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError
