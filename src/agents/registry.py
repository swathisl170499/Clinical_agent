# clinical_agent/src/agents/registry.py
from __future__ import annotations

from typing import Dict, Iterable

from Clinical_agent.src.agents.agent_base import BaseAgent


class AgentRegistry:
    def __init__(self, agents: Iterable[BaseAgent] | None = None):
        self._agents: Dict[str, BaseAgent] = {}
        if agents:
            for agent in agents:
                self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered")
        return self._agents[name]

    def list(self) -> Dict[str, BaseAgent]:
        return dict(self._agents)
