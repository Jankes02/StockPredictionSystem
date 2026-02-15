from typing import List, Literal, TypedDict


class ContributingAgent(TypedDict):
    agent: str
    signal: int
    confidence: float
    kind: str


class Decision(TypedDict):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    score: float
    confidence: float
    contributing_agents: List[ContributingAgent]
