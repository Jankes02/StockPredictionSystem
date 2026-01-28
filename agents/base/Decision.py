from typing import TypedDict, Literal


class Decision(TypedDict):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    score: float
    confidence: float
    contributing_agents: list[str]
