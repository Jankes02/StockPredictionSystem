from typing import TypedDict, Literal


class AgentSignal(TypedDict):
    symbol: str
    agent: str
    signal: int           # -1, 0, 1
    confidence: float     # [0, 1]
    horizon: Literal["intraday", "daily"]
    kind: Literal["trend", "momentum", "volatility", "mean_reversion"]
