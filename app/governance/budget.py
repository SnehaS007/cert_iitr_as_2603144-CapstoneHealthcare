"""Task 15 (Runtime layer) — per-request token / cost budget, enforced twice.

1. Pre-flight: projected = CREW_PROMPT_OVERHEAD_TOKENS + QUERY_TOKEN_FANOUT * tokens(query). A request
   whose projection exceeds the cap is rejected BEFORE any agent runs.
2. Runtime: MockCrewLLM counts the input tokens of every call and raises BudgetExceededError the moment
   the cumulative total passes the cap, so a request can never silently overrun the budget even if the
   projection was wrong. (Agents are built with max_retry_limit=0 so the stop is not retried.)

Tokens are estimated as ceil(chars / 4): deterministic, no tokenizer download (zero-network holds), and
the same estimate MockCrewLLM reports to CrewAI via get_input_tokens_count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import app.config as config


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class BudgetEstimate:
    stage: str  # "preflight" | "runtime"
    tokens: int
    cap: int
    cost_usd: float

    @property
    def within_budget(self) -> bool:
        return self.tokens <= self.cap


class BudgetExceededError(Exception):
    def __init__(self, estimate: BudgetEstimate):
        self.estimate = estimate
        verb = "needs" if estimate.stage == "preflight" else "has consumed"
        super().__init__(
            f"[{estimate.stage}] request {verb} ~{estimate.tokens} tokens (~${estimate.cost_usd:.4f}); "
            f"per-request cap is {estimate.cap} tokens."
        )


class TokenBudget:
    def __init__(
        self,
        cap: int = config.REQUEST_TOKEN_BUDGET,
        overhead_tokens: int = config.CREW_PROMPT_OVERHEAD_TOKENS,
        fanout: int = config.QUERY_TOKEN_FANOUT,
        cost_per_1k_usd: float = config.MOCK_COST_PER_1K_TOKENS_USD,
    ):
        if cap <= overhead_tokens:
            raise ValueError(f"cap ({cap}) must exceed the fixed crew overhead ({overhead_tokens})")
        self.cap, self.overhead_tokens, self.fanout, self.cost_per_1k_usd = cap, overhead_tokens, fanout, cost_per_1k_usd

    def cost(self, tokens: int) -> float:
        return round(tokens / 1000 * self.cost_per_1k_usd, 6)

    def project(self, query: str) -> BudgetEstimate:
        tokens = self.overhead_tokens + self.fanout * estimate_tokens(query)
        return BudgetEstimate("preflight", tokens, self.cap, self.cost(tokens))

    def enforce_preflight(self, query: str) -> BudgetEstimate:
        estimate = self.project(query)
        if not estimate.within_budget:
            raise BudgetExceededError(estimate)
        return estimate

    def enforce_runtime(self, consumed_tokens: int) -> None:
        if consumed_tokens > self.cap:
            raise BudgetExceededError(BudgetEstimate("runtime", consumed_tokens, self.cap, self.cost(consumed_tokens)))
