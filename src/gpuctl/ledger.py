"""Running account of what we rented, whether it worked, and what it cost.

Cost is our own estimate: billable hours x the hourly rate agreed at launch.
Vast does not expose per-instance charges — its invoice feed carries payments and
aggregated billing, not a row per contract — so there is nothing authoritative to
read back. `reconcile()` therefore cross-checks the *total* against the account
balance, which is the one number Vast will confirm.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .state import Deployment
from .vast import VastClient, VastError


@dataclass
class Summary:
    launched: int = 0
    served: int = 0
    live: int = 0
    total_cost: float = 0.0
    cost_served: float = 0.0        # spend on instances that did serve
    cost_wasted: float = 0.0        # spend on instances that never served
    runtime_hours: float = 0.0
    live_dph: float = 0.0
    times_to_serve: list[float] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.launched - self.served

    @property
    def success_rate(self) -> float | None:
        return self.served / self.launched if self.launched else None

    @property
    def wasted_fraction(self) -> float | None:
        return self.cost_wasted / self.total_cost if self.total_cost else None

    @property
    def median_time_to_serve(self) -> float | None:
        return statistics.median(self.times_to_serve) if self.times_to_serve else None

    @property
    def mean_cost_per_success(self) -> float | None:
        """What a working box really costs, counting the failures along the way."""
        return self.total_cost / self.served if self.served else None


def summarise(deps: list[Deployment], now: float | None = None) -> Summary:
    s = Summary()
    for d in deps:
        s.launched += 1
        cost = d.accrued_cost(now)
        s.total_cost += cost
        s.runtime_hours += d.age_hours(now)
        if d.succeeded:
            s.served += 1
            s.cost_served += cost
            tts = d.time_to_serve
            if tts:
                s.times_to_serve.append(tts)
        else:
            s.cost_wasted += cost
        if not d.destroyed_at:
            s.live += 1
            s.live_dph += d.dph_at_launch
    return s


@dataclass
class Reconciliation:
    credit_remaining: float
    credits_added: float | None

    @property
    def actual_spend(self) -> float | None:
        if self.credits_added is None:
            return None
        return max(0.0, self.credits_added - self.credit_remaining)


def reconcile(client: VastClient) -> Reconciliation | None:
    """Ask Vast what the account balance says, to sanity-check our estimate."""
    try:
        account = client.account()
    except VastError:
        return None
    credit = account.get("credit")
    if not isinstance(credit, (int, float)):
        return None

    added: float | None = None
    try:
        rows = client.invoices()
    except VastError:
        rows = []
    if rows:
        # Credit rows carry the amount as a negative number.
        total = sum(abs(float(r.get("amount") or 0))
                    for r in rows
                    if r.get("is_credit") and not r.get("refunded"))
        added = total or None
    return Reconciliation(float(credit), added)
