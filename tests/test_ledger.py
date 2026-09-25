"""Accounting: billable time, success, and running totals."""
import time

import pytest

from gpuctl import ledger, state
from gpuctl.state import Deployment

HOUR = 3600.0


def dep(instance_id=1, *, dph=1.0, created_ago=HOUR, ran=None, served=False,
        witness=False, reason="manual", **kw):
    now = time.time()
    created = now - created_ago
    d = Deployment(
        instance_id=instance_id, recipe="build-a", model="m/x", served_name="x",
        offer_id=1, port=8000, serve_key="k", created_at=created,
        ttl_hours=3.0, dph_at_launch=dph, gpu_label="2x RTX 5090", **kw)
    if served:
        d.served_at = created + 600
    if witness:
        d.notes["linked_model_id"] = "x"
    if ran is not None:
        d.close(reason, now=created + ran)
    state.save(d)
    return d


# -------------------------------------------------------- the cost clock stops


def test_cost_stops_at_teardown():
    """Regression: accrued_cost used `now - created_at` with no upper bound, so a
    box that ran 34 minutes reported $353 of spend two weeks later."""
    d = dep(created_ago=14 * 24 * HOUR, ran=0.57 * HOUR, dph=1.016)
    assert d.ran_seconds() == pytest.approx(0.57 * HOUR)
    assert d.accrued_cost() == pytest.approx(0.58, abs=0.01)


def test_live_cost_still_accrues():
    d = dep(created_ago=2 * HOUR, dph=1.5)
    assert d.accrued_cost() == pytest.approx(3.0, abs=0.02)


def test_final_cost_is_frozen_not_recomputed():
    d = dep(created_ago=HOUR, ran=HOUR, dph=2.0)
    assert d.final_cost == pytest.approx(2.0, abs=0.01)
    d.dph_at_launch = 99.0            # rate changing later must not rewrite history
    assert d.accrued_cost() == pytest.approx(2.0, abs=0.01)


def test_close_records_the_reason():
    d = dep(ran=HOUR, reason="stalled")
    assert d.end_reason == "stalled"
    assert "stalled" in d.outcome


# ------------------------------------------------------------------- success


def test_served_at_marks_success():
    d = dep(created_ago=HOUR, ran=HOUR, served=True)
    assert d.succeeded
    assert d.time_to_serve == pytest.approx(600, abs=1)
    assert d.outcome == "served"


def test_link_witness_survives_teardown():
    """Teardown clears linked_at (it means 'currently linked'), so the ledger
    relies on notes['linked_model_id'], which link_opencode writes only after a
    successful probe and nothing clears."""
    d = dep(created_ago=HOUR, ran=HOUR, witness=True)
    assert d.linked_at is None
    assert d.succeeded
    assert d.time_to_serve is None, "we know it served, not exactly when"


def test_never_served_is_reported_as_such():
    d = dep(created_ago=HOUR, ran=HOUR, reason="ttl")
    assert not d.succeeded
    assert d.outcome == "never served (ttl)"


# ------------------------------------------------------------------- summary


def test_summary_totals_and_split():
    dep(1, dph=1.0, created_ago=2 * HOUR, ran=HOUR, served=True)
    dep(2, dph=2.0, created_ago=2 * HOUR, ran=0.5 * HOUR)          # failed
    dep(3, dph=1.0, created_ago=0.25 * HOUR)                        # still live
    s = ledger.summarise(state.load_all(include_destroyed=True))

    assert s.launched == 3
    assert s.served == 1 and s.failed == 2
    assert s.success_rate == pytest.approx(1 / 3)
    assert s.cost_served == pytest.approx(1.0, abs=0.02)
    assert s.cost_wasted == pytest.approx(1.0 + 0.25, abs=0.02)
    assert s.total_cost == pytest.approx(s.cost_served + s.cost_wasted, abs=0.01)
    assert s.live == 1 and s.live_dph == pytest.approx(1.0)


def test_cost_per_working_box_counts_the_failures():
    dep(1, dph=1.0, created_ago=HOUR, ran=HOUR, served=True)
    dep(2, dph=1.0, created_ago=HOUR, ran=HOUR)
    s = ledger.summarise(state.load_all(include_destroyed=True))
    assert s.mean_cost_per_success == pytest.approx(2.0, abs=0.05)


def test_median_time_to_serve_ignores_failures():
    dep(1, created_ago=HOUR, ran=HOUR, served=True)
    dep(2, created_ago=HOUR, ran=HOUR)
    s = ledger.summarise(state.load_all(include_destroyed=True))
    assert s.median_time_to_serve == pytest.approx(600, abs=2)
    assert len(s.times_to_serve) == 1


def test_empty_summary_is_safe():
    s = ledger.summarise([])
    assert s.launched == 0
    assert s.success_rate is None
    assert s.mean_cost_per_success is None
    assert s.median_time_to_serve is None
    assert s.wasted_fraction is None


# --------------------------------------------------------------- reconcile


class FakeClient:
    def __init__(self, credit=20.79, rows=None, fail=False):
        self.credit = credit
        self.rows = rows if rows is not None else [
            {"type": "payment", "is_credit": True, "amount": -25.0}]
        self.fail = fail

    def account(self):
        if self.fail:
            from gpuctl.vast import VastError
            raise VastError("nope")
        return {"credit": self.credit}

    def invoices(self):
        return self.rows


def test_reconcile_derives_actual_spend():
    r = ledger.reconcile(FakeClient())
    assert r.credit_remaining == pytest.approx(20.79)
    assert r.credits_added == pytest.approx(25.0)
    assert r.actual_spend == pytest.approx(4.21, abs=0.01)


def test_reconcile_skips_refunded_credits():
    r = ledger.reconcile(FakeClient(rows=[
        {"is_credit": True, "amount": -25.0},
        {"is_credit": True, "amount": -10.0, "refunded": True},
    ]))
    assert r.credits_added == pytest.approx(25.0)


def test_reconcile_without_invoices_reports_credit_only():
    r = ledger.reconcile(FakeClient(rows=[]))
    assert r.credits_added is None
    assert r.actual_spend is None


def test_reconcile_degrades_when_the_api_fails():
    assert ledger.reconcile(FakeClient(fail=True)) is None
