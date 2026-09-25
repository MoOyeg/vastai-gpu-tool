"""Hang detection.

A box that wedges while nobody is watching is the expensive failure: it bills at
full rate and looks identical to one that is merely slow. These tests pin the
signals that distinguish the two, including the two false-progress traps found
against a genuinely hung instance.
"""
import time

import pytest

from gpuctl import state, track
from gpuctl.state import Deployment
from gpuctl.track import Phase, progress_marker, snapshot

PORTS = {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}


def dep(**kw):
    base = dict(instance_id=42, recipe="build-a", model="m/x", served_name="x",
                offer_id=1, port=8000, serve_key="k", created_at=time.time(),
                ttl_hours=3.0, dph_at_launch=1.0, gpu_label="2x RTX 5090")
    base.update(kw)
    d = Deployment(**base)
    state.save(d)
    return d


def inst(status="running", msg="success, running", ports=None, disk=30.0, **extra):
    d = {"id": 42, "actual_status": status, "status_msg": msg, "dph_total": 1.0,
         "start_date": time.time() - 600, "public_ipaddr": "1.2.3.4",
         "ports": ports if ports is not None else PORTS, "disk_usage": disk}
    d.update(extra)
    return d


class Client:
    """Fake Vast client. `log` is what the container log currently returns."""

    def __init__(self, instance, log="x" * 100):
        self.instance = instance
        self.log = log
        self.log_calls = 0

    def get_instance(self, _id):
        return self.instance

    def logs(self, _id, tail=200):
        self.log_calls += 1
        return self.log


def age(d, seconds):
    """Pretend the last observed progress was `seconds` ago."""
    d.progress_at = time.time() - seconds
    state.save(d)
    return state.find(d.instance_id)


# ----------------------------------------------------------------- the marker


def test_host_telemetry_is_not_progress():
    """gpu_util/cpu_util/mem_usage change because the host is alive.

    Observed on a hung box: they sat frozen for a minute, then jumped. Counting
    them as progress resets the stall clock forever.
    """
    a = inst(gpu_util=49, cpu_util=0.57, mem_usage=5.98, vmem_usage=19.8)
    b = inst(gpu_util=3, cpu_util=9.1, mem_usage=6.4, vmem_usage=21.0)
    assert progress_marker(a, Phase.RUNNING) == progress_marker(b, Phase.RUNNING)


def test_status_msg_flap_is_not_progress_once_running():
    """Vast rewrites the banner occasionally (observed dropping a '/ssh' suffix)."""
    a = inst(msg="success, running vllm/vllm-openai_latest/ssh")
    b = inst(msg="success, running vllm/vllm-openai_latest")
    assert progress_marker(a, Phase.RUNNING) == progress_marker(b, Phase.RUNNING)


def test_status_msg_is_progress_while_loading():
    """During an image pull it streams real docker layer progress."""
    a = inst(status="loading", msg="aaa: Pull complete", disk=4.0)
    b = inst(status="loading", msg="bbb: Pull complete", disk=5.0)
    assert progress_marker(a, Phase.LOADING) != progress_marker(b, Phase.LOADING)


def test_disk_growth_is_progress():
    assert progress_marker(inst(disk=30.0), Phase.RUNNING) != \
        progress_marker(inst(disk=31.0), Phase.RUNNING)


# -------------------------------------------------------------- the detector


def test_first_look_never_stalls():
    c = Client(inst())
    assert snapshot(c, dep()).phase is Phase.RUNNING


def test_static_box_with_static_log_is_stalled():
    d = dep()
    c = Client(inst())
    snapshot(c, d)                                   # establish the marker
    d = age(state.find(42), 700)                     # RUNNING threshold is 600s
    d.log_size = 100; d.log_checked_at = 0.0; state.save(d)
    snap = snapshot(c, state.find(42))
    assert snap.phase is Phase.STALLED
    assert "no progress" in snap.detail
    assert "log not growing" in snap.detail


def test_growing_log_rescues_a_static_box():
    """Telemetry can be frozen while vLLM is genuinely working."""
    d = dep()
    c = Client(inst(), log="x" * 5000)
    snapshot(c, d)
    d = age(state.find(42), 700)
    d.log_size = 100; d.log_checked_at = 0.0; state.save(d)
    snap = snapshot(c, state.find(42))
    assert snap.phase is Phase.RUNNING
    assert snap.stalled_for == 0.0
    assert state.find(42).log_size == 5000


def test_moving_marker_resets_the_clock():
    d = dep()
    c = Client(inst(status="loading", msg="layer 1", disk=4.0, ports={}))
    snapshot(c, d)
    d = age(state.find(42), 5000)                    # LOADING threshold is 1800s
    c.instance = inst(status="loading", msg="layer 2", disk=9.0, ports={})
    assert snapshot(c, state.find(42)).phase is Phase.LOADING


def test_serving_is_never_stalled(monkeypatch):
    from gpuctl import health
    monkeypatch.setattr(track, "probe",
                        lambda *a, **k: health.Probe(True, True, ["x"], "ok"))
    d = dep()
    c = Client(inst())
    snapshot(c, d)
    d = age(state.find(42), 99999)
    assert snapshot(c, state.find(42)).phase is Phase.SERVING


def test_log_fetches_are_rate_limited():
    """A tight watch loop must not pay for a log fetch on every poll."""
    d = dep()
    c = Client(inst())
    snapshot(c, d)
    for _ in range(5):
        d = age(state.find(42), 700)
        d.log_size = 100
        d.log_checked_at = time.time()               # just checked
        state.save(d)
        snapshot(c, state.find(42))
    assert c.log_calls == 0, "should have respected LOG_CHECK_INTERVAL"


def test_log_baseline_does_not_by_itself_condemn():
    """With no prior measurement we have no evidence either way."""
    d = dep()
    c = Client(inst())
    snapshot(c, d)
    d = age(state.find(42), 400)                     # past limit/2, under limit
    d.log_checked_at = 0.0; state.save(d)
    snap = snapshot(c, state.find(42))
    assert snap.phase is Phase.RUNNING
    assert c.log_calls == 1
    assert state.find(42).log_size == 100, "baseline recorded for next time"


def test_log_fetch_failure_is_not_progress():
    class Broken(Client):
        def logs(self, _id, tail=200):
            raise RuntimeError("logs unavailable")

    d = dep()
    c = Broken(inst())
    snapshot(c, d)
    d = age(state.find(42), 700)
    d.log_size = 100; d.log_checked_at = 0.0; state.save(d)
    assert snapshot(c, state.find(42)).phase is Phase.STALLED


def test_detection_can_be_switched_off():
    d = dep()
    c = Client(inst())
    snapshot(c, d)
    d = age(state.find(42), 9999)
    assert snapshot(c, state.find(42), detect_stall=False).phase is Phase.RUNNING


def test_stalled_counts_as_wasting_money():
    assert Phase.STALLED.wasting_money
    assert Phase.ERROR.wasting_money
    assert not Phase.LOADING.wasting_money
    assert not Phase.SERVING.wasting_money


# ------------------------------------------------- provisioning trouble signals


@pytest.mark.parametrize("msg,expected", [
    ("curl: (6) Could not resolve host: cloud.vast.ai", True),
    ("failed to pull: manifest unknown", True),
    ("write /var/lib/docker: no space left on device", True),
    ("toomanyrequests: rate limit exceeded", True),
    # must NOT match: the apt package name that fooled an earlier version
    ("Get:17 .../liberror-perl all 0.17029-2 [25.6 kB]", False),
    # must NOT match: apt's own wording is "Could not resolve '<host>'", and a
    # transient apt DNS hiccup during package install is not provisioning death
    ("E: Could not resolve 'archive.ubuntu.com'", False),
    ("success, running vllm/vllm-openai_latest/ssh", False),
    ("0db1731e65c7: Download complete", False),
    ("", False),
    (None, False),
])
def test_suspicious_status_matching_is_narrow(msg, expected):
    assert bool(track.suspicious_status(msg)) is expected


def test_trouble_shortens_the_threshold_but_does_not_condemn():
    """A host seen emitting the DNS error recovered and pulled the image, so a
    match must never by itself mark the deployment failed."""
    d = dep()
    c = Client(inst(status="loading", msg="curl: (6) Could not resolve host: cloud.vast.ai",
                    ports={}, disk=0.0))
    snap = snapshot(c, d)
    assert snap.phase is Phase.LOADING, "must not be condemned on sight"

    # LOADING's normal threshold is 1800s; trouble shortens it to 480s.
    d = age(state.find(42), 600)
    d.log_size = 100; d.log_checked_at = time.time(); state.save(d)
    snap = snapshot(c, state.find(42))
    assert snap.phase is Phase.STALLED
    assert "DNS" in snap.detail or "resolve" in snap.detail.lower()


def test_recovering_host_is_not_condemned():
    """Once provisioning moves on, the shortened threshold stops applying."""
    d = dep()
    c = Client(inst(status="loading", msg="curl: (6) Could not resolve host: cloud.vast.ai",
                    ports={}, disk=0.0))
    snapshot(c, d)
    d = age(state.find(42), 600)
    # Vast retried and the image is now pulling.
    c.instance = inst(status="loading", msg="0db1731e65c7: Download complete", ports={}, disk=2.0)
    assert snapshot(c, state.find(42)).phase is Phase.LOADING
