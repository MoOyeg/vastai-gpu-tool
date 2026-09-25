"""Phase machine: Vast's container status is not the same as 'model is serving'."""
import time

import pytest

from gpuctl import health, state, track
from gpuctl.state import Deployment
from gpuctl.track import Phase, snapshot

PORTS = {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}


def dep(**kw):
    base = dict(instance_id=999, recipe="build-a", model="m/x", served_name="x",
                offer_id=1, port=8000, serve_key="k", created_at=time.time(),
                ttl_hours=3.0, dph_at_launch=0.44, gpu_label="2x RTX 3090")
    base.update(kw)
    d = Deployment(**base)
    state.save(d)
    return d


class FakeClient:
    def __init__(self, inst):
        self.inst = inst

    def get_instance(self, _id):
        return self.inst


def inst(status, msg="", ports=None, ip="1.2.3.4"):
    return {"id": 999, "actual_status": status, "status_msg": msg,
            "dph_total": 0.44, "start_date": time.time() - 60,
            "public_ipaddr": ip, "ports": ports or {}}


@pytest.mark.parametrize("status,msg,expected", [
    ("loading", "pulling image", Phase.LOADING),
    ("created", "", Phase.LOADING),
    ("exited", "", Phase.STOPPED),
    ("stopped", "", Phase.STOPPED),
    ("error", "provisioning failed", Phase.ERROR),
])
def test_status_maps_to_phase(status, msg, expected):
    assert snapshot(FakeClient(inst(status, msg)), dep()).phase is expected


def test_apt_output_is_not_an_error():
    """Regression: Vast streams the image build log through status_msg, and the
    apt package name 'liberror-perl' contains the substring 'error'. Only
    actual_status may decide failure."""
    noisy = "Get:17 http://archive.ubuntu.com/ubuntu noble/main amd64 liberror-perl all 0.17029-2"
    assert snapshot(FakeClient(inst("loading", noisy)), dep()).phase is Phase.LOADING
    assert snapshot(FakeClient(inst("running", noisy, PORTS)), dep()).phase is not Phase.ERROR


def test_running_without_port_mapping_is_not_ready():
    snap = snapshot(FakeClient(inst("running", ports={})), dep())
    assert snap.phase is Phase.RUNNING
    assert snap.endpoint is None


def test_running_but_vllm_silent_is_not_serving(monkeypatch):
    monkeypatch.setattr(track, "probe",
                        lambda *a, **k: health.Probe(False, False, [], "refused"))
    snap = snapshot(FakeClient(inst("running", ports=PORTS)), dep())
    assert snap.phase is Phase.RUNNING
    assert snap.endpoint == "http://1.2.3.4:33526"


def test_serving_only_when_models_answer(monkeypatch):
    monkeypatch.setattr(track, "probe",
                        lambda *a, **k: health.Probe(True, True, ["x"], "ok"))
    assert snapshot(FakeClient(inst("running", ports=PORTS)), dep()).phase is Phase.SERVING


def test_missing_instance_is_gone_once_old():
    old = dep(created_at=time.time() - 3600)
    assert snapshot(FakeClient(None), old).phase is Phase.GONE


def test_ttl_and_cost():
    d = dep(created_at=time.time() - 4 * 3600, ttl_hours=3.0)
    assert d.expired()
    assert abs(d.accrued_cost() - 4 * 0.44) < 0.02
    assert not dep(instance_id=1001, ttl_hours=0).expired(), "ttl=0 means no deadline"
