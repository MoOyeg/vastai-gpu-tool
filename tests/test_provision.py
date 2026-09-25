"""Launch payloads: port requests, secret handling, endpoint resolution."""
from gpuctl import provision, recipes
from gpuctl.vast import endpoint_for, normalize_gpu_name


def test_port_is_requested_through_the_env_map():
    """Vast's env map doubles as docker run options; '-p 8000:8000' opens a port."""
    env = provision.build_env(port=8000, serve_key="sk-vast-SECRET")
    assert env["-p 8000:8000"] == "1"
    assert env["VLLM_API_KEY"] == "sk-vast-SECRET"


def test_serving_key_is_not_written_into_the_onstart_text():
    """Vast stores onstart; reference the key by env var instead of inlining it."""
    onstart = provision.build_onstart(recipes.get("build-a"), port=8000)
    assert "SECRET" not in onstart
    assert '--api-key "$VLLM_API_KEY"' in onstart


def test_onstart_stays_within_vast_limit():
    for key, r in recipes.all_recipes().items():
        assert len(provision.build_onstart(r, port=8000)) <= provision.ONSTART_LIMIT, key


def test_endpoint_resolves_from_docker_style_port_map():
    inst = {"public_ipaddr": " 65.130.162.74 ", "ports": {
        "22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "20000"}],
        "8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}}
    assert endpoint_for(inst, 8000) == "http://65.130.162.74:33526"
    assert endpoint_for(inst, 9999) is None
    assert endpoint_for({"public_ipaddr": "1.2.3.4", "ports": {}}, 8000) is None


def test_gpu_name_normalises_to_spaces():
    """Underscored names match zero offers, with HTTP 200 and no error."""
    assert normalize_gpu_name("RTX_3090") == "RTX 3090"
    assert normalize_gpu_name("  RTX   PRO_6000  WS ") == "RTX PRO 6000 WS"
