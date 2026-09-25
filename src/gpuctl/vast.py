"""Thin, typed client over the Vast.ai v0 REST API.

Endpoints and payload shapes here were read off the official `vastai` SDK
(vastai/api/instances.py, vastai/api/offers.py) rather than guessed:

    POST   /bundles/                     search offers        -> {"offers": [...]}
    PUT    /asks/{offer_id}/             rent an offer        -> {"success", "new_contract"}
    GET    /instances/?owner=me          list own instances   -> {"instances": [...]}
    GET    /instances/{id}/?owner=me     one instance
    DELETE /instances/{id}/              destroy
    PUT    /instances/request_logs/{id}/ ask for logs         -> {"result_url": ...}
    GET    /gpu_names/unique/            canonical GPU names
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .config import INSTANCES_V1_URL, VAST_API_BASE, api_key


class VastError(RuntimeError):
    """A Vast.ai API call failed."""


def normalize_gpu_name(name: str) -> str:
    """Canonicalise a GPU name for a /bundles/ query.

    The API matches `gpu_name` against the exact catalogue string, which
    contains SPACES ("RTX 3090"). The underscore form you see in the CLI and
    in Vast's own API docs example is a shell-quoting convention only --
    sending "RTX_3090" silently returns zero offers. Verified live 2026-09-10.
    """
    return " ".join(name.replace("_", " ").split())


class VastClient:
    MAX_RETRIES = 5

    @staticmethod
    def _retry_after(response: httpx.Response, attempt: int) -> float:
        """Honour Vast's retry_after, falling back to exponential backoff."""
        try:
            hinted = float(response.json().get("retry_after", 0))
        except (ValueError, AttributeError, TypeError):
            hinted = 0.0
        if not hinted:
            try:
                hinted = float(response.headers.get("Retry-After", 0))
            except ValueError:
                hinted = 0.0
        return max(hinted, min(2 ** attempt, 30)) + 0.25

    def __init__(self, key: str | None = None, timeout: float = 60.0) -> None:
        self._key = key or api_key()
        self._gpu_names: list[str] | None = None
        self._http = httpx.Client(
            base_url=VAST_API_BASE,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "VastClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- plumbing

    def _request(self, method: str, path: str, absolute: bool = False, **kw: Any) -> Any:
        url = httpx.URL(path) if absolute else path
        for attempt in range(self.MAX_RETRIES):
            try:
                r = self._http.request(method, url, **kw)
            except httpx.HTTPError as exc:
                raise VastError(f"{method} {path} failed: {exc}") from exc
            if r.status_code != 429:
                break
            # Vast rate-limits bursts and tells us how long to wait.
            delay = self._retry_after(r, attempt)
            if attempt == self.MAX_RETRIES - 1:
                raise VastError(f"{method} {path}: rate limited after {self.MAX_RETRIES} attempts")
            time.sleep(delay)
        if r.status_code == 401:
            raise VastError("Vast.ai rejected the API key (401). Check VAST_API_KEY.")
        if r.status_code >= 400:
            raise VastError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:400]}")
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError as exc:
            raise VastError(f"{method} {path} returned non-JSON: {r.text[:200]}") from exc

    # ------------------------------------------------------------------ offers

    def search_offers(
        self,
        query: dict[str, Any],
        *,
        limit: int = 25,
        order: list[list[str]] | None = None,
        storage_gb: float = 5.0,
        offer_type: str = "on-demand",
    ) -> list[dict[str, Any]]:
        """Search rentable offers. `query` uses Vast's operator syntax, e.g.
        {"gpu_name": {"eq": "RTX_3090"}, "num_gpus": {"eq": 2}}."""
        body: dict[str, Any] = {
            # Defaults the official SDK also applies: only offers you can
            # actually rent right now, on verified, non-external machines.
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
        }
        body.update(query)
        body["type"] = offer_type
        body["order"] = order or [["dph_total", "asc"]]
        body["limit"] = int(limit)
        body["allocated_storage"] = storage_gb
        data = self._request("POST", "/bundles/", json=body)
        return data.get("offers", [])

    def gpu_names(self) -> list[str]:
        if self._gpu_names is None:
            data = self._request("GET", "/gpu_names/unique/")
            names = data.get("gpu_names", data) if isinstance(data, dict) else data
            self._gpu_names = sorted(names) if isinstance(names, list) else []
        return self._gpu_names

    def resolve_gpu_name(self, name: str) -> str:
        """Map a user-supplied GPU name onto the exact catalogue string.

        A name the catalogue does not contain matches zero offers with no
        error, so fail loudly here instead.
        """
        wanted = normalize_gpu_name(name)
        names = self.gpu_names()
        if not names:
            return wanted
        for n in names:
            if n.lower() == wanted.lower():
                return n
        low = wanted.lower()
        close = [n for n in names if low in n.lower() or n.lower() in low]
        hint = f" Did you mean: {', '.join(close[:6])}?" if close else ""
        raise VastError(f"No Vast GPU named {wanted!r}.{hint}")

    # --------------------------------------------------------------- instances

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk_gb: int,
        env: dict[str, str] | None = None,
        onstart: str | None = None,
        label: str | None = None,
        runtype: str = "ssh",
    ) -> int:
        """Rent `offer_id`. Returns the new instance (contract) id."""
        body: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "env": env or {},
            "runtype": runtype,
        }
        if onstart:
            body["onstart"] = onstart
        if label:
            body["label"] = label
        data = self._request("PUT", f"/asks/{offer_id}/", json=body)
        if not data.get("success", False):
            raise VastError(f"Vast refused the rental: {data}")
        contract = data.get("new_contract")
        if contract is None:
            raise VastError(f"Rental succeeded but no instance id came back: {data}")
        return int(contract)

    def list_instances(self) -> list[dict[str, Any]]:
        # /api/v0/instances/ was retired (HTTP 410); listing lives on v1 now.
        # Single-instance GET, create, destroy and logs are still v0.
        data = self._request(
            "GET", INSTANCES_V1_URL, params={"owner": "me"}, absolute=True
        )
        return data.get("instances", [])

    def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        try:
            data = self._request("GET", f"/instances/{instance_id}/", params={"owner": "me"})
        except VastError:
            # Fall back to scanning the list; a destroyed instance 404s here.
            for inst in self.list_instances():
                if int(inst.get("id", -1)) == instance_id:
                    return inst
            return None
        inst = data.get("instances", data)
        if isinstance(inst, list):
            return inst[0] if inst else None
        return inst or None

    def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/instances/{instance_id}/", json={})

    # ------------------------------------------------------------- billing

    def account(self) -> dict[str, Any]:
        """Current user record, including the `credit` balance."""
        return self._request("GET", "/users/current/")

    def invoices(self) -> list[dict[str, Any]]:
        """Payment and billing rows. Note: no per-instance charge rows exist."""
        data = self._request("GET", "/users/current/invoices/")
        rows = data.get("invoices", data) if isinstance(data, dict) else data
        return rows if isinstance(rows, list) else []

    def logs(self, instance_id: int, tail: int = 200) -> str:
        data = self._request(
            "PUT", f"/instances/request_logs/{instance_id}/", json={"tail": str(tail)}
        )
        url = data.get("result_url")
        if not url:
            return str(data)
        # The log blob lands in object storage a moment after the request.
        for attempt in range(30):
            try:
                r = httpx.get(url, timeout=20.0)
                if r.status_code == 200:
                    return r.text
            except httpx.HTTPError:
                pass
            time.sleep(0.4 + attempt * 0.1)
        return "(logs were requested but did not materialise in time)"


# ------------------------------------------------------------------- helpers


def endpoint_for(instance: dict[str, Any], internal_port: int) -> str | None:
    """Resolve the public http://ip:port for a container port.

    Vast maps each requested container port to a random external port on the
    machine's shared public IP. The mapping arrives in the instance JSON in
    Docker's shape: {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40021"}]}
    """
    ip = (instance.get("public_ipaddr") or "").strip()
    ports = instance.get("ports") or {}
    mapping = ports.get(f"{internal_port}/tcp")
    if not ip or not mapping:
        return None
    host_port = mapping[0].get("HostPort")
    if not host_port:
        return None
    return f"http://{ip}:{host_port}"


def ssh_target(instance: dict[str, Any]) -> tuple[str, int] | None:
    host = (instance.get("ssh_host") or "").strip()
    port = instance.get("ssh_port")
    if not host or not port:
        return None
    return host, int(port)
