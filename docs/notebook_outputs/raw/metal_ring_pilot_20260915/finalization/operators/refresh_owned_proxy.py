"""Refresh only the newly configured owned mapping; no login or allocation."""
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

from ring_common import read, require, save

OPS = Path(__file__).resolve().parent
CLI_PACKAGE = "/home/mechti/.local/share/uv/tools/google-colab-cli/lib/python3.12/site-packages"


def require_mapping(session, name, endpoint):
    require(session is not None and session.name == name and session.endpoint == endpoint,
            "Owned name or endpoint mapping differs; refusing proxy refresh")


def synchronize(store, client, owned, now=time.time):
    name, endpoint = owned["session_name"], owned["endpoint"]
    require_mapping(store.get(name), name, endpoint)
    requested = now()
    matches = [item for item in client.list_assignments() if item.endpoint == endpoint]
    require(len(matches) == 1, "Owned assignment is missing or ambiguous")
    assignment = matches[0]
    require(assignment.accelerator.value == "G4" and assignment.variant.name == "GPU", "Owned hardware differs")
    proxy = assignment.runtime_proxy_info
    observed = now()
    ttl = proxy.token_expires_in_seconds
    require(isinstance(ttl, int) and not isinstance(ttl, bool) and ttl - (observed-requested) >= 120,
            "Issued proxy lifetime is insufficient")
    parsed = urlparse(proxy.url) if isinstance(proxy.url, str) else None
    require(isinstance(proxy.token, str) and bool(proxy.token) and parsed is not None
            and parsed.scheme == "https" and bool(parsed.netloc), "Issued proxy is unusable")
    with store._lock_exclusive() as stream:
        records = store._load_raw(stream)
        current = records.get(name)
        require_mapping(current, name, endpoint)
        changed, url_changed = current.token != proxy.token, current.url != proxy.url
        records[name] = current.model_copy(update={"token": proxy.token, "url": proxy.url})
        store._save_raw(stream, records)
    return dict(status="runtime_proxy_synchronized", session_name=name, endpoint=endpoint,
                token_changed=changed, url_changed=url_changed, token_expires_in_seconds=ttl,
                requested_epoch=requested, observed_epoch=observed, conservative_expiry_epoch=requested+ttl,
                allocation_unchanged=True, kernel_identity_preserved=True)


def main():
    config, owned = read(OPS/"session_config.json"), read(OPS/"owned_session.json")
    require(owned and owned["session_name"] == config["session_name"], "Missing owned runtime identity")
    entries = [json.loads(line) for line in Path(owned["history_file"]).read_text().splitlines()]
    require(any(row.get("event_type") == "session_created" and row.get("endpoint") == owned["endpoint"]
                and row.get("accelerator") == "G4" and row.get("variant") == "GPU" for row in entries),
            "Owned endpoint has no matching creation history")
    sys.path.append(CLI_PACKAGE)
    from colab_cli import auth
    from colab_cli.common import state
    previous, flow = logging.root.manager.disable, auth._run_remote_flow
    def blocked(*args, **kwargs):
        raise ValueError("Interactive authentication is disabled")
    logging.disable(logging.CRITICAL)
    auth._run_remote_flow = blocked
    request = None
    try:
        require_mapping(state.store.get(config["session_name"]), config["session_name"], owned["endpoint"])
        client = state.client
        request = client.session.request
        def bounded(*args, **kwargs):
            kwargs.setdefault("timeout", 30)
            return request(*args, **kwargs)
        client.session.request = bounded
        result = synchronize(state.store, client, owned)
    finally:
        if request is not None:
            client.session.request = request
        auth._run_remote_flow = flow
        logging.disable(previous)
    save(OPS/"runtime_proxy_refresh.json", result)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(main()))
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(json.dumps(dict(status="refresh_failed", error_type=type(exc).__name__, http_status=status,
                              stop_required=True)))
        raise SystemExit(1) from None
