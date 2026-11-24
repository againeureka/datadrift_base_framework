"""Keep a server without an API key local.

Without ``DDOC_API_KEY`` the API trusts whoever reaches it, which is meant
to be the person at this machine. Two browser tricks let any web page
reach a local server: DNS rebinding (the page's own hostname starts
resolving to 127.0.0.1, so the browser treats the API as same-origin and
reads its answers) and plain cross-site POSTs. While no key is set:

* the ``Host`` header must name this machine (localhost, 127.0.0.1, ::1),
  the address the server is bound to, or an entry of ``DDOC_ALLOWED_HOSTS``
  (comma-separated; ``*`` turns the check off);
* a request that changes something (not GET / HEAD / OPTIONS) and carries
  an ``Origin`` header must come from the same host and port.

With a key set, the key does this job (a cross-site page cannot attach
it), so the guard steps aside and reverse proxies that rewrite ``Host``
keep working. ``/healthz`` answers any host, for monitors.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .auth import get_expected_key

LOOPBACK = ("localhost", "127.0.0.1", "::1")
_WILDCARD = ("0.0.0.0", "::", "")
_SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
_OPEN_PATHS = ("/healthz",)


def allowed_hosts(bind_host: str) -> List[str]:
    """Host names a key-less server answers to when bound to ``bind_host``."""
    hosts = list(LOOPBACK)
    bind = hostname(bind_host)
    if bind not in _WILDCARD and bind not in hosts:
        hosts.append(bind)
    hosts += [h.strip().lower() for h in os.getenv("DDOC_ALLOWED_HOSTS", "").split(",") if h.strip()]
    return hosts


def hostname(value: str) -> str:
    """``example.com:8765`` → ``example.com``; ``[::1]:8765`` → ``::1``."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:value.index("]")] if "]" in value else value
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


def _refuse(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=403,
                        content={"status": "error", "error_code": code, "message": message})


def install(app: FastAPI, hosts: Optional[Sequence[str]] = None) -> None:
    """Add the guard. ``hosts=None`` skips the Host check (an app embedded
    by its own server); the Origin check always applies without a key."""
    allowed = None if hosts is None or "*" in hosts else {h.lower() for h in hosts}

    @app.middleware("http")
    async def _local_only(request: Request, call_next):
        if get_expected_key() or request.url.path in _OPEN_PATHS:
            return await call_next(request)
        host = request.headers.get("host", "")
        if allowed is not None and hostname(host) not in allowed:
            return _refuse(
                "host_not_allowed",
                f"Host '{host}' is not served without an API key. Set DDOC_API_KEY to serve "
                "other machines, or add the name to DDOC_ALLOWED_HOSTS.")
        origin = request.headers.get("origin")
        if (origin is not None and request.method not in _SAFE_METHODS
                and urlsplit(origin).netloc.lower() != host.lower()):
            return _refuse(
                "cross_origin_request",
                "Requests from another site are refused without an API key.")
        return await call_next(request)
