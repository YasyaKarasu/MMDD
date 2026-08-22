"""Shared helpers for Stage-1 Flask GUI network binding."""

from __future__ import annotations

import socket

LOCALHOST_HOST = "127.0.0.1"
LAN_BIND_HOST = "0.0.0.0"
ALL_INTERFACE_HOSTS = {LAN_BIND_HOST, "::", "[::]"}


def resolve_gui_host(host: str | None, lan: bool = False) -> str:
    """Return the Flask bind host for local-only or LAN-visible GUIs."""
    if host:
        return host
    return LAN_BIND_HOST if lan else LOCALHOST_HOST


def is_all_interface_host(host: str) -> bool:
    return host in ALL_INTERFACE_HOSTS


def guess_lan_ipv4() -> str | None:
    """Best-effort local LAN IPv4 address for a friendlier browser URL."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            return None
        return ip if ip and not ip.startswith("127.") else None


def format_gui_urls(name: str, host: str, port: int) -> str:
    if not is_all_interface_host(host):
        return f"{name}: http://{host}:{port}"
    lan_ip = guess_lan_ipv4()
    lan_url = f"http://{lan_ip}:{port}" if lan_ip else f"http://<this-machine-LAN-IP>:{port}"
    return "\n".join(
        [
            f"{name}: listening on {host}:{port}",
            f"  Local browser: http://127.0.0.1:{port}",
            f"  LAN devices:   {lan_url}",
        ]
    )
