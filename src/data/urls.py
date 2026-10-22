"""URL parsing / normalisation helpers. Pure offline string processing (never touches the network)."""
from __future__ import annotations
import re
from .psl import get_psl

_SCHEME = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9+.-]*):/{0,2}\s*")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

def split_url(url: str):
    """Return (scheme, host, path_query_fragment). Tolerates 'http:// host' (space after //)."""
    s = (url or "").strip()
    m = _SCHEME.match(s)
    scheme = ""
    if m and ("//" in s[: m.end()] or s[:m.end()].lower().startswith(("http", "ftp"))):
        scheme = m.group(1).lower(); s = s[m.end():]
    cut = len(s)
    for ch in "/?#":
        i = s.find(ch)
        if i != -1: cut = min(cut, i)
    auth, rest = s[:cut], s[cut:]
    if "@" in auth: auth = auth.rsplit("@", 1)[1]
    host = auth
    if host.startswith("["):
        host = host.split("]")[0] + "]"
    elif ":" in host:
        host = host.split(":", 1)[0]
    return scheme, host.strip().lower().rstrip("."), rest

def hostname(url: str) -> str:
    return split_url(url)[1]

def is_ip_host(host: str) -> bool:
    return bool(_IPV4.match(host)) or host.startswith("[")

def registered_domain(url_or_host: str, is_host=False, include_private=False) -> str:
    h = url_or_host if is_host else hostname(url_or_host)
    if not h: return ""
    if is_ip_host(h): return h
    return get_psl(include_private).registered_domain(h)

def normalise_url(url: str) -> str:
    """Canonical key for duplicate / conflict detection: lowercase, no scheme/whitespace/'www.'/trailing '/'."""
    scheme, host, rest = split_url(url)
    if host.startswith("www."): host = host[4:]
    key = host + rest.strip()
    key = re.sub(r"\s+", "", key)
    return key.rstrip("/").lower()
