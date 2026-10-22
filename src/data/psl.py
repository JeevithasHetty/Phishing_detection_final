"""Minimal Public Suffix List (PSL) implementation (no network, no third-party deps).

Implements the PSL algorithm: normal rules, wildcard rules (*.x) and exception rules (!x).
`registered_domain(host, include_private=False)` returns eTLD+1.
include_private=False ignores the PRIVATE section (e.g. repl.co, workers.dev), so all hosts under
such providers collapse into one group: the conservative choice for domain-disjoint splitting.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

_PSL_PATH = Path(__file__).resolve().parents[2] / "config" / "public_suffix_list.dat"

class PSL:
    def __init__(self, path=_PSL_PATH, include_private=False):
        self.rules, self.wild, self.exc = set(), set(), set()
        priv = False
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("// ===BEGIN PRIVATE DOMAINS"):
                    priv = True
                if not line or line.startswith("//"):
                    continue
                if priv and not include_private:
                    continue
                r = line.split()[0].lower()
                if r.startswith("!"): self.exc.add(r[1:])
                elif r.startswith("*."): self.wild.add(r[2:])
                else: self.rules.add(r)

    def public_suffix(self, host: str) -> str:
        labels = host.split(".")
        n = len(labels)
        best = 1  # default rule "*": last label is the suffix
        for i in range(n):
            cand = ".".join(labels[i:])
            if cand in self.exc:
                return ".".join(labels[i + 1:])
            k = n - i
            if cand in self.rules and k > best:
                best = k
            parent = ".".join(labels[i + 1:])
            if i + 1 < n and parent in self.wild and k > best:
                best = k
        return ".".join(labels[n - best:])

    def registered_domain(self, host: str) -> str:
        if not host: return ""
        labels = host.split(".")
        suf = self.public_suffix(host)
        ns = len(suf.split("."))
        if len(labels) <= ns:
            return host  # host is itself a public suffix / single label
        return ".".join(labels[-(ns + 1):])

@lru_cache(maxsize=2)
def get_psl(include_private: bool = False) -> PSL:
    return PSL(include_private=include_private)
