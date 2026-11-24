"""Causal per-host context features for Stage 1.

Per-flow features cannot see behaviour spread over many short flows: a port scan or a
request flood is a pattern ACROSS flows, while each flow looks ordinary (the Suricata
case study and the CSE-CIC-IDS2018 burst/probe flows showed exactly that). These
features describe what a flow's source and destination hosts have been doing in the
recent past, in the spirit of Kitsune's per-host incremental statistics (Mirsky et al.,
NDSS 2018), using only flows that STARTED AT OR BEFORE the current one (causal, so they
can be computed online by the ingestion module):

  for each window W in `windows` (seconds):
    src_flows_W        flows started by this source in (t - W, t]
    src_dports_W       distinct destination ports this source contacted
    src_dsts_W         distinct destination hosts this source contacted
    src_rst_frac_W     share of this source's flows with a TCP RST
    dst_flows_W        flows sent to this destination
    dst_srcs_W         distinct sources that contacted this destination

Identity is never a feature (no addresses or ports), only counts. Caveat for
evaluation: in the CIC datasets most attacks come from one attacker host, so these
features can partly learn "the busy attacker host"; results should be read per family.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

CONTEXT_BASE = ("src_flows", "src_dports", "src_dsts", "src_rst_frac", "dst_flows", "dst_srcs")


def context_columns(windows: Sequence[float] = (2.0, 30.0)) -> List[str]:
    return [f"{b}_{int(w)}s" for w in windows for b in CONTEXT_BASE]


class _Window:
    """Sliding time window of (time, value) per key with distinct-value counts."""

    def __init__(self, width: float) -> None:
        self.width = width
        self.q: Dict[str, deque] = {}
        self.distinct: Dict[str, Counter] = {}
        self.rst: Dict[str, int] = {}

    def add_and_query(self, key, t: float, value, is_rst: bool):
        q = self.q.setdefault(key, deque())
        c = self.distinct.setdefault(key, Counter())
        q.append((t, value, is_rst))
        c[value] += 1
        self.rst[key] = self.rst.get(key, 0) + int(is_rst)
        while q and q[0][0] <= t - self.width:
            _, v, r = q.popleft()
            c[v] -= 1
            if c[v] == 0:
                del c[v]
            self.rst[key] -= int(r)
        return len(q), len(c), self.rst[key] / len(q)


def add_context_features(df: pd.DataFrame, windows: Sequence[float] = (2.0, 30.0),
                         time_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy of `df` (needs src_ip, dst_ip, dst_port, rst_flag_count and a time
    column) with the context columns added, computed in time order."""
    order = np.argsort(pd.to_datetime(df[time_col]).to_numpy(dtype="datetime64[ns]").astype("int64"), kind="mergesort")
    t = pd.to_datetime(df[time_col]).to_numpy(dtype="datetime64[ns]").astype("int64")[order] / 1e9
    src = df["src_ip"].to_numpy()[order]
    dst = df["dst_ip"].to_numpy()[order]
    dport = df["dst_port"].to_numpy()[order]
    rst = (df["rst_flag_count"].to_numpy()[order] > 0)
    cols = context_columns(windows)
    out = np.zeros((len(df), len(cols)))
    for wi, w in enumerate(windows):
        by_src_port, by_src_dst, by_dst = _Window(w), _Window(w), _Window(w)
        base = wi * len(CONTEXT_BASE)
        for i in range(len(t)):
            n, nports, rfrac = by_src_port.add_and_query(src[i], t[i], dport[i], rst[i])
            _, ndsts, _ = by_src_dst.add_and_query(src[i], t[i], dst[i], rst[i])
            dn, nsrcs, _ = by_dst.add_and_query(dst[i], t[i], src[i], rst[i])
            out[i, base:base + 6] = (n, nports, ndsts, rfrac, dn, nsrcs)
    res = df.copy()
    unsorted = np.empty_like(out)
    unsorted[order] = out
    for j, c in enumerate(cols):
        res[c] = unsorted[:, j]
    return res
