import pandas as pd

from ids_ml.context_features import add_context_features


def _flows(rows):
    return pd.DataFrame(rows, columns=["timestamp", "src_ip", "dst_ip", "dst_port", "rst_flag_count"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"], unit="s"))


def test_scan_like_source_gets_many_distinct_ports_and_only_past_flows_count():
    rows = [(i * 0.01, "A", "V", 1000 + i, 1) for i in range(50)] + [(0.5, "B", "V", 80, 0)]
    out = add_context_features(_flows(rows), windows=(2.0,))
    last_a = out[out["src_ip"] == "A"].iloc[-1]
    first_a = out[out["src_ip"] == "A"].iloc[0]
    assert last_a["src_dports_2s"] == 50 and last_a["src_rst_frac_2s"] == 1.0
    assert first_a["src_flows_2s"] == 1  # causal: only itself at the first flow
    b = out[out["src_ip"] == "B"].iloc[0]
    assert b["src_flows_2s"] == 1 and b["dst_srcs_2s"] == 2


def test_window_expires_old_flows():
    rows = [(0.0, "A", "V", 80, 0), (1.0, "A", "V", 81, 0), (5.0, "A", "V", 82, 0)]
    out = add_context_features(_flows(rows), windows=(2.0,))
    assert list(out.sort_values("timestamp")["src_flows_2s"]) == [1, 2, 1]
