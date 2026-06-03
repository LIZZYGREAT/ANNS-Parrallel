#!/usr/bin/env python3
"""ADC vs SDC figures from qps_arm batch benchmark (latency & QPS)."""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]

METHOD_COLOR = {"ADC": "#1f77b4", "SDC": "#ff7f0e"}
METHOD_MK = {"ADC": "o", "SDC": "s"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def load_qps_arm_tradeoff(cfg=None, data_root=None):
    root = data_root or os.path.join(os.path.dirname(__file__), "..", "data")
    proc = os.path.join(root, "processed_data", "qps_arm", "ivfpq_tradeoff_agg.csv")
    if os.path.isfile(proc):
        df = pd.read_csv(proc)
        for old, new in [("Latency_mean", "Latency(us)"), ("Recall_mean", "Recall@10"), ("QPS_mean", "QPS")]:
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        return df[df["Method"].isin(["ADC", "SDC"])].copy()

    pattern = os.path.join(root, "qps_arm", "*", "ivfpq_tradeoff.csv")
    runs = sorted(glob.glob(pattern))
    if not runs:
        return pd.DataFrame()
    return pd.read_csv(runs[-1])


def plot_adc_sdc_qps_arm(tradeoff=None, out_dir=None, nprobe=32, cfg=None, data_root=None):
    df = tradeoff if tradeoff is not None else load_qps_arm_tradeoff(cfg, data_root)
    if df.empty:
        print("[Warn] no qps_arm tradeoff data")
        return

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "..", "report", "images", "4_adc_sdc")
    sub = df[df["NProbe"] == nprobe].copy()
    if sub.empty:
        print(f"[Warn] no rows for nprobe={nprobe}")
        return

    threads = sorted(sub["Threads"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for method in ("ADC", "SDC"):
        cur = sub[sub["Method"] == method].sort_values("Threads")
        if cur.empty:
            continue
        t = cur["Threads"].values
        lat = cur["Latency(us)"].astype(float).values
        qps = cur["QPS"].astype(float).values if "QPS" in cur.columns else 1e6 / lat
        axes[0].plot(t, lat, f"{METHOD_MK[method]}-", color=METHOD_COLOR[method], lw=2.2, ms=7, label=method)
        axes[1].plot(t, qps, f"{METHOD_MK[method]}-", color=METHOD_COLOR[method], lw=2.2, ms=7, label=method)
        for xi, la, qp in zip(t, lat, qps):
            axes[0].annotate(f"{la:.0f}", (xi, la), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
            axes[1].annotate(f"{qp/1000:.1f}k", (xi, qp), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    axes[0].set_xlabel("Threads")
    axes[0].set_ylabel("Latency (us/query)")
    axes[0].set_title(f"Latency @ nprobe={nprobe} (qps_arm Batch)")
    axes[0].set_xticks(threads)
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.45)

    axes[1].set_xlabel("Threads")
    axes[1].set_ylabel("QPS")
    axes[1].set_title(f"Throughput @ nprobe={nprobe} (qps_arm Batch)")
    axes[1].set_xticks(threads)
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.45)

    plt.suptitle("ARM Batch/QPS: ADC vs SDC", y=1.02, fontsize=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "adc_sdc_qps_arm_threads.png"))

    t_fix = 4
    sub_t = df[df["Threads"] == t_fix].sort_values("NProbe")
    nprobes = sorted(sub_t["NProbe"].unique())
    x = np.arange(len(nprobes))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for mi, method in enumerate(("ADC", "SDC")):
        cur = sub_t[sub_t["Method"] == method].set_index("NProbe")
        lat = [float(cur.loc[p, "Latency(us)"]) if p in cur.index else np.nan for p in nprobes]
        qps = [float(cur.loc[p, "QPS"]) if p in cur.index and "QPS" in cur.columns else np.nan for p in nprobes]
        axes[0].bar(x + (mi - 0.5) * w, lat, width=w, label=method, color=METHOD_COLOR[method], alpha=0.88)
        axes[1].bar(x + (mi - 0.5) * w, qps, width=w, label=method, color=METHOD_COLOR[method], alpha=0.88)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(p) for p in nprobes])
    axes[0].set_xlabel("NProbe")
    axes[0].set_ylabel("Latency (us/query)")
    axes[0].set_title(f"Latency @ T={t_fix}")
    axes[0].legend()
    axes[0].grid(axis="y", ls="--", alpha=0.45)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(p) for p in nprobes])
    axes[1].set_xlabel("NProbe")
    axes[1].set_ylabel("QPS")
    axes[1].set_title(f"QPS @ T={t_fix}")
    axes[1].legend()
    axes[1].grid(axis="y", ls="--", alpha=0.45)
    plt.suptitle("ARM Batch/QPS: ADC vs SDC across nprobe", y=1.02, fontsize=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "adc_sdc_qps_arm_nprobe.png"))


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    plot_adc_sdc_qps_arm()
