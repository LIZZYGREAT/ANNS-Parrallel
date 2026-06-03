#!/usr/bin/env python3
"""Per-Query task parallel vs Batch (QPS) data parallel — x86 / ARM comparison figures."""
import os
import sys
import shutil
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from aggregate_runs import aggregate_dataset, load_config
from data_loader import (
    load_yaml_config,
    load_datasets,
    figures_root,
    processed_root as agg_processed_root,
    scheduling_dataset_keys,
    scheduling_tradeoff,
    scheduling_label,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]

PALETTE = {"Per-Query": "#4e79a7", "Batch/QPS": "#e15759"}
METHOD_STYLE = {"ADC": ("o", "-"), "SDC": ("s", "--")}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def _add_labels(sub):
    sub = sub.copy()
    sub["SchedLabel"] = sub.apply(scheduling_label, axis=1)
    return sub


def _yerr(col_std):
    return col_std.fillna(0).values if col_std is not None else None


def plot_latency_vs_threads(sub, arch, out_dir, error_bars=True):
    sub = _add_labels(sub)
    methods = [m for m in ("ADC", "SDC") if m in sub["Method"].values]
    labels = [lb for lb in ("Per-Query", "Batch/QPS") if lb in sub["SchedLabel"].values]
    if not labels:
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5), dpi=150, squeeze=False)
    for mi, method in enumerate(methods):
        ax = axes[0, mi]
        msub = sub[sub["Method"] == method]
        nprobes = sorted(msub["NProbe"].unique())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(nprobes)))
        for si, sched in enumerate(labels):
            for pi, p in enumerate(nprobes):
                cur = msub[(msub["SchedLabel"] == sched) & (msub["NProbe"] == p)].sort_values("Threads")
                if cur.empty:
                    continue
                y = cur["Latency(us)"].values
                ls = METHOD_STYLE[method][1] if sched == "Per-Query" else ":"
                mk = METHOD_STYLE[method][0]
                c = colors[pi]
                label = f"{sched} P={p}" if si == 0 or pi == 0 else None
                if error_bars and "Latency_std" in cur.columns:
                    ax.errorbar(
                        cur["Threads"], y,
                        yerr=_yerr(cur["Latency_std"]),
                        fmt=f"{mk}{ls}", color=c, lw=1.8, ms=5, capsize=2,
                        label=f"{sched} P={p}",
                    )
                else:
                    ax.plot(cur["Threads"], y, f"{mk}{ls}", color=c, lw=1.8, ms=5, label=f"{sched} P={p}")
        ax.set_title(f"{method} — Avg Latency (us/query)")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Latency (us)")
        ax.set_xticks(sorted(msub["Threads"].unique()))
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend(fontsize=6, ncol=2)
    plt.suptitle(f"{arch.upper()}: Per-Query vs Batch @ Latency", y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"1_latency_{arch}", "latency_vs_threads.png"))


def plot_speedup(sub, arch, out_dir, nprobe_highlight=32):
    sub = _add_labels(sub)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for mi, method in enumerate(["ADC", "SDC"]):
        ax = axes[mi]
        msub = sub[(sub["Method"] == method) & (sub["NProbe"] == nprobe_highlight)]
        for sched in ("Per-Query", "Batch/QPS"):
            cur = msub[msub["SchedLabel"] == sched].sort_values("Threads")
            if cur.empty:
                continue
            b = cur[cur["Threads"] == 1]["Latency(us)"]
            if b.empty:
                continue
            base = float(b.iloc[0])
            sp = base / cur["Latency(us)"].values
            ax.plot(cur["Threads"], sp, "o-", lw=2, color=PALETTE.get(sched, "#333"), label=sched)
        ideal = np.array(sorted(sub["Threads"].unique()), dtype=float)
        ax.plot(ideal, ideal / ideal[0], "k:", alpha=0.35, label="Ideal")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"{method} Speedup @ P={nprobe_highlight}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup (T=1 baseline)")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.45)
    plt.suptitle(f"{arch.upper()}: Speedup", y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"2_speedup_{arch}", f"speedup_P{nprobe_highlight}.png"))


def plot_grouped_bars(sub, arch, out_dir, threads_list, nprobe, error_bars=True):
    sub = _add_labels(sub)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    sched_order = [s for s in ("Per-Query", "Batch/QPS") if s in sub["SchedLabel"].values]
    for mi, method in enumerate(["ADC", "SDC"]):
        ax = axes[mi]
        for ti, t in enumerate(threads_list):
            s = sub[(sub["Method"] == method) & (sub["Threads"] == t) & (sub["NProbe"] == nprobe)]
            if s.empty:
                continue
            x = np.arange(len(sched_order)) + ti * (len(sched_order) + 0.5)
            w = 0.38
            ys, yerr = [], []
            for sched in sched_order:
                row = s[s["SchedLabel"] == sched]
                ys.append(float(row["Latency(us)"].iloc[0]) if len(row) else np.nan)
                yerr.append(float(row["Latency_std"].iloc[0]) if error_bars and len(row) and "Latency_std" in row else 0.0)
            ax.bar(
                x, ys, width=w, label=f"T={t}",
                color=[PALETTE.get(s, "#888") for s in sched_order],
                yerr=yerr if error_bars else None, capsize=3, ecolor="#333", alpha=0.9,
            )
        ax.set_xticks(np.arange(len(sched_order)) + (len(threads_list) - 1) * (len(sched_order) + 0.5) / 2)
        ax.set_xticklabels(sched_order)
        ax.set_ylabel("Latency (us)")
        ax.set_title(f"{method} @ P={nprobe}")
        ax.legend()
        ax.grid(axis="y", ls="--", alpha=0.45)
    plt.suptitle(f"{arch.upper()}: Per-Query vs Batch (grouped)", y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"3_bars_{arch}", f"latency_T{'_'.join(map(str, threads_list))}_P{nprobe}.png"))


def plot_qps(sub, arch, out_dir, error_bars=True):
    sub = _add_labels(sub)
    if "QPS" not in sub.columns and "QPS_mean" not in sub.columns:
        sub = sub.copy()
        sub["QPS"] = 1e6 / sub["Latency(us)"]
        sub["QPS_note"] = "derived"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for mi, method in enumerate(["ADC", "SDC"]):
        ax = axes[mi]
        msub = sub[sub["Method"] == method]
        nprobe_h = int(msub["NProbe"].median()) if not msub.empty else 32
        for sched in ("Per-Query", "Batch/QPS"):
            cur = msub[(msub["SchedLabel"] == sched) & (msub["NProbe"] == nprobe_h)].sort_values("Threads")
            if cur.empty:
                cur = msub[msub["SchedLabel"] == sched].sort_values(["NProbe", "Threads"])
                if cur.empty:
                    continue
            ycol = "QPS" if "QPS" in cur.columns else "Latency(us)"
            if ycol == "Latency(us)":
                y = 1e6 / cur["Latency(us)"].values
            else:
                y = cur["QPS"].values
            yerr = None
            if error_bars and ycol == "QPS" and "QPS_std" in cur.columns:
                yerr = _yerr(cur["QPS_std"])
            ax.plot(cur["Threads"], y, "o-", lw=2, ms=6, label=sched, color=PALETTE.get(sched))
        ax.set_title(f"{method} — Throughput")
        ax.set_xlabel("Threads")
        ax.set_ylabel("QPS (batch measured or 1e6/latency)")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.45)
    plt.suptitle(f"{arch.upper()}: QPS / Throughput", y=1.02)
    plt.tight_layout()
    subpath = os.path.join(out_dir, f"4_qps_{arch}", "qps_vs_threads.png")
    _save(fig, subpath)
    shutil.copy2(subpath, os.path.join(out_dir, f"qps_vs_threads_{arch}.png"))


def plot_qps_speedup(sub, arch, out_dir, nprobe_highlight=32):
    sub = _add_labels(sub)
    batch = sub[sub["SchedLabel"] == "Batch/QPS"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for mi, method in enumerate(["ADC", "SDC"]):
        ax = axes[mi]
        cur = batch[(batch["Method"] == method) & (batch["NProbe"] == nprobe_highlight)].sort_values("Threads")
        if cur.empty:
            continue
        if "QPS" in cur.columns:
            q = cur["QPS"].astype(float).values
        else:
            q = (1e6 / cur["Latency(us)"].astype(float)).values
        t1 = cur[cur["Threads"] == 1]
        if t1.empty:
            continue
        base = float(q[cur["Threads"].values == 1][0])
        sp = q / base
        ax.plot(cur["Threads"], sp, "o-", lw=2, ms=7, color=PALETTE["Batch/QPS"], label="Batch/QPS")
        ideal = np.array(sorted(cur["Threads"].unique()), dtype=float)
        ax.plot(ideal, ideal / ideal[0], "k:", alpha=0.35, label="Ideal")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"{method} QPS Speedup @ P={nprobe_highlight}")
        ax.set_xlabel("Threads")
        ax.set_ylabel("QPS Speedup (T=1)")
        ax.set_xticks(sorted(cur["Threads"].unique()))
        ax.legend()
        ax.grid(True, ls="--", alpha=0.45)
    plt.suptitle(f"{arch.upper()}: Batch QPS Speedup", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"4_qps_{arch}", "qps_speedup.png")
    _save(fig, path)
    shutil.copy2(path, os.path.join(out_dir, f"qps_speedup_{arch}.png"))


def plot_recall_latency(sub, arch, out_dir):
    sub = _add_labels(sub)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for method in ("ADC", "SDC"):
        for sched in ("Per-Query", "Batch/QPS"):
            s = sub[(sub["Method"] == method) & (sub["SchedLabel"] == sched)].sort_values("Recall@10")
            if s.empty:
                continue
            mk, ls = METHOD_STYLE[method]
            ax.plot(
                s["Recall@10"], s["Latency(us)"], f"{mk}{ls}",
                color=PALETTE.get(sched), lw=1.8, alpha=0.85,
                label=f"{sched} {method}",
            )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Latency (us)")
    ax.set_title(f"{arch.upper()}: Recall–Latency Trade-off")
    ax.grid(True, ls="--", alpha=0.45)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"5_tradeoff_{arch}", "recall_latency.png"))


def plot_cross_arch(tradeoff, out_dir, threads, nprobe, error_bars=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    panels = [
        ("per_query", "Per-Query", "x86", "per_query_x86"),
        ("per_query", "Per-Query", "arm", "per_query_arm"),
        ("qps", "Batch/QPS", "x86", "qps_x86"),
        ("qps", "Batch/QPS", "arm", "qps_arm"),
    ]
    for ax, (_, title, arch, ds_key) in zip(axes.flat, panels):
        sub = tradeoff[
            (tradeoff["Dataset"] == ds_key)
            & (tradeoff["Threads"] == threads)
            & (tradeoff["NProbe"] == nprobe)
        ]
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(2)
        w = 0.35
        for i, method in enumerate(["ADC", "SDC"]):
            row = sub[sub["Method"] == method]
            y = float(row["Latency(us)"].iloc[0]) if len(row) else np.nan
            ye = float(row["Latency_std"].iloc[0]) if error_bars and len(row) and "Latency_std" in row else 0.0
            ax.bar(i, y, width=0.6, label=method, alpha=0.88, yerr=ye if error_bars else None, capsize=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ADC", "SDC"])
        ax.set_ylabel("Latency (us)")
        ax.set_title(f"{title} ({arch.upper()}) T={threads} P={nprobe}")
        ax.grid(axis="y", ls="--", alpha=0.45)
    plt.suptitle("Cross-Arch Latency (single strategy per panel)", y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "6_cross_arch", f"latency_T{threads}_P{nprobe}.png"))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    groups, vals, colors, yerr = [], [], [], []
    for ds_key, sched, arch in [
        ("per_query_x86", "Per-Query", "x86"),
        ("per_query_arm", "Per-Query", "arm"),
        ("qps_x86", "Batch", "x86"),
        ("qps_arm", "Batch", "arm"),
    ]:
        row = tradeoff[
            (tradeoff["Dataset"] == ds_key)
            & (tradeoff["Method"] == "ADC")
            & (tradeoff["Threads"] == threads)
            & (tradeoff["NProbe"] == nprobe)
        ]
        if row.empty:
            continue
        groups.append(f"{sched}\n{arch}")
        vals.append(float(row["Latency(us)"].iloc[0]))
        colors.append(PALETTE["Per-Query"] if "per_query" in ds_key else PALETTE["Batch/QPS"])
        yerr.append(float(row["Latency_std"].iloc[0]) if error_bars and "Latency_std" in row.columns else 0.0)
    if groups:
        ax.bar(np.arange(len(groups)), vals, color=colors, yerr=yerr if error_bars else None, capsize=3)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_ylabel("Latency (us)")
        ax.set_title(f"ADC @ T={threads} P={nprobe} — All configs (avg over runs)")
        ax.grid(axis="y", ls="--", alpha=0.45)
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, "6_cross_arch", f"adc_all_platforms_T{threads}_P{nprobe}.png"))


def write_manifest(out_dir, meta, tradeoff):
    path = os.path.join(out_dir, "DATA_SOURCES.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Scheduling Compare: Per-Query vs Batch/QPS\n")
        f.write("=" * 44 + "\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Processed: {meta.get('processed_dir')}\n")
        for k, m in meta.get("aggregation", {}).items():
            if not str(k).startswith(("per_query", "qps")):
                continue
            f.write(f"  [{k}] n_sources={m.get('n_sources')} variant={m.get('variant')}\n")
            f.write(f"    runs: {m.get('sources')}\n")
        if "N_runs" in tradeoff.columns:
            f.write(f"Avg runs per point: {tradeoff['N_runs'].mean():.1f}\n")
        f.write("\nSubdirectories:\n")
        f.write("  1_latency_{x86,arm}/\n")
        f.write("  2_speedup_{x86,arm}/\n")
        f.write("  3_bars_{x86,arm}/\n")
        f.write("  4_qps_{x86,arm}/   qps_vs_threads, qps_speedup\n")
        f.write("  5_tradeoff_{x86,arm}/\n")
        f.write("  6_cross_arch/\n")
    print(f"[OK] {path}")


def scheduling_figures_root(cfg=None):
    cfg = cfg or load_yaml_config()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(figures_root(cfg), f"scheduling_{stamp}")
    os.makedirs(root, exist_ok=True)
    return root


def aggregate_scheduling_only(cfg=None):
    cfg = cfg or load_config()
    proc = agg_processed_root(cfg)
    keys = scheduling_dataset_keys(cfg)
    datasets = cfg.get("datasets", {})
    metas = {}
    for name in keys:
        ds = datasets.get(name)
        if not ds:
            continue
        m = aggregate_dataset(name, ds, proc, cfg)
        if m:
            metas[name] = m
    return proc, metas


def main():
    parser = argparse.ArgumentParser(description="Plot Per-Query vs Batch scheduling comparison")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml_config()
    sc = cfg.get("scheduling_compare", {})
    keys = scheduling_dataset_keys(cfg)

    if not args.no_aggregate:
        print("[Step 1] Aggregating scheduling datasets...")
        proc, agg_meta = aggregate_scheduling_only(cfg)
    else:
        proc, agg_meta = agg_processed_root(cfg), {}

    print("[Step 2] Loading & plotting...")
    tradeoff, _, meta = load_datasets(cfg, dataset_keys=keys)
    if tradeoff.empty:
        print("[Error] No scheduling data. Check data/per_query_* and data/qps_*")
        sys.exit(1)

    meta.setdefault("aggregation", {}).update(agg_meta)
    meta["mode"] = "aggregated"
    meta["processed_dir"] = proc

    out_dir = args.out or scheduling_figures_root(cfg)
    err = bool(cfg.get("plot", {}).get("error_bars", True))
    threads_list = sc.get("compare_threads", [4, 8])
    nprobe = int(sc.get("highlight_nprobe", 32))

    print(f"[Out] {out_dir}")

    for arch in ("x86", "arm"):
        sub = scheduling_tradeoff(tradeoff, arch, cfg)
        if sub.empty:
            print(f"[Warn] no data for {arch}")
            continue
        plot_latency_vs_threads(sub, arch, out_dir, error_bars=err)
        plot_speedup(sub, arch, out_dir, nprobe_highlight=nprobe)
        plot_grouped_bars(sub, arch, out_dir, threads_list, nprobe, error_bars=err)
        plot_qps(sub, arch, out_dir, error_bars=err)
        plot_qps_speedup(sub, arch, out_dir, nprobe_highlight=nprobe)
        plot_recall_latency(sub, arch, out_dir)

    for t in threads_list:
        plot_cross_arch(tradeoff, out_dir, t, nprobe, error_bars=err)

    from plot_adc_sdc import plot_adc_sdc_qps_arm

    qps_arm = tradeoff[tradeoff["Dataset"] == "qps_arm"] if "Dataset" in tradeoff.columns else pd.DataFrame()
    adc_dir = os.path.join(out_dir, "4_adc_sdc")
    plot_adc_sdc_qps_arm(
        tradeoff=qps_arm if not qps_arm.empty else None,
        out_dir=adc_dir,
        nprobe=nprobe,
        cfg=cfg,
    )

    write_manifest(out_dir, meta, tradeoff)

    report_img = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "report", "images"))
    if os.path.normpath(out_dir) == report_img:
        print(f"\n[Done] figures -> {out_dir}")
        return
    for root, _, files in os.walk(out_dir):
        rel = os.path.relpath(root, out_dir)
        if rel == ".":
            rel = ""
        for fn in files:
            if not fn.endswith(".png"):
                continue
            src = os.path.join(root, fn)
            dst_dir = os.path.join(report_img, rel) if rel else report_img
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, fn))
    print(f"[Mirror] -> {report_img}")
    print(f"\n[Done] figures -> {out_dir}")


if __name__ == "__main__":
    main()
