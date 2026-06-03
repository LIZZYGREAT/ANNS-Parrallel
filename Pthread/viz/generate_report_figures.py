#!/usr/bin/env python3
"""Generate all experiment-report figures into timestamped directories."""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from aggregate_runs import build_all, collect_evolution_best
from data_loader import (
    load_yaml_config,
    load_datasets,
    parallel_tradeoff,
    processed_root,
    figures_root,
    report_figures_root,
    stages_for,
    STAGE_LABELS,
    collapse_stages,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_scalability(tradeoff, out_dir, nprobe_highlight=32, error_bars=False):
    methods = ["ADC", "SDC"]
    frameworks = [f for f in tradeoff["Framework"].unique() if f != "unknown"]

    # --- latency lines ---
    fig, axes = plt.subplots(len(frameworks), 2, figsize=(14, 4.5 * max(1, len(frameworks))), dpi=150, squeeze=False)
    for fi, fw in enumerate(frameworks):
        sub_fw = tradeoff[tradeoff["Framework"] == fw]
        for mi, method in enumerate(methods):
            ax = axes[fi, mi]
            sub = sub_fw[sub_fw["Method"] == method]
            nprobes = sorted(sub["NProbe"].unique())
            colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(nprobes)))
            for i, p in enumerate(nprobes):
                cur = sub[sub["NProbe"] == p].sort_values("Threads")
                y = cur["Latency(us)"].values
                if error_bars and "Latency_std" in cur.columns:
                    yerr = cur["Latency_std"].fillna(0).values
                    ax.errorbar(cur["Threads"], y, yerr=yerr, fmt="o-", color=colors[i], lw=2, ms=6, capsize=3, label=f"P={p}")
                else:
                    ax.plot(cur["Threads"], y, "o-", color=colors[i], lw=2, ms=6, label=f"P={p}")
            ax.set_title(f"{fw} {method}: Latency vs Threads")
            ax.set_xlabel("Threads")
            ax.set_ylabel("Latency (us)")
            ax.set_xticks(sorted(sub["Threads"].unique()))
            ax.grid(True, ls="--", alpha=0.45)
            ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "1_scalability", "latency_vs_threads.png"))

    # --- speedup ---
    fig, axes = plt.subplots(len(frameworks), 2, figsize=(14, 4.5 * max(1, len(frameworks))), dpi=150, squeeze=False)
    for fi, fw in enumerate(frameworks):
        sub_fw = tradeoff[tradeoff["Framework"] == fw]
        for mi, method in enumerate(methods):
            ax = axes[fi, mi]
            sub = sub_fw[sub_fw["Method"] == method]
            base = sub[sub["Threads"] == 1].set_index("NProbe")["Latency(us)"]
            nprobes = sorted(sub["NProbe"].unique())
            colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(nprobes)))
            for i, p in enumerate(nprobes):
                if p not in base.index:
                    continue
                b = float(base.loc[p])
                cur = sub[sub["NProbe"] == p].sort_values("Threads")
                sp = b / cur["Latency(us)"].values
                ax.plot(cur["Threads"], sp, "s-", color=colors[i], lw=2, ms=6, label=f"P={p}")
            ax.axhline(1.0, color="gray", ls="--", alpha=0.6)
            ideal = np.array(sorted(sub["Threads"].unique()), dtype=float)
            ax.plot(ideal, ideal / ideal[0], "k:", alpha=0.35, label="Ideal")
            ax.set_title(f"{fw} {method}: Speedup (T=1 baseline)")
            ax.set_xlabel("Threads")
            ax.set_ylabel("Speedup")
            ax.set_xticks(sorted(sub["Threads"].unique()))
            ax.grid(True, ls="--", alpha=0.45)
            ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "1_scalability", "speedup_vs_threads.png"))

    # --- Amdahl-style: fixed nprobe ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for fw in frameworks:
        for method in methods:
            sub = tradeoff[
                (tradeoff["Framework"] == fw)
                & (tradeoff["Method"] == method)
                & (tradeoff["NProbe"] == nprobe_highlight)
            ].sort_values("Threads")
            if sub.empty:
                continue
            b = sub[sub["Threads"] == 1]["Latency(us)"]
            if b.empty:
                continue
            sp = float(b.iloc[0]) / sub["Latency(us)"].values
            ax.plot(sub["Threads"], sp, "o-", lw=2, label=f"{fw}-{method}")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup @ nprobe={nprobe_highlight}")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.45)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "1_scalability", f"speedup_nprobe{nprobe_highlight}.png"))


def plot_algorithm_evolution(cfg, out_dir):
    evo = cfg.get("evolution", {})
    p = int(evo.get("nprobe", 32))
    t = int(evo.get("threads", 4))
    method = evo.get("method", "ADC")
    points = collect_evolution_best(cfg)
    if not points:
        print("[Warn] no evolution best points")
        return

    palette = ["#bab0ac", "#f28e2b", "#4e79a7", "#e15759", "#76b7b2", "#59a14f"]
    labels, vals, colors, notes = [], [], [], []
    for i, pt in enumerate(points):
        labels.append(pt["label"].replace(" + ", "\n+ "))
        vals.append(pt["latency"])
        colors.append(palette[i % len(palette)])
        notes.append(pt["run_name"])

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Latency (us/query)")
    ax.set_title(f"Algorithm Evolution — best run ({method}, nprobe={p}, T={t})")
    ax.grid(axis="y", ls="--", alpha=0.45)
    for xi, v, note in zip(x, vals, notes):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, -0.12, note, ha="center", va="top", fontsize=6, color="#555",
                transform=ax.get_xaxis_transform(), rotation=12)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "2_algorithm", "version_evolution.png"))

    slabels, svals, scolors = [], [], []
    for i, pt in enumerate(points):
        prof = pt["profiler"]
        if prof.empty:
            continue
        row = prof.set_index("Stage")["Avg_us"]
        v = float(row.get("5_FastScan_ADC", row.get("5_FastScan_SDC", row.get("3_Probe_Scan", 0))))
        if v <= 0:
            continue
        slabels.append(pt["label"].replace(" + ", "\n+ "))
        svals.append(v)
        scolors.append(palette[i % len(palette)])
    if slabels:
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.bar(np.arange(len(slabels)), svals, color=scolors)
        ax.set_xticks(np.arange(len(slabels)))
        ax.set_xticklabels(slabels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Profiler Avg (us/query)")
        ax.set_title(f"Scan / Probe Stage — best run ({method}, nprobe={p}, T={t})")
        ax.grid(axis="y", ls="--", alpha=0.45)
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, "2_algorithm", "scan_stage_compare.png"))


def plot_framework_compare(tradeoff, out_dir, threads_list, error_bars=False):
    sub = tradeoff[tradeoff["Threads"].isin(threads_list)]
    if sub.empty:
        return
    for t in threads_list:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        for mi, method in enumerate(["ADC", "SDC"]):
            ax = axes[mi]
            s = sub[(sub["Threads"] == t) & (sub["Method"] == method)]
            nprobes = sorted(s["NProbe"].unique())
            x = np.arange(len(nprobes))
            w = 0.35
            for i, fw in enumerate(["OpenMP", "Pthread"]):
                fw_s = s[s["Framework"] == fw]
                ys, yerr = [], []
                for p in nprobes:
                    row = fw_s[fw_s["NProbe"] == p]
                    ys.append(float(row["Latency(us)"].iloc[0]) if len(row) else np.nan)
                    yerr.append(float(row["Latency_std"].iloc[0]) if error_bars and len(row) and "Latency_std" in row else 0.0)
                ax.bar(x + (i - 0.5) * w, ys, width=w, label=fw, alpha=0.88,
                       yerr=yerr if error_bars else None, capsize=3, ecolor="#333")
            ax.set_xticks(x)
            ax.set_xticklabels([str(p) for p in nprobes])
            ax.set_xlabel("NProbe")
            ax.set_ylabel("Latency (us)")
            ax.set_title(f"{method} @ T={t}")
            ax.legend()
            ax.grid(axis="y", ls="--", alpha=0.45)
        plt.suptitle(f"OpenMP vs Pthread Latency (T={t})", y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(out_dir, "3_framework", f"latency_compare_T{t}.png"))


def plot_adc_sdc_tradeoff(tradeoff, out_dir, cfg=None):
    from plot_adc_sdc import plot_adc_sdc_qps_arm

    cfg = cfg or {}
    nprobe = int(cfg.get("scheduling_compare", {}).get("highlight_nprobe", 32))
    qps_arm = tradeoff[tradeoff["Dataset"] == "qps_arm"] if "Dataset" in tradeoff.columns else tradeoff
    if qps_arm.empty:
        qps_arm = None
    plot_adc_sdc_qps_arm(tradeoff=qps_arm, out_dir=os.path.join(out_dir, "4_adc_sdc"), nprobe=nprobe, cfg=cfg)


def _stacked_stage_pct(prof, dataset_key, method, threads):
    sub = prof[
        (prof["Dataset"] == dataset_key)
        & (prof["Method"] == method)
        & (prof["Threads"] == threads)
    ] if "Dataset" in prof.columns else prof[
        (prof["Framework"] == dataset_key)
        & (prof["Method"] == method)
        & (prof["Threads"] == threads)
    ]
    if sub.empty:
        return None
    fw = str(sub["Framework"].iloc[0])
    stages = stages_for(method, fw)
    nprobes = sorted(sub["NProbe"].unique())
    mat, labels = [], []
    for p in nprobes:
        row = sub[sub["NProbe"] == p].set_index("Stage")["Avg_us"]
        vals = [float(row.get(s, 0.0)) for s in stages]
        total = sum(vals) or 1.0
        mat.append([v / total * 100 for v in vals])
        labels = [STAGE_LABELS.get(s, s) for s in stages]
    return {"nprobes": nprobes, "mat": np.array(mat), "labels": labels}


def plot_profiling(prof, out_dir, threads, nprobe, cfg=None):
    cfg = cfg or {}
    keys = cfg.get("report_compare", ["openmp_x86", "pthread_x86"])
    if "Dataset" in prof.columns:
        prof = prof[prof["Dataset"].isin(keys)]
    else:
        prof = prof[prof["Framework"].isin(["openmp", "pthread", "OpenMP", "Pthread"])]
    for key in keys:
        tag = key.replace("_x86", "").replace("_arm", "")
        for method in ["ADC", "SDC"]:
            data = _stacked_stage_pct(prof, key, method, threads)
            if data is None:
                continue
            fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
            x = np.arange(len(data["nprobes"]))
            cmap = plt.cm.tab20(np.linspace(0, 1, len(data["labels"])))
            bottom = np.zeros(len(data["nprobes"]))
            for j, lab in enumerate(data["labels"]):
                ax.bar(x, data["mat"][:, j], bottom=bottom, label=lab, color=cmap[j], edgecolor="white", lw=0.3)
                bottom += data["mat"][:, j]
            ax.set_xticks(x)
            ax.set_xticklabels([str(p) for p in data["nprobes"]])
            ax.set_ylabel("Share (%)")
            ax.set_xlabel("NProbe")
            ax.set_title(f"{tag} {method} Stage Breakdown (T={threads})")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            _save(fig, os.path.join(out_dir, "5_profiling", f"stacked_{tag}_T{threads}_{method}.png"))

    # framework compare @ fixed T,P
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for mi, method in enumerate(["ADC", "SDC"]):
        ax = axes[mi]
        groups, vals, colors = [], [], []
        palette = {"openmp": "#4e79a7", "pthread": "#e15759"}
        for fw_key, fw_name in [("openmp_x86", "openmp"), ("pthread_x86", "pthread")]:
            sub = prof[
                (prof["Dataset"] == fw_key)
                & (prof["Method"] == method)
                & (prof["Threads"] == threads)
                & (prof["NProbe"] == nprobe)
            ] if "Dataset" in prof.columns else prof[
                (prof["Framework"] == fw_name)
                & (prof["Method"] == method)
                & (prof["Threads"] == threads)
                & (prof["NProbe"] == nprobe)
            ]
            if sub.empty:
                continue
            row = sub.set_index("Stage")["Avg_us"]
            c = collapse_stages(row, fw_name, method)
            bucket_keys = ["Coarse", "Probe Pipeline", "Merge+Rerank"] if fw_name == "pthread" else [
                "Coarse", "LUT/Residual", "FastScan", "Merge+Rerank"]
            for b in bucket_keys:
                v = c.get(b, 0.0)
                if v <= 0:
                    continue
                groups.append(f"{fw_name}\n{b}")
                vals.append(v)
                colors.append(palette[fw_name])
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("Avg us/query")
        ax.set_title(f"{method} @ T={threads} P={nprobe}")
        ax.grid(axis="y", ls="--", alpha=0.45)
    plt.suptitle("OpenMP (fine stages) vs Pthread (probe pipeline)", y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "5_profiling", f"framework_buckets_T{threads}_P{nprobe}.png"))


def write_manifest(out_dir, meta, tradeoff, profiler):
    path = os.path.join(out_dir, "DATA_SOURCES.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("IVF-PQ Report Figure Data Sources\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mode: {meta.get('mode', 'unknown')}\n")
        if meta.get("mode") == "aggregated":
            f.write(f"Processed: {meta.get('processed_dir')}\n")
            for k, m in meta.get("aggregation", {}).items():
                f.write(f"  [{k}] n_sources={m.get('n_sources', m.get('n_runs'))} "
                        f"variant={m.get('variant')} sources={m.get('sources', m.get('run_dirs'))}\n")
        else:
            f.write(f"OpenMP run: {meta.get('openmp_run')}\n")
            f.write(f"Pthread run: {meta.get('pthread_run')}\n")
        f.write(f"Tradeoff rows: {len(tradeoff)}\n")
        f.write(f"Profiler rows: {len(profiler)}\n")
        if "N_runs" in tradeoff.columns:
            f.write(f"Avg runs per point: {tradeoff['N_runs'].mean():.1f}\n")
        f.write("\nSubdirectories:\n")
        f.write("  1_scalability/   latency & speedup\n")
        f.write("  2_algorithm/     IVF-PQ → FastScan → 优化版\n")
        f.write("  3_framework/     OpenMP vs Pthread\n")
        f.write("  4_adc_sdc/       Recall-Latency curves\n")
        f.write("  5_profiling/     stage stacked bars\n")
    print(f"[OK] {path}")


def mirror_to_processed(meta, combined_root):
    import shutil
    if meta.get("mode") != "aggregated":
        return
    proc = meta.get("processed_dir")
    for key in meta.get("datasets", []):
        dst = os.path.join(proc, key, "figures")
        os.makedirs(dst, exist_ok=True)
        for root, _, files in os.walk(combined_root):
            rel = os.path.relpath(root, combined_root)
            if rel == ".":
                rel = ""
            for fn in files:
                if not fn.endswith(".png"):
                    continue
                src = os.path.join(root, fn)
                dst_dir = os.path.join(proc, key, "figures", rel)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, fn))
        print(f"[Mirror] {os.path.join(proc, key, 'figures')}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report figures")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip aggregation, use single run")
    parser.add_argument("--single-run", action="store_true", help="Load latest single run only")
    parser.add_argument("--out", default=None, help="Output figures directory")
    args = parser.parse_args()

    cfg = load_yaml_config()
    do_agg = not args.no_aggregate and not args.single_run
    if do_agg:
        print("[Step 1] Aggregating runs...")
        build_all(cfg)

    print("[Step 2] Loading data & plotting...")
    all_keys = list(cfg.get("datasets", {}).keys())
    tradeoff, profiler, meta = load_datasets(cfg, dataset_keys=all_keys)
    if tradeoff.empty:
        print("[Error] No data. Run: python3 viz/aggregate_runs.py")
        sys.exit(1)

    out_dir = args.out or report_figures_root(meta)
    plot_cfg = cfg.get("plot", {})
    err = bool(plot_cfg.get("error_bars", True))
    parallel = parallel_tradeoff(tradeoff, cfg)

    print(f"[Out] {out_dir}")
    print(f"[Mode] {meta.get('mode')}")

    plot_scalability(parallel, out_dir, error_bars=err)
    plot_algorithm_evolution(cfg, out_dir)
    plot_framework_compare(parallel, out_dir, plot_cfg.get("compare_threads", [4, 8]), error_bars=err)
    plot_adc_sdc_tradeoff(tradeoff, out_dir, cfg)
    if not profiler.empty:
        plot_profiling(
            profiler,
            out_dir,
            int(plot_cfg.get("profile_threads", 4)),
            int(plot_cfg.get("profile_nprobe", 32)),
            cfg,
        )

    write_manifest(out_dir, meta, tradeoff, profiler)
    mirror_to_processed(meta, out_dir)
    print(f"\n[Done] figures -> {out_dir}")
    if do_agg:
        print(f"        aggregated CSV -> {processed_root(cfg)}")


if __name__ == "__main__":
    main()
