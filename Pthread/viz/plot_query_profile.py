import os
import matplotlib.pyplot as plt
import numpy as np
from data_loader import (
    load_query_profiler, ensure_fig_dir, ADC_STAGES, SDC_STAGES,
    STAGE_LABELS, QUERY_THREADS, QUERY_NPROBE,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _stage_series(df, stages):
    row = df.set_index("Stage")["Avg_us"]
    vals = [float(row.get(s, 0.0)) for s in stages]
    labels = [STAGE_LABELS.get(s, s) for s in stages]
    total = sum(vals) or 1.0
    pct = [v / total * 100.0 for v in vals]
    return labels, vals, pct, total


def plot_query_profile(threads=QUERY_THREADS, nprobe=QUERY_NPROBE, out_dir=None):
    out_dir = out_dir or ensure_fig_dir()
    adc_df = load_query_profiler("ADC", threads, nprobe)
    sdc_df = load_query_profiler("SDC", threads, nprobe)
    if adc_df.empty and sdc_df.empty:
        print(f"[Error] missing profiler_detail_*_T{threads}_P{nprobe}.csv")
        return

    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])

    for idx, (method, df, stages) in enumerate([
        ("ADC", adc_df, ADC_STAGES),
        ("SDC", sdc_df, SDC_STAGES),
    ]):
        if df.empty:
            continue
        labels, vals, pct, total = _stage_series(df, stages)
        order = np.argsort(pct)[::-1]
        sl = [labels[i] for i in order]
        sp = [pct[i] for i in order]
        sv = [vals[i] for i in order]

        ax = fig.add_subplot(gs[idx, 0])
        y = np.arange(len(sl))
        ax.barh(y, sp, color=plt.cm.Set3(np.linspace(0, 1, len(sl))))
        ax.set_yticks(y)
        ax.set_yticklabels(sl, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Share (%)")
        ax.set_title(f"{method} Query Stage Share (T={threads}, nprobe={nprobe})")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.axvline(100.0 / len(sl), color="gray", linestyle=":", alpha=0.5)

        bottleneck = sl[0]
        print(f"[{method}] avg latency/query: {total:.1f} us | bottleneck: {bottleneck} ({sp[0]:.1f}%)")

        axp = fig.add_subplot(gs[idx, 1])
        nz = [(l, p) for l, p in zip(labels, pct) if p > 0.5]
        axp.pie([p for _, p in nz], labels=[l for l, _ in nz], autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 7})
        axp.set_title(f"{method} Distribution")

    if not adc_df.empty and not sdc_df.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=150)
        al, _, ap, _ = _stage_series(adc_df, ADC_STAGES)
        sl, _, sp, _ = _stage_series(sdc_df, SDC_STAGES)
        keys = sorted(set(al) | set(sl))
        ai = {l: p for l, p in zip(al, ap)}
        si = {l: p for l, p in zip(sl, sp)}
        x = np.arange(len(keys))
        w = 0.38
        ax2.bar(x - w / 2, [ai.get(k, 0) for k in keys], w, label="ADC", color="#1f77b4")
        ax2.bar(x + w / 2, [si.get(k, 0) for k in keys], w, label="SDC", color="#ff7f0e")
        ax2.set_xticks(x)
        ax2.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Share (%)")
        ax2.set_title(f"ADC vs SDC Query Stage Share (T={threads}, nprobe={nprobe})")
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"query_compare_T{threads}_P{nprobe}.png")
        plt.savefig(p2, bbox_inches="tight")
        plt.close()
        print(f"[OK] {p2}")

    plt.tight_layout()
    out = os.path.join(out_dir, f"query_profile_T{threads}_P{nprobe}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    plot_query_profile()
