import os
import matplotlib.pyplot as plt
import numpy as np
from data_loader import (
    load_profiler_details, ensure_fig_dir, ADC_STAGES, SDC_STAGES, STAGE_LABELS,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _pct_table(df, method, threads, stages):
    sub = df[(df["Method"] == method) & (df["Threads"] == threads)]
    if sub.empty:
        return None
    nprobes = sorted(sub["NProbe"].unique())
    mat = []
    for p in nprobes:
        row = sub[sub["NProbe"] == p].set_index("Stage")["Avg_us"]
        vals = [float(row.get(s, 0.0)) for s in stages]
        total = sum(vals) or 1.0
        mat.append([v / total * 100 for v in vals])
    labels = [STAGE_LABELS.get(s, s) for s in stages]
    return {"nprobes": nprobes, "mat": np.array(mat), "labels": labels}


def _draw_stacked(ax, data, title):
    nprobes, mat, labels = data["nprobes"], data["mat"], data["labels"]
    x = np.arange(len(nprobes))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    bottom = np.zeros(len(nprobes))
    for j, lab in enumerate(labels):
        ax.bar(x, mat[:, j], bottom=bottom, label=lab, color=cmap[j], edgecolor="white", linewidth=0.3)
        bottom += mat[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in nprobes])
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("NProbe")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))


def plot_stage_breakdown(profiler=None, threads=4, out_dir=None):
    df = profiler if profiler is not None else load_profiler_details()
    out_dir = out_dir or ensure_fig_dir()
    if df.empty:
        print("[Error] no profiler_detail_* data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    adc = _pct_table(df, "ADC", threads, ADC_STAGES)
    sdc = _pct_table(df, "SDC", threads, SDC_STAGES)
    if adc:
        _draw_stacked(axes[0, 0], adc, f"ADC Stage Breakdown (T={threads})")
        mid = len(adc["nprobes"]) // 2
        axes[1, 0].pie(
            adc["mat"][mid], labels=adc["labels"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 7},
        )
        axes[1, 0].set_title(f"ADC @ nprobe={adc['nprobes'][mid]}")
    if sdc:
        _draw_stacked(axes[0, 1], sdc, f"SDC Stage Breakdown (T={threads})")
        mid = len(sdc["nprobes"]) // 2
        axes[1, 1].pie(
            sdc["mat"][mid], labels=sdc["labels"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 7},
        )
        axes[1, 1].set_title(f"SDC @ nprobe={sdc['nprobes'][mid]}")

    plt.tight_layout()
    out = os.path.join(out_dir, f"stage_breakdown_T{threads}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")

    for t in sorted(df["Threads"].unique()):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        for method, stages, scan_key in [
            ("ADC", ADC_STAGES, "5_FastScan_ADC"),
            ("SDC", SDC_STAGES, "5_FastScan_SDC"),
        ]:
            d = _pct_table(df, method, t, stages)
            if d is None:
                continue
            si = stages.index(scan_key)
            ax.plot(d["nprobes"], d["mat"][:, si], "o-", linewidth=2, label=f"{method} Scan")
        ax.set_xlabel("NProbe")
        ax.set_ylabel("Scan Stage Share (%)")
        ax.set_title(f"Scan Stage Share vs NProbe (T={t})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"scan_ratio_T{t}.png")
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        print(f"[OK] {p}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    plot_stage_breakdown(threads=4)
