import os
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_tradeoff, ensure_fig_dir, DATA_DIR

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_speedup(tradeoff=None, out_dir=None):
    df = tradeoff if tradeoff is not None else load_tradeoff()
    out_dir = out_dir or ensure_fig_dir()
    if df.empty:
        print("[Error] ivfpq_tradeoff.csv is empty")
        return

    nprobes = sorted(df["NProbe"].unique())
    methods = ["ADC", "SDC"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(nprobes)))

    for ax, method in zip(axes, methods):
        sub = df[df["Method"] == method]
        base = sub[sub["Threads"] == 1].set_index("NProbe")["Latency(us)"]
        for i, p in enumerate(nprobes):
            if p not in base.index:
                continue
            b = base.loc[p]
            cur = sub[sub["NProbe"] == p].sort_values("Threads")
            speedup = b / cur["Latency(us)"].values
            ax.plot(cur["Threads"], speedup, "o-", color=colors[i], linewidth=2, markersize=7, label=f"nprobe={p}")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(f"{method} Speedup (baseline T=1)")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup")
        ax.set_xticks(sorted(sub["Threads"].unique()))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out = os.path.join(out_dir, "speedup_vs_threads.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    plot_speedup()
