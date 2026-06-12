import matplotlib.pyplot as plt
from data_loader import ALGOS, NEON_NODES, COLORS_ADC, COLORS_SDC, OUT, load_neon, search_col, search_vals

OUT.mkdir(parents=True, exist_ok=True)
METRICS = [("latency", "Latency (us)"), ("QPS", "QPS"), ("Recall", "Recall"), ("MPIOverhead_us", "MPI Overhead (us)")]


def plot_algo(name):
    df = load_neon(name)
    sc = search_col(name)
    vals = search_vals(df, name)
    adc = ALGOS[name]["adc_sdc"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} Neon Scaling vs Nodes", fontsize=14)
    for ax, (col, ylab) in zip(axes.flat, METRICS):
        for v in vals:
            sub = df[df[sc] == v]
            if adc:
                for alg, color in [("ADC", COLORS_ADC), ("SDC", COLORS_SDC)]:
                    s = sub[sub["Algorithm"] == alg].sort_values("nodes")
                    ax.plot(s["nodes"], s[col], "o-", color=color, label=f"{alg} {sc}={v}", markersize=5)
            else:
                s = sub.sort_values("nodes")
                ax.plot(s["nodes"], s[col], "o-", label=f"{sc}={v}", markersize=5)
        ax.set_xlabel("Nodes")
        ax.set_ylabel(ylab)
        ax.set_xticks(NEON_NODES)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"1_neon_scaling_{name}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for n in ALGOS:
        plot_algo(n)
    print(f"saved to {OUT}")
