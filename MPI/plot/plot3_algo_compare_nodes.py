import matplotlib.pyplot as plt
import numpy as np
from data_loader import ALGOS, COLORS_ADC, COLORS_SDC, OUT, load_neon, search_col

OUT.mkdir(parents=True, exist_ok=True)
ALGO_COLORS = {"HNSW": "#1abc9c", "IVF-HNSW": "#e67e22", "HNSW-HNSW": "#8e44ad", "IVF-PQ": "#c0392b"}


def plot_nodes(nodes, metric, ylab):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"Algorithm Compare (nodes={nodes}) - {ylab}", fontsize=14)
    offset = 0
    xticks, xlabels = [], []
    for name, cfg in ALGOS.items():
        df = load_neon(name)
        sub = df[df["nodes"] == nodes]
        sc = search_col(name)
        vals = sorted(sub[sc].unique())
        for v in vals:
            if cfg["adc_sdc"]:
                for alg, hatch in [("ADC", ""), ("SDC", "//")]:
                    row = sub[(sub[sc] == v) & (sub["Algorithm"] == alg)]
                    if len(row):
                        color = COLORS_ADC if alg == "ADC" else COLORS_SDC
                        ax.bar(offset, row[metric].values[0], width=0.7, color=color, hatch=hatch,
                               edgecolor="black", linewidth=0.5)
                        xlabels.append(f"{name}\n{alg}\n{sc}={v}")
                        xticks.append(offset)
                        offset += 1
            else:
                row = sub[sub[sc] == v]
                if len(row):
                    ax.bar(offset, row[metric].values[0], width=0.7, color=ALGO_COLORS[name], edgecolor="black", linewidth=0.5)
                    xlabels.append(f"{name}\n{sc}={v}")
                    xticks.append(offset)
                    offset += 1
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=0)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3, axis="y")
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=ALGO_COLORS[k], label=k) for k in ALGOS if not ALGOS[k]["adc_sdc"]]
    legend += [Patch(facecolor=COLORS_ADC, label="IVF-PQ ADC"), Patch(facecolor=COLORS_SDC, hatch="//", label="IVF-PQ SDC")]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"3_algo_compare_nodes{nodes}_{metric}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for nodes in [1, 2, 4]:
        plot_nodes(nodes, "latency", "Latency (us)")
        plot_nodes(nodes, "QPS", "QPS")
    print(f"saved to {OUT}")
