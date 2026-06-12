import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import ALGOS, NEON_NODES, COLORS_ADC, COLORS_SDC, OUT, load_neon, load_x86, search_col

OUT.mkdir(parents=True, exist_ok=True)
NEON_COLORS = {1: "#2ecc71", 2: "#f39c12", 4: "#9b59b6"}


def _get_val(df, sc, v, col, platform, nodes=None, alg=None):
    sub = df[df[sc] == v]
    if platform == "x86":
        sub = sub[sub["platform"] == "X86"]
    else:
        sub = sub[(sub["platform"] == "Neon") & (sub["nodes"] == nodes)]
    if alg:
        sub = sub[sub["Algorithm"] == alg]
    return sub[col].values[0] if len(sub) else None


def plot_algo_lines(name):
    x86 = load_x86(name)
    neon = load_neon(name)
    df = pd.concat([x86, neon], ignore_index=True)
    sc = search_col(name)
    common = sorted(set(x86[sc]) & set(neon[sc]))
    adc = ALGOS[name]["adc_sdc"]

    if adc:
        slots = [
            ("x86", None, "ADC", COLORS_ADC, "", "X86 ADC"),
            ("x86", None, "SDC", COLORS_SDC, "//", "X86 SDC"),
        ]
        for n in NEON_NODES:
            slots += [
                ("neon", n, "ADC", NEON_COLORS[n], "", f"Neon n={n} ADC"),
                ("neon", n, "SDC", NEON_COLORS[n], "//", f"Neon n={n} SDC"),
            ]
    else:
        slots = [("x86", None, None, "#95a5a6", "", "X86")]
        for n in NEON_NODES:
            slots.append(("neon", n, None, NEON_COLORS[n], "", f"Neon nodes={n}"))

    n_slots = len(slots)
    x = np.arange(len(common))
    group_w = 0.82
    bar_w = group_w / n_slots

    for col, ylab in [("latency", "Latency (us)"), ("QPS", "QPS"), ("Recall", "Recall")]:
        fw = max(8, len(common) * (2.2 if adc else 1.8))
        fig, ax = plt.subplots(figsize=(fw, 6))
        fig.suptitle(f"{name} {ylab}: X86 vs Neon", fontsize=13)

        for si, (plat, nodes, alg, color, hatch, label) in enumerate(slots):
            vals = []
            for v in common:
                val = _get_val(df, sc, v, col, plat, nodes, alg)
                vals.append(val if val is not None else 0)
            offset = x + (si - (n_slots - 1) / 2) * bar_w
            bars = ax.bar(offset, vals, bar_w * 0.92, color=color, hatch=hatch,
                          edgecolor="black", linewidth=0.4, label=label)

        ax.set_xlabel(sc)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in common])
        ax.set_xlim(-0.6, len(common) - 0.4)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=7, ncol=2 if adc else 1, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / f"2_x86_vs_neon_{name}_{col}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    for n in ALGOS:
        plot_algo_lines(n)
    print(f"saved to {OUT}")
