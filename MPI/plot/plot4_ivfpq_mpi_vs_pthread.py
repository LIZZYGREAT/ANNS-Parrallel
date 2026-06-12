import matplotlib.pyplot as plt
from data_loader import COLORS_ADC, COLORS_SDC, OUT, load_ivfpq_mpi_neon, load_ivfpq_pthread

OUT.mkdir(parents=True, exist_ok=True)
THREADS = 4
MPI_NODES = 1
SERIES = [
    ("mpi", "ADC", COLORS_ADC, "-", "o", "MPI ADC"),
    ("mpi", "SDC", COLORS_SDC, "-", "s", "MPI SDC"),
    ("pthread", "ADC", COLORS_ADC, "--", "o", "Pthread ADC"),
    ("pthread", "SDC", COLORS_SDC, "--", "s", "Pthread SDC"),
]


def _plot_tradeoff(ycol, ylab, mpi, pthread):
    fig, ax = plt.subplots(figsize=(9, 6))
    sources = {"mpi": mpi, "pthread": pthread}
    for src, alg, color, ls, marker, label in SERIES:
        sub = sources[src][sources[src]["Algorithm"] == alg].sort_values("Recall")
        ax.plot(sub["Recall"], sub[ycol], linestyle=ls, marker=marker, color=color,
                label=label, markersize=5, linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel(ylab)
    title = f"IVF-PQ {ylab} vs Recall: MPI vs Pthread (Threads={THREADS})"
    if ycol == "QPS":
        title += " (QPS = 10^6 / Latency; MPI includes MPIOverhead)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"4_ivfpq_mpi_vs_pthread_{ycol}_recall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mpi = load_ivfpq_mpi_neon(MPI_NODES, THREADS)
    pthread = load_ivfpq_pthread(THREADS)
    _plot_tradeoff("latency", "Latency (us)", mpi, pthread)
    _plot_tradeoff("QPS", "QPS", mpi, pthread)
    print(f"saved to {OUT}")
