from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent / "files"
OUT = Path(__file__).parent / "output"

ALGOS = {
    "HNSW": {"dir": "HNSW", "x86": "1", "search": "ef_search", "adc_sdc": False},
    "IVF-HNSW": {"dir": "IVF-HNSW", "x86": "8", "search": "NProbe", "adc_sdc": False},
    "HNSW-HNSW": {"dir": "HNSW-HNSW", "x86": "8", "search": "NProbe", "adc_sdc": False},
    "IVF-PQ": {"dir": "IVF-PQ-MPI", "x86": "1024", "search": "NProbe", "adc_sdc": True},
}

NEON_NODES = [1, 2, 4]
COLORS_ADC = "#e74c3c"
COLORS_SDC = "#3498db"


def _norm(df, platform, nodes=None):
    if "LocalLatency_us" in df.columns:
        df["latency"] = df["LocalLatency_us"]
    elif "AvgLocalLatency_us" in df.columns:
        df["latency"] = df["AvgLocalLatency_us"]
    elif "Latency" in df.columns:
        df["latency"] = df["Latency"]
    if "NProbe" in df.columns and "ef_search" not in df.columns:
        if df["NProbe"].max() <= 200 and (df["Algorithm"].str.contains("HNSW").any()):
            df["ef_search"] = df["NProbe"]
    df["platform"] = platform
    if nodes is not None:
        df["nodes"] = nodes
    return df


def load_neon(name):
    cfg = ALGOS[name]
    frames = []
    for n in NEON_NODES:
        p = BASE / cfg["dir"] / "Neon" / str(n) / "results.csv"
        df = pd.read_csv(p)
        df = _norm(df, "Neon", n)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_x86(name):
    cfg = ALGOS[name]
    p = BASE / cfg["dir"] / "X86" / cfg["x86"] / "results.csv"
    df = pd.read_csv(p, header=None, skiprows=1,
                      names=["Algorithm", "Threads", "NProbe", "Recall", "Latency", "QPS"]) if name == "IVF-PQ" else pd.read_csv(p)
    df = _norm(df, "X86")
    return df[df["Threads"] == 4]


def search_col(name):
    return ALGOS[name]["search"]


def search_vals(df, name):
    col = search_col(name)
    if col not in df.columns:
        col = "ef_search"
    return sorted(df[col].unique())


def load_ivfpq_pthread(threads=4):
    df = pd.read_csv(BASE / "IVF-PQ-Pthread" / "ivfpq_tradeoff.csv")
    df = df[df["Threads"] == threads].copy()
    df["Algorithm"] = df["Method"]
    df["Recall"] = df["Recall@10"]
    df["latency"] = df["Latency(us)"]
    df["QPS"] = 1e6 / df["latency"]
    return df.sort_values("NProbe")


def load_ivfpq_mpi_neon(nodes=1, threads=4):
    df = pd.read_csv(BASE / "IVF-PQ-MPI" / "Neon" / str(nodes) / "results.csv")
    df = df[df["Threads"] == threads].copy()
    df = _norm(df, "Neon", nodes)
    if "MPIOverhead_us" in df.columns:
        df["latency_total"] = df["latency"] + df["MPIOverhead_us"]
        df["QPS"] = 1e6 / df["latency_total"]
    return df.sort_values("NProbe")
