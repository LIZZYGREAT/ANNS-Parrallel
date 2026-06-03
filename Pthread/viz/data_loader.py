import os
import re
import glob
import json
import pandas as pd
import yaml

LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIZ_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(VIZ_DIR, "report_config.yaml")

PROFILER_RE = re.compile(r"profiler_(ADC|SDC)_T(\d+)_P(\d+)\.csv$")

STAGE_LABELS = {
    "1_Coarse_Dist": "Coarse Dist",
    "2_Coarse_Sort": "Coarse Sort",
    "3_Compute_Residual": "Residual",
    "3_Probe_Scan": "Probe LUT+Scan",
    "4_Build_LUT": "Build LUT",
    "4_Quantize_Query": "Quantize",
    "4.5_Build_LUT": "LUT Pack",
    "5_FastScan_ADC": "FastScan",
    "5_FastScan_SDC": "FastScan",
    "6_Local_TopK_Trim": "Local TopK",
    "7_Thread_Merge": "Thread Merge",
    "8_Global_Merge": "Global Merge",
    "8.5_Re_Rank": "Re-rank",
    "9_Build_Result": "Build Result",
}

ADC_STAGES_OMP = [
    "1_Coarse_Dist", "2_Coarse_Sort", "3_Compute_Residual", "4_Build_LUT",
    "5_FastScan_ADC", "6_Local_TopK_Trim", "7_Thread_Merge", "8_Global_Merge",
    "8.5_Re_Rank", "9_Build_Result",
]
SDC_STAGES_OMP = [
    "1_Coarse_Dist", "2_Coarse_Sort", "3_Compute_Residual", "4_Quantize_Query",
    "4.5_Build_LUT", "5_FastScan_SDC", "6_Local_TopK_Trim", "7_Thread_Merge",
    "8_Global_Merge", "8.5_Re_Rank", "9_Build_Result",
]
ADC_STAGES_PTHREAD = [
    "1_Coarse_Dist", "2_Coarse_Sort", "3_Probe_Scan",
    "7_Thread_Merge", "8_Global_Merge", "8.5_Re_Rank", "9_Build_Result",
]
SDC_STAGES_PTHREAD = ADC_STAGES_PTHREAD
ADC_STAGES_BASELINE = [
    "1_Coarse_Dist", "2_Coarse_Sort", "3_Compute_Residual", "4_Build_LUT",
    "5_FastScan_ADC", "6_Local_TopK_Trim", "7_Thread_Merge", "8_Global_Merge",
    "8.5_Re_Rank", "9_Build_Result",
]


def load_yaml_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def data_root_path(cfg=None):
    cfg = cfg or load_yaml_config()
    root = cfg.get("data_root", "data")
    return root if os.path.isabs(root) else os.path.join(LAB_ROOT, root)


def processed_root(cfg=None):
    cfg = cfg or load_yaml_config()
    rel = cfg.get("aggregation", {}).get("processed_dir", "processed_data")
    return rel if os.path.isabs(rel) else os.path.join(data_root_path(cfg), rel)


def figures_root(cfg=None):
    cfg = cfg or load_yaml_config()
    rel = cfg.get("plot", {}).get("figures_dir", "figures")
    return rel if os.path.isabs(rel) else os.path.join(data_root_path(cfg), rel)


def load_aggregated_tradeoff(dataset_name, proc_root=None):
    proc_root = proc_root or processed_root()
    path = os.path.join(proc_root, dataset_name, "ivfpq_tradeoff_agg.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "Latency(us)" not in df.columns and "Latency_mean" in df.columns:
        df["Latency(us)"] = df["Latency_mean"]
    if "Recall@10" not in df.columns and "Recall_mean" in df.columns:
        df["Recall@10"] = df["Recall_mean"]
    for c in ("Threads", "NProbe"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    df["Dataset"] = dataset_name
    return df


def load_aggregated_profiler(dataset_name, proc_root=None):
    proc_root = proc_root or processed_root()
    path = os.path.join(proc_root, dataset_name, "profiler_agg.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "Avg_us" not in df.columns and "Avg_us_mean" in df.columns:
        df["Avg_us"] = df["Avg_us_mean"]
    if "Total_us" not in df.columns and "Total_us_mean" in df.columns:
        df["Total_us"] = df["Total_us_mean"]
    df["Dataset"] = dataset_name
    return df


def load_datasets(cfg=None, dataset_keys=None):
    cfg = cfg or load_yaml_config()
    proc = processed_root(cfg)
    keys = dataset_keys or list(cfg.get("datasets", {}).keys())
    trade_parts, prof_parts, metas = [], [], {}
    for key in keys:
        t = load_aggregated_tradeoff(key, proc)
        p = load_aggregated_profiler(key, proc)
        if t.empty:
            continue
        trade_parts.append(t)
        if not p.empty:
            prof_parts.append(p)
        mp = os.path.join(proc, key, "aggregation_meta.json")
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as f:
                metas[key] = json.load(f)
    tradeoff = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    profiler = pd.concat(prof_parts, ignore_index=True) if prof_parts else pd.DataFrame()
    meta = {
        "mode": "aggregated",
        "processed_dir": proc,
        "datasets": keys,
        "aggregation": metas,
        "config": cfg,
    }
    return tradeoff, profiler, meta


def parallel_tradeoff(tradeoff, cfg=None):
    cfg = cfg or load_yaml_config()
    keys = cfg.get("report_compare", ["openmp_x86", "pthread_x86"])
    if "Dataset" in tradeoff.columns:
        return tradeoff[tradeoff["Dataset"].isin(keys)].copy()
    return tradeoff[tradeoff["Framework"].isin(["OpenMP", "Pthread"])].copy()


def scheduling_dataset_keys(cfg=None):
    cfg = cfg or load_yaml_config()
    sc = cfg.get("scheduling_compare", {})
    keys = []
    for arch in ("x86", "arm"):
        keys.extend(sc.get(arch, []))
    return list(dict.fromkeys(keys))


def scheduling_tradeoff(tradeoff, arch, cfg=None):
    cfg = cfg or load_yaml_config()
    keys = cfg.get("scheduling_compare", {}).get(arch, [])
    if not keys or "Dataset" not in tradeoff.columns:
        return pd.DataFrame()
    return tradeoff[tradeoff["Dataset"].isin(keys)].copy()


def scheduling_label(row):
    ds = str(row.get("Dataset", ""))
    if ds.startswith("per_query"):
        return "Per-Query"
    if ds.startswith("qps"):
        return "Batch/QPS"
    return str(row.get("Variant", row.get("Framework", "unknown")))


def report_figures_root(meta=None):
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    keys = "_".join(meta.get("datasets", ["report"])[:4]) if meta else "report"
    root = os.path.join(figures_root(), f"report_{stamp}_{keys}")
    os.makedirs(root, exist_ok=True)
    return root


def stages_for(method, framework_or_variant):
    fw = str(framework_or_variant).lower()
    if "pthread" in fw or framework_or_variant == "Pthread":
        return ADC_STAGES_PTHREAD if method == "ADC" else SDC_STAGES_PTHREAD
    if framework_or_variant in ("OpenMP", "openmp"):
        return ADC_STAGES_OMP if method == "ADC" else SDC_STAGES_OMP
    return ADC_STAGES_BASELINE if method == "ADC" else ADC_STAGES_OMP


def collapse_stages(df_row, framework, method):
    s = df_row
    coarse = float(s.get("1_Coarse_Dist", 0)) + float(s.get("2_Coarse_Sort", 0))
    probe = 0.0
    fw = str(framework).lower()
    if fw == "pthread" or "3_Probe_Scan" in s.index and float(s.get("3_Probe_Scan", 0)) > 0 and fw != "openmp":
        if "3_Probe_Scan" in s.index:
            probe = float(s.get("3_Probe_Scan", 0))
        lut_res, scan = 0.0, 0.0
    else:
        lut_res = float(s.get("3_Compute_Residual", 0)) + float(
            s.get("4_Build_LUT", s.get("4_Quantize_Query", 0))
        ) + float(s.get("4.5_Build_LUT", 0))
        scan = float(s.get("5_FastScan_ADC", s.get("5_FastScan_SDC", 0)))
    merge = (
        float(s.get("6_Local_TopK_Trim", 0))
        + float(s.get("7_Thread_Merge", 0))
        + float(s.get("8_Global_Merge", 0))
        + float(s.get("8.5_Re_Rank", 0))
        + float(s.get("9_Build_Result", 0))
    )
    return {
        "Coarse": coarse,
        "LUT/Residual": lut_res,
        "FastScan": scan,
        "Probe Pipeline": probe,
        "Merge+Rerank": merge,
    }
