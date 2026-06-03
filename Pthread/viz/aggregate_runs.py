#!/usr/bin/env python3
"""Aggregate benchmark runs under data/ (mean/std)."""
import os
import re
import io
import json
import argparse
import glob
from datetime import datetime

import pandas as pd
import yaml

LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIZ_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(VIZ_DIR, "report_config.yaml")

PROFILER_RE = re.compile(r"profiler_(?:detail_)?(ADC|SDC)_T(\d+)_P(\d+)\.csv$")
RUN_DIR_RE = re.compile(r"\d{8}_\d{6}_IVFPQ")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def data_root_path(cfg=None):
    cfg = cfg or load_config()
    root = cfg.get("data_root", "data")
    return root if os.path.isabs(root) else os.path.join(LAB_ROOT, root)


def resolve_path(rel, cfg=None):
    if not rel:
        return ""
    if os.path.isabs(rel):
        return rel
    return os.path.join(data_root_path(cfg), rel)


def discover_run_dirs(root, include=None, exclude=None):
    if not os.path.isdir(root):
        return []
    include = include or []
    exclude = exclude or []
    found = []
    direct = os.path.join(root, "ivfpq_tradeoff.csv")
    if os.path.isfile(direct):
        found.append(root)
    for csv_path in glob.glob(os.path.join(root, "*", "ivfpq_tradeoff.csv")):
        run_dir = os.path.dirname(csv_path)
        if not RUN_DIR_RE.search(os.path.basename(run_dir)):
            continue
        if include and not any(x in run_dir for x in include):
            continue
        if exclude and any(x in run_dir for x in exclude):
            continue
        found.append(run_dir)
    return sorted(set(found), key=lambda p: os.path.basename(p))


def load_tradeoff_file(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
    blocks, cur = [], []
    for line in lines:
        if line.startswith("Method,Threads"):
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
        else:
            if cur:
                cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    if not blocks:
        return pd.read_csv(path)
    dfs = [pd.read_csv(io.StringIO(b)) for b in blocks]
    return pd.concat(dfs, ignore_index=True)


def _normalize_tradeoff(df, run_dir, framework, variant):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df[df["Method"].isin(["ADC", "SDC"])]
    for c in ("Threads", "NProbe"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Latency(us)"] = pd.to_numeric(df["Latency(us)"], errors="coerce")
    df["Recall@10"] = pd.to_numeric(df["Recall@10"], errors="coerce")
    if "QPS" in df.columns:
        df["QPS"] = pd.to_numeric(df["QPS"], errors="coerce")
    df = df.dropna(subset=["Threads", "NProbe", "Latency(us)"])
    df["Threads"] = df["Threads"].astype(int)
    df["NProbe"] = df["NProbe"].astype(int)
    df["Framework"] = framework
    df["Variant"] = variant
    df["RunDir"] = run_dir
    return df


def load_tradeoff_runs(run_dirs, framework, variant):
    frames = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "ivfpq_tradeoff.csv")
        if not os.path.isfile(path):
            continue
        raw = load_tradeoff_file(path)
        frames.append(_normalize_tradeoff(raw, run_dir, framework, variant))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_tradeoff_flat_blocks(root, framework, variant):
    path = os.path.join(root, "ivfpq_tradeoff.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
    blocks, cur = [], []
    for line in lines:
        if line.startswith("Method,Threads"):
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    frames = []
    for i, block in enumerate(blocks):
        df = pd.read_csv(io.StringIO(block))
        run_id = root if len(blocks) == 1 else f"{root}#run{i}"
        frames.append(_normalize_tradeoff(df, run_id, framework, variant))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_dataset_raw_tradeoff(name, ds_cfg, cfg):
    root = resolve_path(ds_cfg.get("root", ""), cfg)
    variant = ds_cfg.get("variant_label", name)
    framework = ds_cfg.get("framework_label", variant)
    if ds_cfg.get("type", "multi_run") == "flat":
        return load_tradeoff_flat_blocks(root, framework, variant)
    run_dirs = discover_run_dirs(
        root, ds_cfg.get("include_runs") or [], ds_cfg.get("exclude_runs") or []
    )
    return load_tradeoff_runs(run_dirs, framework, variant)


def pick_best_tradeoff(raw, method, nprobe, threads):
    sub = raw[
        (raw["Method"] == method)
        & (raw["NProbe"] == int(nprobe))
        & (raw["Threads"] == int(threads))
    ]
    if sub.empty:
        return None
    return sub.loc[sub["Latency(us)"].idxmin()]


def profiler_run_dir(run_dir):
    return str(run_dir).split("#")[0]


def load_profiler_one_run(run_dir, method, threads, nprobe, framework):
    base = profiler_run_dir(run_dir)
    for name in (
        f"profiler_detail_{method}_T{threads}_P{nprobe}.csv",
        f"profiler_{method}_T{threads}_P{nprobe}.csv",
    ):
        fp = os.path.join(base, name)
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp, header=None, names=["Stage", "Total_us", "Avg_us"])
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "Framework": framework,
                "Method": method,
                "Threads": threads,
                "NProbe": nprobe,
                "Stage": str(r["Stage"]).strip(),
                "Avg_us": float(r["Avg_us"]),
                "RunDir": run_dir,
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


def collect_evolution_best(cfg, dataset_keys=None, method=None, nprobe=None, threads=None):
    cfg = cfg or load_config()
    evo = cfg.get("evolution", {})
    keys = dataset_keys or evo.get("datasets", [])
    method = method or evo.get("method", "ADC")
    nprobe = int(nprobe if nprobe is not None else evo.get("nprobe", 32))
    threads = int(threads if threads is not None else evo.get("threads", 4))
    datasets = cfg.get("datasets", {})
    out = []
    for key in keys:
        ds = datasets.get(key)
        if not ds:
            continue
        raw = load_dataset_raw_tradeoff(key, ds, cfg)
        best = pick_best_tradeoff(raw, method, nprobe, threads)
        if best is None:
            continue
        fw = ds.get("framework_label", key)
        fw_key = fw.lower() if str(fw) in ("OpenMP", "Pthread") else "baseline"
        prof = load_profiler_one_run(best["RunDir"], method, threads, nprobe, fw_key)
        run_name = os.path.basename(profiler_run_dir(best["RunDir"]))
        if "#run" in str(best["RunDir"]):
            run_name += " " + str(best["RunDir"]).split("#", 1)[1]
        out.append({
            "dataset": key,
            "label": ds.get("variant_label", key),
            "latency": float(best["Latency(us)"]),
            "recall": float(best["Recall@10"]),
            "run_dir": best["RunDir"],
            "run_name": run_name,
            "profiler": prof,
        })
    return out


def aggregate_tradeoff(raw):
    if raw.empty:
        return pd.DataFrame()
    gcols = ["Variant", "Framework", "Method", "Threads", "NProbe"]
    agg_kw = dict(
        Latency_mean=("Latency(us)", "mean"),
        Latency_std=("Latency(us)", "std"),
        Recall_mean=("Recall@10", "mean"),
        Recall_std=("Recall@10", "std"),
        N_runs=("Latency(us)", "count"),
    )
    if "QPS" in raw.columns:
        agg_kw["QPS_mean"] = ("QPS", "mean")
        agg_kw["QPS_std"] = ("QPS", "std")
    agg = raw.groupby(gcols, as_index=False).agg(**agg_kw)
    agg["Latency_std"] = agg["Latency_std"].fillna(0.0)
    agg["Recall_std"] = agg["Recall_std"].fillna(0.0)
    agg["Latency(us)"] = agg["Latency_mean"]
    agg["Recall@10"] = agg["Recall_mean"]
    if "QPS_mean" in agg.columns:
        agg["QPS_std"] = agg["QPS_std"].fillna(0.0)
        agg["QPS"] = agg["QPS_mean"]
    return agg


def load_profiler_runs(run_dirs, framework):
    rows = []
    for run_dir in run_dirs:
        for pattern in (
            os.path.join(run_dir, "profiler_detail_*_T*_P*.csv"),
            os.path.join(run_dir, "profiler_*_T*_P*.csv"),
        ):
            for fp in glob.glob(pattern):
                m = PROFILER_RE.search(os.path.basename(fp))
                if not m:
                    continue
                method, threads, nprobe = m.group(1), int(m.group(2)), int(m.group(3))
                try:
                    df = pd.read_csv(fp, header=None, names=["Stage", "Total_us", "Avg_us"])
                    for _, r in df.iterrows():
                        rows.append({
                            "Framework": framework,
                            "Method": method,
                            "Threads": threads,
                            "NProbe": nprobe,
                            "Stage": str(r["Stage"]).strip(),
                            "Total_us": float(r["Total_us"]),
                            "Avg_us": float(r["Avg_us"]),
                            "RunDir": run_dir,
                        })
                except Exception as e:
                    print(f"[Warning] skip {fp}: {e}")
    return pd.DataFrame(rows)


def aggregate_profiler(raw, variant):
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw["Variant"] = variant
    gcols = ["Variant", "Framework", "Method", "Threads", "NProbe", "Stage"]
    agg = raw.groupby(gcols, as_index=False).agg(
        Total_us_mean=("Total_us", "mean"),
        Total_us_std=("Total_us", "std"),
        Avg_us_mean=("Avg_us", "mean"),
        Avg_us_std=("Avg_us", "std"),
        N_runs=("Avg_us", "count"),
    )
    for c in ("Total_us_std", "Avg_us_std"):
        agg[c] = agg[c].fillna(0.0)
    agg["Total_us"] = agg["Total_us_mean"]
    agg["Avg_us"] = agg["Avg_us_mean"]
    return agg


def aggregate_dataset(name, ds_cfg, processed_root, cfg):
    root = resolve_path(ds_cfg.get("root", ""), cfg)
    ds_type = ds_cfg.get("type", "multi_run")
    variant = ds_cfg.get("variant_label", name)
    framework = ds_cfg.get("framework_label", variant)
    include = ds_cfg.get("include_runs") or []
    exclude = ds_cfg.get("exclude_runs") or []

    if ds_type == "flat":
        if not os.path.isdir(root):
            print(f"[Skip] {name}: missing {root}")
            return None
        run_dirs = [root]
    else:
        run_dirs = discover_run_dirs(root, include, exclude)
        if not run_dirs:
            print(f"[Skip] {name}: no runs under {root}")
            return None

    print(f"[{name}] ({ds_type}) aggregating {len(run_dirs)} source(s) variant={variant}")
    for rd in run_dirs:
        print(f"  - {os.path.basename(rd)}")

    trade_raw = load_tradeoff_runs(run_dirs, framework, variant)
    prof_raw = load_profiler_runs(run_dirs, framework.lower() if framework in ("OpenMP", "Pthread") else "baseline")
    trade_agg = aggregate_tradeoff(trade_raw)
    prof_agg = aggregate_profiler(prof_raw, variant)

    out_dir = os.path.join(processed_root, name)
    os.makedirs(out_dir, exist_ok=True)
    trade_agg.to_csv(os.path.join(out_dir, "ivfpq_tradeoff_agg.csv"), index=False)
    prof_agg.to_csv(os.path.join(out_dir, "profiler_agg.csv"), index=False)

    meta = {
        "dataset": name,
        "type": ds_type,
        "variant": variant,
        "framework": framework,
        "root": root,
        "n_sources": len(run_dirs),
        "sources": [os.path.basename(r) for r in run_dirs],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tradeoff_rows": int(len(trade_agg)),
        "profiler_rows": int(len(prof_agg)),
    }
    with open(os.path.join(out_dir, "aggregation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "runs_used.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta["sources"]) + "\n")
    print(f"[OK] {out_dir} tradeoff={len(trade_agg)} profiler={len(prof_agg)}")
    return meta


def build_all(cfg=None, processed_dir=None):
    cfg = cfg or load_config()
    proc_rel = cfg.get("aggregation", {}).get("processed_dir", "processed_data")
    processed_root = proc_rel if os.path.isabs(proc_rel) else os.path.join(data_root_path(cfg), proc_rel)
    datasets = cfg.get("datasets", {})
    metas = {}
    for name, ds in datasets.items():
        m = aggregate_dataset(name, ds, processed_root, cfg)
        if m:
            metas[name] = m
    summary_path = os.path.join(processed_root, "aggregation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] {processed_root}")
    return processed_root, metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config() if args.config == CONFIG_PATH else yaml.safe_load(open(args.config, encoding="utf-8"))
    out = args.out
    if out and not os.path.isabs(out):
        out = os.path.join(data_root_path(cfg), out)
    build_all(cfg=cfg, processed_dir=out)


if __name__ == "__main__":
    main()
