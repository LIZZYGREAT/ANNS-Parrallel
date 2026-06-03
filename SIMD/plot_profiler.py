import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_profiler_data(data_dir="files"):
    """
    扫描目录下的所有 profiler_detail_T*_C*.csv 文件，解析并合并为一个完整的 DataFrame
    """
    file_pattern = os.path.join(data_dir, "profiler_detail_T*_C*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"[Error] 未在 {data_dir}/ 目录下找到任何 profiler_detail_T*_C*.csv 文件。")
        return pd.DataFrame()

    all_data = []
    
    # 正则表达式用于从文件名提取 T 和 C 的值
    pattern = re.compile(r"profiler_detail_T(\d+)_C(\d+)\.csv")
    
    for file in files:
        match = pattern.search(os.path.basename(file))
        if match:
            threads = int(match.group(1))
            top_c = int(match.group(2))
            
            # 读取当前 CSV（无表头，指定列名）
            try:
                df = pd.read_csv(file, header=None, names=["Stage", "Latency(us)"])
                df["Threads"] = threads
                df["Top_C"] = top_c
                all_data.append(df)
            except Exception as e:
                print(f"[Warning] 读取文件 {file} 失败: {e}")
                
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

def plot_stacked_bars(df):
    """
    绘制微观性能剖析堆叠柱状图
    """
    # 强制固定阶段的顺序，确保堆叠顺序符合时间流
    stage_order = ["1_Rotate_Query", "2_Build_LUT", "3_ADC_Scan", "4_Exact_Rerank"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 对应各阶段的颜色
    
    # 建立画板，包含左右两张子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    plt.rcParams['font.family'] = 'sans-serif'
    
    target_thread = 4
    df_t1 = df[df["Threads"] == target_thread].copy()
    
    if not df_t1.empty:
        # 数据透视表：行是 Top_C，列是 Stage
        pivot_t1 = df_t1.pivot(index="Top_C", columns="Stage", values="Latency(us)")
        # 补全缺失的阶段并按顺序排列
        for col in stage_order:
            if col not in pivot_t1.columns:
                pivot_t1[col] = 0
        pivot_t1 = pivot_t1[stage_order]
        
        pivot_t1.plot(kind="bar", stacked=True, ax=ax1, color=colors, edgecolor='black', linewidth=0.5)
        
        ax1.set_title(f"Time Breakdown vs Top_C (Fixed Threads={target_thread})", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Top_C (Candidate Pool Size)", fontsize=11)
        ax1.set_ylabel("Total Latency (us)", fontsize=11)
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(title="Execution Stage", fontsize=9)
    else:
        ax1.text(0.5, 0.5, f"No data for Threads={target_thread}", ha='center', va='center')

    # ==========================================
    # 图 2：固定 Top_C = 100，观察随 Threads 变化的并行加速比
    # ==========================================
    target_c = 100
    df_c100 = df[df["Top_C"] == target_c].copy()
    
    if not df_c100.empty:
        # 数据透视表：行是 Threads，列是 Stage
        pivot_c100 = df_c100.pivot(index="Threads", columns="Stage", values="Latency(us)")
        for col in stage_order:
            if col not in pivot_c100.columns:
                pivot_c100[col] = 0
        pivot_c100 = pivot_c100[stage_order]
        
        pivot_c100.plot(kind="bar", stacked=True, ax=ax2, color=colors, edgecolor='black', linewidth=0.5)
        
        ax2.set_title(f"Time Breakdown vs Threads (Fixed Top_C={target_c})", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Number of Threads (OpenMP)", fontsize=11)
        ax2.set_ylabel("Total Latency (us)", fontsize=11)
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.legend(title="Execution Stage", fontsize=9)
    else:
        ax2.text(0.5, 0.5, f"No data for Top_C={target_c}", ha='center', va='center')

    # 调整布局并保存
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    output_filename = "../figures/profiler_breakdown_analysis.png"
    plt.savefig(output_filename, format="png")
    print(f"\n[Success] 性能剖析图表已成功绘制并保存为 {output_filename}")
    # plt.show()

if __name__ == "__main__":
    print("[System] 正在解析微观 Profiler 数据...")
    df_all = load_profiler_data(data_dir="files/pq_data")
    
    if not df_all.empty:
        plot_stacked_bars(df_all)
    else:
        print("[Error] 没有足够的数据进行绘图。")