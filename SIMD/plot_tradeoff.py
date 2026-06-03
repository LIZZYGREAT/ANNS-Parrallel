import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_latency_recall_tradeoff(csv_path):
    if not os.path.exists(csv_path):
        print(f"[Error] 找不到数据文件: {csv_path}")
        print("请检查路径，确保已经从服务器拉取了 files/ 文件夹。")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] 读取 CSV 失败: {e}")
        return

    df.columns = df.columns.str.strip()

    required_cols = {'Threads', 'Top_C', 'Recall@10', 'Latency(us)'}
    if not required_cols.issubset(df.columns):
        print(f"[Error] CSV 列名不匹配。找到的列: {df.columns.tolist()}")
        return

    plt.figure(figsize=(10, 7), dpi=150)
    plt.rcParams['font.family'] = 'sans-serif'
    
    threads = sorted(df['Threads'].unique())
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, thread_count in enumerate(threads):
        subset = df[df['Threads'] == thread_count].sort_values(by='Recall@10')
        
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        plt.plot(subset['Recall@10'], subset['Latency(us)'], 
                 marker=marker, 
                 color=color, 
                 linewidth=2, 
                 markersize=8,
                 label=f'Threads = {thread_count}')

        for _, row in subset.iterrows():
            plt.annotate(f"C={int(row['Top_C'])}", 
                         (row['Recall@10'], row['Latency(us)']),
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=9,
                         color='dimgray')

    plt.xlabel('Recall@10', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (us)', fontsize=12, fontweight='bold')
    plt.title('ANN Search: Latency vs. Recall Trade-off', fontsize=14, fontweight='bold', pad=15)
    
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
    os.makedirs('../figures', exist_ok=True)
    output_filename = '../figures/latency_recall_tradeoff_curve.png'
    plt.savefig(output_filename, format='png')
    print(f"\n[Success] 图表已成功绘制并保存为当前目录下的 {output_filename}")
    

if __name__ == "__main__":
    data_file = "files/pq_data/latency_recall_tradeoff.csv"
    plot_latency_recall_tradeoff(data_file)