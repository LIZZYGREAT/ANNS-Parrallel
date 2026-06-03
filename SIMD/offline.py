import numpy as np
import os


def load_fbin(filename):
    print(f"Loading {filename}...")
    with open(filename, "rb") as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32).reshape(n, d)
    print(f"Loaded data shape: {data.shape}")
    return data

def save_fbin(filename, data):
    print(f"Saving to {filename}...")
    with open(filename, "wb") as f:
        np.array([data.shape[0]], dtype=np.int32).tofile(f)
        np.array([data.shape[1]], dtype=np.int32).tofile(f)
        data.astype(np.float32).tofile(f)
    print("Save completed.")

def save_matrix_to_header(R, filename="opq_matrix.h"):
    print(f"Exporting R matrix to {filename}...")
    with open(filename, "w") as f:
        f.write("#pragma once\n\n")
        f.write(f"const float OPQ_R[{R.shape[0]}][{R.shape[1]}] = {{\n")
        for row in R:
            # 格式化输出为 C++ 数组格式
            f.write("    {" + ", ".join([f"{val:.6f}f" for val in row]) + "},\n")
        f.write("};\n")

        
def train_opq_rotation():
    data_path = "./anndata/DEEP100K.base.100k.fbin"
    rotated_base_path = "./anndata/DEEP100K.base.100k.rotated.fbin"
    r_matrix_path = "./anndata/OPQ_R_matrix.fbin"

    M = 16  # 子空间数量
    D = 96  # 原始维度
    D_SUB = D // M

    # 1. 加载 Base 数据
    X = load_fbin(data_path)

    # 2. 计算协方差矩阵与特征分解
    print("Computing covariance matrix and PCA...")
    # rowvar=False 表示每一列代表一个特征
    cov_mat = np.cov(X, rowvar=False)
    
    # 求解特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    # np.linalg.eigh 返回的特征值是从小到大的，我们需要降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eps = 1e-9
    eigenvalues = np.maximum(eigenvalues, eps)

    # 3. 贪心算法：Eigenvalue Allocation (修正版)
    print("Performing Eigenvalue Allocation...")
    
    # 核心修正：取对数，并平移为全正数权重，彻底消除连乘 < 1 导致的逻辑崩溃
    log_evs = np.log(eigenvalues)
    log_evs = log_evs - np.min(log_evs)  # 整体平移，确保所有权重 >= 0
    
    bucket_sums = np.zeros(M)
    bucket_indices = [[] for _ in range(M)]

    for i, weight in enumerate(log_evs):
        best_bucket = -1
        min_sum = float('inf')
        
        # 寻找当前“权重和最小”且没装满的桶
        for m in range(M):
            if len(bucket_indices[m]) < D_SUB:
                if bucket_sums[m] < min_sum:
                    min_sum = bucket_sums[m]
                    best_bucket = m
        
        # 装入桶中并累加权重
        bucket_indices[best_bucket].append(i)
        bucket_sums[best_bucket] += weight

    # 验证平衡结果
    print("Bucket Log-Weight sums after allocation (should be balanced):")
    for m in range(M):
        print(f"Subspace {m}: Weight Sum = {bucket_sums[m]:.4f}, Indices = {bucket_indices[m]}")

    # 4. 组装正交旋转矩阵 R
    print("Constructing orthogonal matrix R...")
    reordered_indices = []
    for m in range(M):
        reordered_indices.extend(bucket_indices[m])

    # 按照重排后的索引提取特征向量，构成 R 矩阵
    R = eigenvectors[:, reordered_indices]

    # 5. 对 Base 数据进行旋转
    print("Applying rotation R to base data...")
    # 数据矩阵 X (N x D) 乘以 旋转矩阵 R (D x D) = 旋转后的 X' (N x D)
    X_rotated = X @ R

    # 6. 保存旋转后的 Base 和 矩阵 R
    save_fbin(rotated_base_path, X_rotated)
    # 将 R (96x96) 当作有 96 个 96维向量的数据集存下来，方便 C++ 复用 LoadData
    save_fbin(r_matrix_path, R)
    print("Offline OPQ training completely finished.")

    save_matrix_to_header(R)


if __name__ == "__main__":
    train_opq_rotation()