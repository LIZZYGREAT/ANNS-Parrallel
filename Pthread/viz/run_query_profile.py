#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from data_loader import ensure_fig_dir, QUERY_THREADS, QUERY_NPROBE
from plot_query_profile import plot_query_profile


def main():
    ensure_fig_dir()
    print(f"[Query Profile] T={QUERY_THREADS}, nprobe={QUERY_NPROBE}")
    plot_query_profile(QUERY_THREADS, QUERY_NPROBE)
    print(f"[Done] figures/query_profile_T{QUERY_THREADS}_P{QUERY_NPROBE}.png")


if __name__ == "__main__":
    main()
