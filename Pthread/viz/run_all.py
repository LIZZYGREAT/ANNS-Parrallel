#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from aggregate_runs import build_all
from generate_report_figures import main as plot_main

if __name__ == "__main__":
    build_all()
    sys.argv = [sys.argv[0], "--no-aggregate"]
    plot_main()
