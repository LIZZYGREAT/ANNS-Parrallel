#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from generate_scheduling_figures import main

if __name__ == "__main__":
    main()
