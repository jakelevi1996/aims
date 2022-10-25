import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

sys.path.append(ROOT_DIR)

RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")
