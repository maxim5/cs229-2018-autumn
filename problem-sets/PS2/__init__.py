import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath("__file__"))
SCRIPT_DIR_PARENT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR_PARENT))