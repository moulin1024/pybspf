"""Archived bridge to the historical monolith; isolated legacy environment only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from legacy.bspf1d import *
from legacy.bspf1d import _Knot
