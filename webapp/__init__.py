"""Local Stuff+ leaderboard and grader website. Run: python serve.py"""
import sys
from pathlib import Path

# the model code (features.py, physics.py, ...) lives at the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
