import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.make_manuscript_figures import fig_descriptor_effect

fig_descriptor_effect()
