"""Central location for the two directories the notebook reads/writes."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Excel option/return data used throughout the notebook.
DATA_DIR = PROJECT_DIR / "Data"

# Where lecture figures get saved, so they land directly in the LaTeX slides.
LATEX_DIR = Path(
    "/Users/anderstrolle/Library/CloudStorage/Dropbox-CBS/Anders Bjerre Trolle/"
    "Latex/CBS Financial Engineering"
)
