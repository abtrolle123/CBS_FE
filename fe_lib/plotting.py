"""Small plotting helper -- the repeated tight_layout/savefig/show boilerplate."""

import matplotlib.pyplot as plt

from fe_lib.paths import LATEX_DIR


def save_and_show(relative_path=None, tight_layout=True):
    """tight_layout + (optional) save into the LaTeX slides folder + show.

    relative_path: e.g. "Lecture2/Hes1.png", relative to LATEX_DIR. If None,
    the figure is just shown (not saved).
    """
    if tight_layout:
        plt.tight_layout()
    if relative_path is not None:
        plt.savefig(LATEX_DIR / relative_path)
    plt.show()
