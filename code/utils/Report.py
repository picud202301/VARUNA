# ===============================================================================
# Report Base Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Lightweight reporting helper that configures a consistent seaborn/matplotlib
#   theme and exposes small utilities for downstream report classes. It stores
#   input/output paths, provides a stub `generate()` entry point for concrete
#   implementations, and offers a convenience axis helper to render "no data"
#   placeholders in composite figures.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Report:
    """
    Base reporting utility that standardizes plot styling and IO paths.

    Parameters
    ----------
    database_file : str
        Absolute or relative path to the SQLite database containing experiment data.
    graphics_path : str
        Directory path where generated figures and artifacts will be written.

    Attributes
    ----------
    database_file : str
        Path to the source database.
    graphics_path : str
        Output directory for figures and JSON artifacts.
    """

    def __init__(self, database_file: str, graphics_path: str) -> None:
        """
        Initialize the report context and configure global plotting style.

        Parameters
        ----------
        database_file : str
            Path to the SQLite database file.
        graphics_path : str
            Output directory for report graphics.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during initialization or style configuration.
        """
        try:
            self.database_file: str = database_file
            self.graphics_path: str = graphics_path

            sns.set_context("talk", font_scale=0.9)
            sns.set_style("whitegrid")
            plt.rcParams.update({
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
            })
        except Exception as e:
            print(f"[ERROR] Report.__init__ failed: {e}")
            raise

    def generate(self) -> None:
        """
        Entry point to execute the full report generation pipeline.

        Notes
        -----
        Subclasses should override this method to load data, create charts,
        and write outputs under `graphics_path`.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error from subclass implementations.
        """
        try:
            pass
        except Exception as e:
            print(f"[ERROR] Report.generate failed: {e}")
            raise

    @staticmethod
    def axisNoData(ax: Axes, title: str) -> None:
        """
        Render a standardized 'No data' placeholder on the provided axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axis to annotate.
        title : str
            Title text to display on the axis.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during axis manipulation.
        """
        try:
            ax.set_title(title, fontsize=14)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        except Exception as e:
            print(f"[ERROR] Report.axisNoData failed: {e}")
            raise
