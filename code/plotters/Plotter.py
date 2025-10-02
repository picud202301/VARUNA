# ===============================================================================
# Plotter Base Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   This module provides a base class for creating and managing matplotlib
#   plots. It encapsulates figure and axis creation, offers a standard color
#   palette, and includes helper methods for showing and saving figures.
#   It is designed to be inherited by more specific plotting classes.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from utils.Geometry import Geometry


class Plotter(Geometry):
    """A base class for managing and creating matplotlib plots."""

    def __init__(self, parameters: Dict[str, Any]) -> None:
        """Initialize the Plotter instance.

        Parameters
        ----------
        parameters : Dict[str, Any]
            A dictionary of configuration parameters for the plotter.

        Raises
        ------
        Exception
            Propagates any error that occurs during initialization.
        """
        try:
            Geometry.__init__(self)
            self.parameters: Dict[str, Any] = parameters
            self.figures: Dict[str, plt.Figure] = {}
            self.axis: Dict[str, plt.Axes] = {}
            self._last_figure_id: Optional[str] = None
            self.colors: List[str] = [
                'red', 'green', 'blue', 'orange', 'purple',
                'cyan', 'magenta', 'yellow', 'lime', 'pink',
                'brown', 'gold', 'olive', 'navy', 'teal',
                'coral', 'orchid', 'darkgreen', 'slateblue', 'crimson'
            ]
            return
        except Exception as e:
            print(f"[ERROR] Plotter.__init__ failed: {e}")
            raise

    def createFigure(self, figure_id: str, figure_size: Tuple[int, int]) -> None:
        """Create and store a new matplotlib figure.

        Parameters
        ----------
        figure_id : str
            A unique identifier for the figure.
        figure_size : Tuple[int, int]
            The size of the figure as a (width, height) tuple in inches.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during figure creation.
        """
        try:
            self.figures[figure_id] = plt.figure(figsize=figure_size)
            self._last_figure_id = figure_id
        except Exception as e:
            print(f"[ERROR] Plotter.createFigure failed: {e}")
            raise

    def saveToPdf(self, filename: str) -> None:
        """Save the last created figure to a PDF file.

        Parameters
        ----------
        filename : str
            The output path for the PDF file, e.g., 'plots/scene.pdf'.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during the file saving process.
        """
        try:
            print("Saving figure at:", filename)
            fig: Optional[plt.Figure] = self.figures.get(self._last_figure_id, None)
            if fig is None:
                plt.savefig(filename, dpi=600, bbox_inches="tight")
            else:
                fig.savefig(filename, dpi=600, bbox_inches="tight")
        except Exception as e:
            print(f"[ERROR] Plotter.saveToPdf failed: {e}")
            raise

    def show(self) -> None:
        """Display all created matplotlib figures.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error that occurs while displaying the plots.
        """
        try:
            plt.show()
        except Exception as e:
            print(f"[ERROR] Plotter.show failed: {e}")
            raise