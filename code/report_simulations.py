# ===============================================================================
# Zermelo Reporting Runner
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Minimal entry-point script to generate consolidated reports and figures for
#   Zermelo experiments stored in an SQLite database. It prepares import paths,
#   configures input/output locations, and delegates the full reporting workflow
#   to `ReportZermelo.generate()`.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import os
import sys
from typing import NoReturn

from problems.zermelo.ReportZermelo import ReportZermelo

# =======================================================================
# PREPARE CODE TO LOAD MODULES
# =======================================================================
code_path: str = os.path.dirname(os.path.abspath(__file__))
framework_path: str = os.path.dirname(code_path)
figures_path: str = os.path.join(framework_path, "figures")
data_path: str = os.path.join(framework_path, "data")
if code_path not in sys.path:
    sys.path.insert(0, code_path)

# =======================================================================
# CONFIGURE EXECUTION
# =======================================================================
database_file: str = os.path.join(data_path, "zermelo", "zermelo.db")
graphics_path: str = os.path.join(figures_path, "zermelo")

# =======================================================================
# MAIN SCRIPT
# =======================================================================
def runReport() -> NoReturn:
    """
    Execute the Zermelo reporting pipeline.

    Notes
    -----
    This function instantiates :class:`ReportZermelo` with the configured database
    and graphics paths and triggers the full generation routine. All figures and
    JSON summaries are written under `graphics_path`.

    Raises
    ------
    Exception
        Any exceptions raised by the report generator are logged to stdout and
        re-raised to preserve calling context.
    """
    try:
        report_zermelo = ReportZermelo(database_file=database_file, graphics_path=graphics_path)
        report_zermelo.generate()
    except Exception as e:
        print(f"[ERROR] Report runner failed: {e}")
        raise


if __name__ == "__main__":
    runReport()
