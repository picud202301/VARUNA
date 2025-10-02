# ===============================================================================
# ReportZermelo Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Strongly typed reporting utility for Zermelo navigation experiments. This class
#   reads simulation results from a SQLite database and generates compact JSON
#   summaries as well as publication-ready visualizations (distributions, hybrid
#   success/win-share charts). It also performs data coherence checks and applies
#   deterministic winner selection with priority rules (favoring A* in near-ties
#   against Analytic). All outputs are saved to a graphics/exports directory.
# ===============================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import os, sqlite3, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from typing import Any, Callable
from utils.Report import Report
from marine.CurrentField import CurrentField
import gzip
from io import TextIOWrapper
import textwrap


class ReportZermelo(Report):
    """
    Reporting class for Zermelo experiments. It loads simulation data, checks basic
    coherence between solvers, computes winner statistics, and produces compact JSON
    artifacts and figures for analysis and dashboards.

    Attributes
    ----------
    solver_names : dict[int, str]
        Mapping from solver_id to human-readable solver name.
    size_names : dict[int, str]
        Mapping from size_id to human-readable scenario size name.
    solver_pallete : dict[str, str]
        Color palette by solver name (hex triplets), used in plots.
    solver_order : list[str]
        Canonical order of solvers for consistent plotting and reporting.
    exclude_analytic : bool
        Whether to exclude the Analytic solver from analysis/plots/exports.
    current_types : Any
        Raw current type descriptor as returned by `CurrentField.getCurrentTypes()`.
    current_field_names : list[str]
        Normalized list of current-field display names.
    _cf_name : Callable[[int], str]
        Resolver from current_field_id to display name, robust to 0/1-based ids.
    """

    def __init__(self, database_file: str, graphics_path: str, exclude_analytic: bool = False) -> None:
        """
        Initialize the reporting utility.

        Parameters
        ----------
        database_file : str
            Path to the SQLite database file containing experiment results.
        graphics_path : str
            Output directory for figures and JSON artifacts.
        exclude_analytic : bool, optional
            If True, exclude the Analytic solver from summaries and plots.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during initialization.
        """
        try:
            super().__init__(database_file, graphics_path)
            self.solver_names: dict[int, str] = {0: 'Analytic', 1: 'A*', 2: 'PSO', 3: 'Ipopt'}
            self.size_names: dict[int, str] = {1: '200x200 m', 2: '2000x2000 m', 3: '20000x20000 m'}
            self.solver_pallete: dict[str, str] = {
                'Analytic': '#0173b2', 'A*': "#8f082e", 'Ipopt': '#029e73', 'PSO': "#c0d500"
            }
            self.solver_order: list[str] = ['Analytic', 'A*', 'PSO', 'Ipopt']
            self.exclude_analytic: bool = exclude_analytic

            # --- Current-type normalization (list or dict; robust 0/1-based ids) ---
            self.current_types: Any = CurrentField.getCurrentTypes()
            if isinstance(self.current_types, dict):
                self.current_field_names: list[str] = list(self.current_types.values())

                def _name(cf_id: int) -> str:
                    return self.current_types.get(cf_id, str(cf_id))
                self._cf_name: Callable[[int], str] = _name
            else:
                self.current_field_names = list(self.current_types)

                def _name(cf_id: int) -> str:
                    try:
                        if 0 <= cf_id < len(self.current_types):
                            return self.current_types[cf_id]
                        if 1 <= (cf_id - 1) < len(self.current_types):
                            return self.current_types[cf_id - 1]
                    except Exception:
                        pass
                    return str(cf_id)

                self._cf_name = _name
            # -----------------------------------------------------------------------
        except Exception as e:
            print(f"[ERROR] ReportZermelo.__init__ failed: {e}")
            raise

    def generate(self) -> None:
        """
        Generate all standard outputs: hybrid summary charts, per-size distributions,
        and compact JSON report (including gzipped variant).

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during database access or generation steps.
        """
        conn: sqlite3.Connection = sqlite3.connect(self.database_file)
        try:
            df_all: pd.DataFrame = pd.read_sql_query(
                "SELECT s.solver_id, s.goal_objective, s.navegation_index, s.scenario_id FROM SIMULATIONS s",
                conn
            )
            df_all["solver"] = df_all["solver_id"].map(self.solver_names)
            self.checkDataCoherence(df_all[df_all["goal_objective"] == 1])

            self.generateHybridSummaryCharts(conn)
            for size_id, size_name in self.size_names.items():
                self.generateImprovedDistributionPlots(int(size_id), str(size_name), conn)
            self.generateJsonReportCompact(conn)
        except Exception as e:
            print(f"[ERROR] ReportZermelo.generate failed: {e}")
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def selectWinnersWithPriority(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select per-scenario winners among non-Analytic solvers by minimum Navigation Index.
        If Analytic and A* are within a small absolute difference (< 0.1), force A* as winner
        for those scenarios (tie-breaking priority).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame filtered to successful runs (goal_objective == 1) with columns:
            ['solver', 'scenario_id', 'navegation_index'].

        Returns
        -------
        pd.DataFrame
            Winners with columns matching df plus a boolean 'forced_astar' flag.
            Returns an empty DataFrame if there are no non-Analytic entries.

        Raises
        ------
        Exception
            Propagates any error during grouping/merging operations.
        """
        try:
            df_no_analytic: pd.DataFrame = df[df["solver"] != "Analytic"].copy()
            if df_no_analytic.empty:
                return pd.DataFrame()

            best_per_solver = df_no_analytic.loc[df_no_analytic.groupby(
                ["scenario_id", "solver"], observed=False
            )["navegation_index"].idxmin()].copy()

            winners = best_per_solver.loc[best_per_solver.groupby(
                "scenario_id", observed=False
            )["navegation_index"].idxmin()].copy()
            winners["forced_astar"] = False

            analytic = df[df["solver"] == "Analytic"][["scenario_id", "navegation_index"]].rename(
                columns={"navegation_index": "analytic_index"})
            astar = best_per_solver[best_per_solver["solver"] == "A*"][["scenario_id", "navegation_index"]].rename(
                columns={"navegation_index": "astar_index"})
            comparison = pd.merge(analytic, astar, on="scenario_id", how="inner")
            comparison["diff"] = (comparison["analytic_index"] - comparison["astar_index"]).abs()

            tie_ids = comparison.loc[comparison["diff"] < 0.1, "scenario_id"].unique()
            if len(tie_ids) > 0:
                astar_winners = astar[astar["scenario_id"].isin(tie_ids)].merge(
                    df_no_analytic,
                    left_on=["scenario_id", "astar_index"],
                    right_on=["scenario_id", "navegation_index"],
                    how="left"
                )
                astar_winners = astar_winners.drop_duplicates(subset=["scenario_id", "solver"]).copy()
                astar_winners["forced_astar"] = True

                winners = winners[~winners["scenario_id"].isin(tie_ids)]
                winners = pd.concat(
                    [winners, astar_winners[df_no_analytic.columns.tolist() + ["forced_astar"]]],
                    ignore_index=True
                )

            return winners
        except Exception as e:
            print(f"[ERROR] ReportZermelo.selectWinnersWithPriority failed: {e}")
            raise

    def checkDataCoherence(self, df: pd.DataFrame) -> None:
        """
        Perform basic coherence checks between Analytic and A*:
        - Report scenarios that have Analytic but miss A*.
        - Warn if Analytic outperforms A* (NI lower) and export details to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of successful runs with columns at least
            ['solver', 'scenario_id', 'navegation_index'].

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during checks or file export.
        """
        try:
            analytic = df[df["solver"] == "Analytic"][["scenario_id", "navegation_index"]]
            astar = df[df["solver"] == "A*"][["scenario_id", "navegation_index"]]

            missing_astar = set(analytic["scenario_id"]) - set(astar["scenario_id"])
            if missing_astar:
                print("[ERROR] Scenarios with Analytic but without A*:", sorted(missing_astar))
                # To abort, uncomment the next line:
                # raise ValueError("Data inconsistency: missing A* runs.")

            comparison = pd.merge(analytic, astar, on="scenario_id", suffixes=("_analytic", "_astar"))
            incoherent = comparison[comparison["navegation_index_analytic"] < comparison["navegation_index_astar"]]
            if not incoherent.empty:
                print(f"[WARN] Analytic outperforms A* in {len(incoherent)} scenarios. See CSV for details.")
                try:
                    incoherent.to_csv(os.path.join(self.graphics_path, "incoherent_analytic_better_than_astar.csv"), index=False)
                except Exception:
                    pass
                # To abort, uncomment:
                # raise ValueError("Analytic outperforms A* in some scenarios.")
        except Exception as e:
            print(f"[ERROR] ReportZermelo.checkDataCoherence failed: {e}")
            raise

    def applyPenaltyForFailures(self, df: pd.DataFrame, metrics_to_penalize: list[str]) -> pd.DataFrame:
        """
        Apply a deterministic penalty to failed runs (goal_objective != 1) for the
        specified metrics by setting them to 1.5x the maximum observed among successes.
        If there are no successes, set a large sentinel (999999).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least ['goal_objective'] and the metrics to penalize.
        metrics_to_penalize : list[str]
            List of metric column names to adjust.

        Returns
        -------
        pd.DataFrame
            The same DataFrame reference with additional 'adjusted_<metric>' columns.

        Raises
        ------
        Exception
            Propagates any error during transformation.
        """
        try:
            df_ok = df[df["goal_objective"] == 1]
            if df_ok.empty:
                for metric in metrics_to_penalize:
                    df[f"adjusted_{metric}"] = np.where(df["goal_objective"] == 1, df[metric], 999999)
                return df

            for metric in metrics_to_penalize:
                penalty_value = df_ok[metric].max() * 1.5
                df[f"adjusted_{metric}"] = np.where(
                    df["goal_objective"] == 1,
                    df[metric],
                    penalty_value
                )
            return df
        except Exception as e:
            print(f"[ERROR] ReportZermelo.applyPenaltyForFailures failed: {e}")
            raise

    def generateImprovedDistributionPlots(self, size_id: int, size_name: str, conn: sqlite3.Connection) -> None:
        """
        Produce per-size violin+box distributions for key metrics (success-only),
        with means overlaid and consistent styling, saved as a PDF.

        Parameters
        ----------
        size_id : int
            Scenario size identifier to filter SCENARIOS.
        size_name : str
            Human-readable size name (used in titles/filenames).
        conn : sqlite3.Connection
            Active database connection.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during query, plotting, or file I/O.
        """
        try:
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            # === Desired font sizes (explicit control per element) ===
            fs_base = 12   # general base
            fs_title = 14  # subplot title
            fs_label = 12  # axis labels
            fs_tick = 12   # tick labels
            mean_s = 30    # marker size for means

            metrics: dict[str, str] = {
                "total_time": "Total Time [s]",
                "total_distance": "Total Distance [m]",
                "navegation_index": "Navigation Index [-]",
                "execution_time": "Execution Time [s]"
            }
            pdf_file = f"{self.graphics_path}distribution_{size_id}.pdf"

            query = """
            SELECT s.solver_id, s.goal_objective, s.total_time, s.total_distance,
                   s.execution_time, s.navegation_index
            FROM SIMULATIONS s JOIN SCENARIOS sc ON s.scenario_id = sc.id
            WHERE sc.size_id = ?
            """
            df = pd.read_sql_query(query, conn, params=(size_id,))
            if self.exclude_analytic:
                df = df[df["solver_id"] != 0]

            # Success-only
            df = df[df["goal_objective"] == 1]
            if df.empty:
                print(f"[INFO] No successful data for {size_name}")
                return

            df["solver"] = pd.Categorical(
                df["solver_id"].map(self.solver_names),
                categories=self.solver_order,
                ordered=True
            )

            fig, axes = plt.subplots(len(metrics), 1, figsize=(7.5, 2.4 * len(metrics)))
            fig.suptitle(f'Performance Metrics Distribution (Success Only) - {size_name}', fontsize=fs_title + 1)

            for ax, (col, label) in zip(axes, metrics.items()):
                sns.violinplot(
                    data=df, x="solver", y=col, hue="solver",
                    palette=self.solver_pallete, order=self.solver_order,
                    inner="box", ax=ax, dodge=False
                )

                # Mean per solver (aligned with order)
                means = (df.groupby("solver", observed=False)[col].mean()
                         .reindex(self.solver_order))
                ax.scatter(
                    x=np.arange(len(self.solver_order)),
                    y=means.values,
                    color="black",
                    marker='x',
                    s=mean_s,
                    zorder=10,
                    label="Mean"
                )

                ax.set_title(label, fontsize=fs_title, pad=6)
                ax.set_ylabel(label, fontsize=fs_label)
                ax.set_xlabel("")

                ax.tick_params(axis='x', labelsize=fs_tick, pad=2)
                ax.tick_params(axis='y', labelsize=fs_tick, pad=2)

                for txt in ax.get_xticklabels() + ax.get_yticklabels():
                    txt.set_fontsize(fs_tick)

                leg = ax.get_legend()
                if leg:
                    leg.remove()

                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

            fig.tight_layout(pad=0.8)
            plt.savefig(pdf_file, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            print(f"[INFO] Distributions saved: {pdf_file}")
        except Exception as e:
            print(f"[ERROR] ReportZermelo.generateImprovedDistributionPlots failed: {e}")
            raise

    def generateJsonReportCompact(self, conn: sqlite3.Connection, decimals: int = 2, write_gzip: bool = True) -> None:
        """
        Build a highly compact JSON for per-scenario tables (success-only). It stores:
          - Success means per scenario/solver (tt, td, ni, et) + n_success
          - Winner per scenario (solver_idx + forced_astar flag)
          - Global win_share per solver (%)

        Parameters
        ----------
        conn : sqlite3.Connection
            Active database connection.
        decimals : int, optional
            Number of decimals for means rounding.
        write_gzip : bool, optional
            If True, also write a gzipped copy of the JSON.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during computation or file I/O.
        """
        try:
            query = """
            SELECT s.solver_id, s.goal_objective, s.total_time, s.total_distance,
                   s.execution_time, s.navegation_index,
                   sc.size_id, sc.id as scenario_id
            FROM SIMULATIONS s 
            JOIN SCENARIOS sc ON s.scenario_id = sc.id
            """
            df = pd.read_sql_query(query, conn)
            if self.exclude_analytic:
                df = df[df["solver_id"] != 0]

            df["solver"] = df["solver_id"].map(self.solver_names)
            df["size_name"] = df["size_id"].map(self.size_names)
            df_ok = df[df["goal_objective"] == 1].copy()

            solvers: list[str] = self.solver_order[:]
            solver_idx: dict[str, int] = {s: i for i, s in enumerate(solvers)}

            metrics = ["total_time", "total_distance", "execution_time", "navegation_index"]

            means_ok = (
                df_ok.groupby(["scenario_id", "size_id", "solver"], observed=False)[metrics]
                    .mean()
                    .round(decimals)
                    .reset_index()
            )
            n_ok = df_ok.groupby(["scenario_id", "solver"], observed=False).size().rename("n_ok").reset_index()

            merged = pd.merge(means_ok, n_ok, on=["scenario_id", "solver"], how="left")

            winners_df = self.selectWinnersWithPriority(df_ok)
            winners_by_scenario: dict[int, tuple[int, bool]] = {}
            if winners_df is not None and not winners_df.empty:
                tmp = winners_df.drop_duplicates(subset=["scenario_id"]).set_index("scenario_id")
                for scen_id, row in tmp[["solver", "forced_astar"]].to_dict(orient="index").items():
                    winners_by_scenario[int(scen_id)] = (
                        int(solver_idx.get(str(row["solver"]), -1)),
                        bool(row.get("forced_astar", False))
                    )

            n_w = len(winners_by_scenario)
            win_counts = [0] * len(solvers)
            if n_w > 0:
                for (_sid, (w_idx, _flag)) in winners_by_scenario.items():
                    if 0 <= w_idx < len(solvers):
                        win_counts[w_idx] += 1
            win_share = [round((c / n_w * 100.0), 2) if n_w > 0 else 0.0 for c in win_counts]

            scen_map: dict[int, dict[str, Any]] = {}
            for _, r in merged.iterrows():
                scen_id = int(r["scenario_id"])
                size_id = int(r["size_id"])
                s_name = str(r["solver"])
                idx = solver_idx.get(s_name, None)
                if idx is None:
                    continue
                if scen_id not in scen_map:
                    scen_map[scen_id] = {
                        "sz": size_id,
                        "S": [[0, None, None, None, None] for _ in solvers]
                    }
                scen_map[scen_id]["S"][idx] = [
                    int(r.get("n_ok", 0)),
                    float(r["total_time"]) if pd.notna(r["total_time"]) else None,
                    float(r["total_distance"]) if pd.notna(r["total_distance"]) else None,
                    float(r["navegation_index"]) if pd.notna(r["navegation_index"]) else None,
                    float(r["execution_time"]) if pd.notna(r["execution_time"]) else None,
                ]

            scenarios_list: list[dict[str, Any]] = []
            winners_list: list[list[Any]] = []
            for scen_id in sorted(scen_map.keys()):
                entry = {"id": int(scen_id), "sz": int(scen_map[scen_id]["sz"]), "S": scen_map[scen_id]["S"]}
                scenarios_list.append(entry)
                if scen_id in winners_by_scenario:
                    w_idx, fflag = winners_by_scenario[scen_id]
                    winners_list.append([int(scen_id), int(w_idx), bool(fflag)])

            report = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "ns": int(df["scenario_id"].nunique()),
                    "nt": int(len(df)),
                    "solvers": solvers,
                    "sizes": {int(k): str(v) for k, v in self.size_names.items()},
                },
                "win_share": win_share,
                "winners": winners_list,
                "scenarios": scenarios_list
            }

            out_json = os.path.join(self.graphics_path, "report_experiment_compact.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(report, f, separators=(",", ":"), ensure_ascii=False)
            print(f"[INFO] JSON compact saved: {out_json} ({os.path.getsize(out_json)/1024:.2f} KB)")

            if write_gzip:
                out_gz = out_json + ".gz"
                with gzip.open(out_gz, "wt", encoding="utf-8") as gz:
                    json.dump(report, gz, separators=(",", ":"), ensure_ascii=False)
                print(f"[INFO] Gzipped compact JSON: {out_gz} ({os.path.getsize(out_gz)/1024:.2f} KB)")
        except Exception as e:
            print(f"[ERROR] ReportZermelo.generateJsonReportCompact failed: {e}")
            raise

    def generateJsonReport(self, conn: sqlite3.Connection) -> None:
        """
        Build a compact (but richer) JSON for dashboards:
          - scenarios: per-scenario per-solver success-only means + counts
          - winners: winner per scenario (A*-priority near-tie vs Analytic) and global summary
          - solvers: global num_simulations, num_success, success_rate, and micro-averaged metrics
          - win_share: percentage of scenarios won by each solver

        Parameters
        ----------
        conn : sqlite3.Connection
            Active database connection.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during computation or file I/O.
        """
        try:
            query = """
            SELECT s.solver_id, s.goal_objective, s.total_time, s.total_distance,
                   s.execution_time, s.navegation_index,
                   sc.size_id, sc.current_field_id, sc.id as scenario_id
            FROM SIMULATIONS s 
            JOIN SCENARIOS sc ON s.scenario_id = sc.id
            """
            df = pd.read_sql_query(query, conn)
            if self.exclude_analytic:
                df = df[df["solver_id"] != 0]

            df["solver"] = df["solver_id"].map(self.solver_names)
            df["size_name"] = df["size_id"].map(self.size_names)

            df_ok = df[df["goal_objective"] == 1].copy()
            metrics = ["total_time", "total_distance", "execution_time", "navegation_index"]

            counts_all_global = df.groupby("solver", observed=False).size().rename("n_all")
            counts_ok_global = df_ok.groupby("solver", observed=False).size().rename("n_ok")

            counts_all = df.groupby(["scenario_id", "solver"], observed=False).size().rename("n_all")
            counts_ok = df_ok.groupby(["scenario_id", "solver"], observed=False).size().rename("n_ok")

            means_ok = (
                df_ok.groupby(["scenario_id", "size_id", "solver"], observed=False)[metrics]
                     .mean()
                     .round(4)
            )

            base = means_ok.reset_index().set_index(["scenario_id", "solver"])
            base = base.join(counts_all, how="left").join(counts_ok, how="left")
            base["n_all"] = base["n_all"].fillna(0).astype(int)
            base["n_ok"] = base["n_ok"].fillna(0).astype(int)
            base["success_rate"] = np.where(
                base["n_all"] > 0, (base["n_ok"] / base["n_all"] * 100.0).round(2), 0.0
            )

            size_map = df.set_index("scenario_id")["size_name"].to_dict()
            size_id_map = df.set_index("scenario_id")["size_id"].to_dict()

            winners_global_df = self.selectWinnersWithPriority(df_ok)
            winners_count: dict[str, int] = {}
            winners_by_scenario: dict[int, dict[str, Any]] = {}
            if winners_global_df is not None and not winners_global_df.empty:
                winners_count = winners_global_df.groupby("solver").size().astype(int).to_dict()
                tmp = winners_global_df.drop_duplicates(subset=["scenario_id"]).set_index("scenario_id")
                for scen_id, row in tmp[["solver", "forced_astar"]].to_dict(orient="index").items():
                    winners_by_scenario[int(scen_id)] = {
                        "solver": str(row["solver"]),
                        "forced_astar": bool(row.get("forced_astar", False))
                    }

            n_scen_winner = len({int(k) for k in winners_by_scenario.keys()})
            win_share_pct: dict[str, float] = {}
            if n_scen_winner > 0:
                for s in self.solver_order:
                    win_share_pct[s] = round(float(winners_count.get(s, 0)) / n_scen_winner * 100.0, 2)

            solvers_block: dict[str, Any] = {}

            def wmean(g: pd.DataFrame, col: str) -> float:
                w = g["n_ok"].values
                v = g[col].values
                if len(v) == 0 or np.sum(w) == 0:
                    return 0.0
                return float(np.average(v, weights=w))

            micro_src = (base.reset_index()
                             .rename(columns={"total_time": "time",
                                              "total_distance": "dist",
                                              "navegation_index": "ni",
                                              "execution_time": "exec"}))

            for solver in [s for s in self.solver_order if s in counts_all_global.index]:
                n_all = int(counts_all_global.get(solver, 0))
                n_ok = int(counts_ok_global.get(solver, 0))
                success_rate = round((n_ok / n_all) * 100.0, 2) if n_all > 0 else 0.0

                g = micro_src[micro_src["solver"] == solver]
                solvers_block[solver] = {
                    "num_simulations": n_all,
                    "num_success": n_ok,
                    "success_rate": float(success_rate),
                    "metrics": {
                        "success_only_micro_mean": {
                            "total_time": wmean(g, "time"),
                            "total_distance": wmean(g, "dist"),
                            "navegation_index": wmean(g, "ni"),
                            "execution_time": wmean(g, "exec"),
                        }
                    }
                }

            report: dict[str, Any] = {
                "generated_at": datetime.now().isoformat(),
                "total_scenarios": int(df["scenario_id"].nunique()),
                "total_simulations": int(len(df)),
                "success_rate_global": round(float(df["goal_objective"].mean() * 100), 2) if len(df) else 0.0,
                "solvers": solvers_block,
                "win_share": win_share_pct,
                "winners": {str(k): int(v) for k, v in winners_count.items()},
                "scenarios": {}
            }

            base = base.reset_index()
            solver_pos = {s: i for i, s in enumerate(self.solver_order)}
            base["__pos__"] = base["solver"].map(lambda s: solver_pos.get(s, 999))

            for scen_id, g in base.sort_values(["scenario_id", "__pos__"]).groupby("scenario_id", sort=False):
                scen_id_int = int(scen_id)
                size_id = int(size_id_map.get(scen_id_int, df.loc[df["scenario_id"] == scen_id_int, "size_id"].iloc[0]))
                size_name = str(size_map.get(scen_id_int, self.size_names.get(size_id, str(size_id))))

                solvers_per_scen: dict[str, Any] = {}
                for _, row in g.iterrows():
                    solver = str(row["solver"])
                    solvers_per_scen[solver] = {
                        "num_simulations": int(row["n_all"]),
                        "num_success": int(row["n_ok"]),
                        "success_rate": float(row["success_rate"]),
                        "metrics": {
                            "success_only": {
                                "total_time": float(row["total_time"]),
                                "total_distance": float(row["total_distance"]),
                                "navegation_index": float(row["navegation_index"]),
                                "execution_time": float(row["execution_time"])
                            }
                        }
                    }

                report["scenarios"][str(scen_id_int)] = {
                    "scenario_id": scen_id_int,
                    "size_id": size_id,
                    "size_name": size_name,
                    "winner": winners_by_scenario.get(scen_id_int, None),
                    "solvers": solvers_per_scen
                }

            output_file = os.path.join(self.graphics_path, "report_experiment.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"[INFO] JSON report saved: {output_file} ({os.path.getsize(output_file)/1024:.2f} KB)")
        except Exception as e:
            print(f"[ERROR] ReportZermelo.generateJsonReport failed: {e}")
            raise

    def generateHybridSummaryCharts(self, conn: sqlite3.Connection) -> None:
        """
        Create a two-panel chart per scenario size:
        (Left) Success Rate heatmap by solver x current-field type.
        (Right) Stacked bar chart of Win Share (%) by current-field type.
        Saves a multi-row PDF assembling all sizes.

        Parameters
        ----------
        conn : sqlite3.Connection
            Active database connection.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Propagates any error during query, plotting, or file I/O.
        """
        try:
            import textwrap
            import matplotlib as mpl
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd

            # --- Forced font sizes (slightly larger than compact plots) ---
            suptitle_fs = 22
            title_fs = 20
            label_fs = 15
            tick_fs = 13
            curr_tick_fs = 12   # x tick labels for current types (diagonal)
            legend_fs = 14
            annot_fs = 14
            cbar_fs = 14

            def _wrap_xticklabels(ax, max_width: int = 14) -> None:
                """Wrap long tick labels to multiple lines to reduce figure height."""
                labels = [l.get_text() for l in ax.get_xticklabels()]
                wrapped = [textwrap.fill(t, max_width) for t in labels]
                ax.set_xticklabels(wrapped)

            pdf_file = f"{self.graphics_path}hybrid_summary_charts.pdf"
            n_rows = len(self.size_names)

            with mpl.rc_context({
                "font.size": tick_fs,
                "axes.titlesize": title_fs,
                "axes.labelsize": label_fs,
                "xtick.labelsize": tick_fs,
                "ytick.labelsize": tick_fs,
                "legend.fontsize": legend_fs,
                "figure.titlesize": suptitle_fs
            }):
                fig, axes = plt.subplots(n_rows, 2, figsize=(20, 7 * n_rows), squeeze=False)
                fig.suptitle('Solver Performance: Success Rate and Win Share', y=0.99)

                for i, (size_id, size_name) in enumerate(self.size_names.items()):
                    ax_success, ax_wins = axes[i, 0], axes[i, 1]
                    is_last = (i == n_rows - 1)

                    query = """
                    SELECT s.solver_id, sc.current_field_id, s.goal_objective, s.navegation_index, s.scenario_id
                    FROM SIMULATIONS s JOIN SCENARIOS sc ON s.scenario_id = sc.id WHERE sc.size_id = ?
                    """
                    df = pd.read_sql_query(query, conn, params=(size_id,))
                    if self.exclude_analytic:
                        df = df[df["solver_id"] != 0]

                    if df.empty:
                        self.axisNoData(ax_success, f"Success Rate (%) - {size_name}")
                        self.axisNoData(ax_wins,    f"Win Share (%) - {size_name}")
                        ax_success.title.set_fontsize(title_fs)
                        ax_wins.title.set_fontsize(title_fs)
                        ax_success.tick_params(axis='both', labelsize=tick_fs)
                        ax_wins.tick_params(axis='both', labelsize=tick_fs)
                        continue

                    df["solver"] = df["solver_id"].map(self.solver_names)
                    df["current_field"] = df["current_field_id"].map(self._cf_name)

                    # --- Heatmap: Success Rate by solver x current-field type ---
                    success_summary = (
                        df.groupby(["solver", "current_field"], observed=False)["goal_objective"]
                        .mean().unstack()
                        .reindex(index=self.solver_order, columns=self.current_field_names)
                        .fillna(0) * 100
                    )

                    hm = sns.heatmap(
                        success_summary, ax=ax_success, annot=True, fmt=".1f", cmap="YlGnBu",
                        vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                        cbar_kws={"shrink": 0.85}, annot_kws={"size": annot_fs}
                    )
                    ax_success.set_title(f"Success Rate (%) - {size_name}", pad=12)
                    ax_success.set_xlabel("")
                    ax_success.set_ylabel("")
                    ax_success.tick_params(axis='y', labelsize=tick_fs)

                    if is_last:
                        ax_success.tick_params(axis='x', labelsize=curr_tick_fs, pad=6)
                        _wrap_xticklabels(ax_success, max_width=14)
                        for lbl in ax_success.get_xticklabels():
                            lbl.set_rotation(45)
                            lbl.set_ha('right')
                    else:
                        ax_success.set_xticklabels([])
                        ax_success.tick_params(axis='x', length=0)

                    try:
                        cbar = hm.collections[0].colorbar
                        cbar.ax.tick_params(labelsize=cbar_fs)
                    except Exception:
                        pass

                    # --- Stacked bars: Win Share by current-field type ---
                    winners = self.selectWinnersWithPriority(df[df["goal_objective"] == 1])
                    if winners.empty:
                        self.axisNoData(ax_wins, f"Win Share (%) - {size_name}")
                        ax_wins.title.set_fontsize(title_fs)
                        ax_wins.tick_params(axis='both', labelsize=tick_fs)
                    else:
                        wins_count = pd.crosstab(
                            winners['current_field'], winners['solver']
                        ).reindex(index=self.current_field_names, columns=self.solver_order, fill_value=0)
                        wins_pct = wins_count.div(wins_count.sum(axis=1), axis=0).fillna(0) * 100

                        wins_pct.plot(
                            kind='bar', stacked=True, ax=ax_wins,
                            color=[self.solver_pallete.get(s, '#333333') for s in wins_pct.columns],
                            width=0.7
                        )
                        ax_wins.set_title(f"Win Share (%) - {size_name}", pad=12)
                        ax_wins.set_ylabel("Share [%]", labelpad=6)
                        ax_wins.tick_params(axis='y', labelsize=tick_fs)

                        if is_last:
                            ax_wins.set_xlabel("Current field type", labelpad=6)
                            ax_wins.tick_params(axis='x', labelsize=curr_tick_fs, pad=6)
                            _wrap_xticklabels(ax_wins, max_width=14)
                            for lbl in ax_wins.get_xticklabels():
                                lbl.set_rotation(45)
                                lbl.set_ha('right')
                        else:
                            ax_wins.set_xlabel("")
                            ax_wins.set_xticklabels([])
                            ax_wins.tick_params(axis='x', length=0)

                        ax_wins.legend(title="Solver", fontsize=legend_fs, title_fontsize=legend_fs)

                    for ax in (ax_success, ax_wins):
                        ax.grid(True, linestyle='--', linewidth=0.5, axis='y', alpha=0.5)

                fig.tight_layout(pad=1.0)
                plt.subplots_adjust(top=0.93, bottom=0.12, hspace=0.55, wspace=0.25)

                plt.savefig(pdf_file, bbox_inches='tight')
                plt.show()
                plt.close(fig)
                print(f"[INFO] Hybrid summary charts saved: {pdf_file}")
        except Exception as e:
            print(f"[ERROR] ReportZermelo.generateHybridSummaryCharts failed: {e}")
            raise
