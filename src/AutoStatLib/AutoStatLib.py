from __future__ import annotations

from typing import Optional, Union

import numpy as np

from AutoStatLib.statistical_tests import StatisticalTests
from AutoStatLib.normality_tests import NormalityTests
from AutoStatLib.helpers import Helpers
from AutoStatLib.text_formatting import TextFormatting
from AutoStatLib._version import __version__


class StatisticalAnalysis(StatisticalTests, NormalityTests, TextFormatting, Helpers):
    """
    Automatic statistical analysis tool

    Selects and runs the appropriate statistical test based on the
    properties of the input data (normality, number of groups, pairing).

    Parameters
    ----------
    groups_list:
        Input data as a list of groups, where each group is a list of
        numeric values. Non-numeric values are silently dropped.
    paired:
        Whether the groups are paired / repeated measures. Default False.
    tails:
        Number of tails for the hypothesis test (1 or 2). Default 2.
    popmean:
        Population mean for single-sample tests. Default None (uses 0
        with a warning if not provided).
    posthoc:
        Whether to run post-hoc pairwise comparisons after multi-group
        tests. Default False.
    verbose:
        Whether to print the full summary to stdout after each test.
        Default True.
    raise_errors:
        Whether to raise ValueError on invalid input instead of printing
        an error message. Default False.
    groups_name:
        Optional list of group labels. Cycled if shorter than the number
        of groups. Default None (auto-generates "Group 1", "Group 2", …).
    subgrouping:
        Optional subgrouping metadata passed through to the result dict.
        Default None (stored as [0]).
    """

    def __init__(
        self,
        groups_list: list[list],
        paired: bool = False,
        tails: int = 2,
        popmean: Optional[float] = None,
        posthoc: bool = False,
        verbose: bool = True,
        raise_errors: bool = False,
        groups_name: Optional[list[str]] = None,
        subgrouping: Optional[list] = None,
    ) -> None:

        self.groups_list: list[list] = groups_list
        self.paired: bool = paired
        self.tails: int = tails
        self.popmean: Optional[float] = popmean
        self.posthoc: bool = posthoc
        self.verbose: bool = verbose
        self.raise_errors: bool = raise_errors
        self.n_groups: int = len(self.groups_list)
        self.groups_name: list[str] = (
            [groups_name[i % len(groups_name)] for i in range(self.n_groups)]
            if groups_name and groups_name != [""]
            else [f"Group {i + 1}" for i in range(self.n_groups)]
        )
        self.subgrouping: list = subgrouping if subgrouping is not None else [0]
        self.warning_flag_non_numeric_data: bool = False
        self.summary: str = "AutoStatLib v{}".format(__version__)

        # State reset on every run
        self.results: Optional[dict] = None
        self.error: bool = False
        self.warnings: list[str] = []
        self.normals: list[bool] = []
        self.test_name: str = ""
        self.test_id: Optional[str] = None
        self.test_stat: Optional[np.float64] = None
        self.p_value: Optional[np.float64] = None
        self.posthoc_matrix_df: Optional[object] = None
        self.posthoc_matrix: list[list[float]] = []
        self.posthoc_name: str = ""
        self.data: list[list[float]] = []
        self.parametric: Optional[bool] = None

        # Test ID classification
        self.test_ids_all: list[str] = [
            "anova_1w_ordinary",
            "anova_1w_rm",
            "friedman",
            "kruskal_wallis",
            "mann_whitney",
            "t_test_independent",
            "t_test_paired",
            "t_test_single_sample",
            "wilcoxon",
            "wilcoxon_single_sample",
        ]
        self.test_ids_parametric: list[str] = [
            "anova_1w_ordinary",
            "anova_1w_rm",
            "t_test_independent",
            "t_test_paired",
            "t_test_single_sample",
        ]
        self.test_ids_dependent: list[str] = [
            "anova_1w_rm",
            "friedman",
            "t_test_paired",
            "wilcoxon",
        ]
        self.test_ids_3sample: list[str] = [
            "anova_1w_ordinary",
            "anova_1w_rm",
            "friedman",
            "kruskal_wallis",
        ]
        self.test_ids_2sample: list[str] = [
            "mann_whitney",
            "t_test_independent",
            "t_test_paired",
            "wilcoxon",
        ]
        self.test_ids_1sample: list[str] = [
            "t_test_single_sample",
            "wilcoxon_single_sample",
        ]
        self.warning_ids_all: dict[str, str] = {
            "param_test_with_non-normal_data": (
                "\nWarning: Parametric test was manualy chosen for Not-Normaly distributed data.\n"
                "         The results might be skewed. \n"
                "         Please, run non-parametric test or preform automatic test selection.\n"
            ),
            "non-param_test_with_normal_data": (
                "\nWarning: Non-Parametric test was manualy chosen for Normaly distributed data.\n"
                "         The results might be skewed. \n"
                "         Please, run parametric test or preform automatic test selection.\n"
            ),
            "no_pop_mean_set": (
                "\nWarning: No Population Mean was set up for single-sample test, used default 0 value.\n"
                "         The results might be skewed. \n"
                "         Please, set the Population Mean and run the test again.\n"
            ),
            "paired_test_with_independend_samples": (
                "\nWarning: A paired test was manually selected, even though the samples were declared independent.\n"
                "         The results might be skewed. \n"
                "         Please, run test for independend samples or preform automatic test selection.\n"
            ),
            "independend_test_with_paired_samples": (
                "\nWarning: An independent test was manually selected, even though the samples were declared paired.\n"
                "         The results might be skewed. \n"
                "         Please, run test for paired samples or preform automatic test selection.\n"
            ),
        }

    # ------------------------------------------------------------------ #
    # Internal orchestration                                             #
    # ------------------------------------------------------------------ #

    def run_test(self, test: str = "auto") -> None:
        """
        Core test runner. Validates input, checks normality, dispatches the
        chosen test, builds the result dict, and optionally prints a summary.

        Parameters
        ----------
        test:
            Test ID string (from ``test_ids_all``) or ``"auto"`` for
            automatic selection. Default ``"auto"``.
        """
        # Reset state from any previous run
        self.results = None
        self.error = False
        self.warnings = []
        self.normals = []
        self.test_name = ""
        self.test_id = None
        self.test_stat = None
        self.p_value = None
        self.parametric = None
        self.posthoc_matrix_df = None
        self.posthoc_matrix = []
        self.posthoc_name = ""

        self.log("\n" + "-" * 67)
        self.log("Statistical analysis initiated for data in {} groups\n".format(len(self.groups_list)))

        # Coerce input to float, drop non-numeric values
        self.data = self.floatify_recursive(self.groups_list)  # type: ignore[assignment]
        if self.warning_flag_non_numeric_data:
            self.log("Text or other non-numeric data in the input was ignored:")

        # Drop completely empty columns
        self.data = [col for col in self.data if any(x is not None for x in col)]
        self.n_groups = len(self.data)

        # Input validation
        try:
            assert self.data, "There is no input data"
            assert self.tails in [1, 2], "Tails parameter can be 1 or 2 only"
            assert (
                test in self.test_ids_all or test == "auto"
            ), "Wrong test id choosen, ensure you called correct function"
            assert all(
                len(group) >= 4 for group in self.data
            ), "Each group must contain at least four values"
            assert not (
                test in self.test_ids_dependent
                and not all(len(lst) == len(self.data[0]) for lst in self.data)
            ), "Paired samples must have the same length"
            assert not (
                test in self.test_ids_2sample and self.n_groups != 2
            ), f"Only two groups of data must be given for 2-groups tests, got {self.n_groups}"
            assert not (
                test in self.test_ids_1sample and self.n_groups > 1
            ), f"Only one group of data must be given for single-group tests, got {self.n_groups}"
            assert not (
                test in self.test_ids_3sample and self.n_groups < 3
            ), f"At least three groups of data must be given for multi-groups tests, got {self.n_groups}"
        except AssertionError as error:
            self.run_test_by_id("none")
            self.results = self.create_results_dict()

            if self.raise_errors:
                raise ValueError(error)

            if self.verbose:
                self.log("\nTest  :", test)
                self.log("Error :", error)
                self.log("-" * 67 + "\n")
                self.error = True
                print(self.summary)
            else:
                print("AutoStatLib Error :", error)

            return

        # Print data table
        self.print_groups()

        # Normality checks
        self.log("\n\nThe group is assumed to be normally distributed if at least one")
        self.log("normality test result is positive. Normality checked by tests:")
        self.log("Shapiro-Wilk, Lilliefors, Anderson-Darling, D'Agostino-Pearson")
        self.log("[+] -positive, [-] -negative, [ ] -too small group for the test\n")
        self.log("                   SW  LF  AD  AP  ")
        for i, group_data in enumerate(self.data):
            poll = self.check_normality(group_data)
            isnormal: bool = any(poll)
            poll_print = tuple(
                "+" if x is True else "-" if x is False else " " if x is None else "e"
                for x in poll
            )
            self.normals.append(isnormal)
            self.log(
                f"    {self.groups_name[i].ljust(11, ' ')[:11]}:    "
                f"{poll_print[0]}   {poll_print[1]}   {poll_print[2]}   {poll_print[3]}   "
                f"so disrtibution seems {'normal' if isnormal else 'not normal'}"
            )
        self.parametric = all(self.normals)

        # Log test selection context
        self.log("\n\nInput:\n")
        self.log("Data Normaly Distributed:     ", self.parametric)
        self.log("Paired Groups:                ", self.paired)
        self.log("Groups:                       ", self.n_groups)
        self.log("Test chosen by user:          ", test)

        # Warn when the manually chosen test is inappropriate
        if test != "auto" and not self.parametric and test in self.test_ids_parametric:
            self.AddWarning("param_test_with_non-normal_data")
        if test != "auto" and self.parametric and test not in self.test_ids_parametric:
            self.AddWarning("non-param_test_with_normal_data")
        if test != "auto" and not self.paired and test in self.test_ids_dependent:
            self.AddWarning("paired_test_with_independend_samples")
        if test != "auto" and self.paired and test not in self.test_ids_dependent:
            self.AddWarning("independend_test_with_paired_samples")

        # Dispatch
        if test in self.test_ids_all:
            self.run_test_by_id(test)
        else:
            self.run_test_auto()

        # Build and print results
        self.results = self.create_results_dict()
        self.print_results()
        self.log("\n\nResults above are accessible as a dictionary via GetResult() method")
        self.log("-" * 67 + "\n")

        if self.verbose is True:
            print(self.summary)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def RunAuto(self) -> None:
        """Run automatic test selection."""
        self.run_test(test="auto")

    def RunManual(self, test: str) -> None:
        """Run a specific test by ID. See ``GetTestIDs()`` for valid values."""
        self.run_test(test)

    def RunOnewayAnova(self) -> None:
        """Run Ordinary One-Way ANOVA."""
        self.run_test(test="anova_1w_ordinary")

    def RunOnewayAnovaRM(self) -> None:
        """Run Repeated Measures One-Way ANOVA."""
        self.run_test(test="anova_1w_rm")

    def RunFriedman(self) -> None:
        """Run Friedman test."""
        self.run_test(test="friedman")

    def RunKruskalWallis(self) -> None:
        """Run Kruskal-Wallis test."""
        self.run_test(test="kruskal_wallis")

    def RunMannWhitney(self) -> None:
        """Run Mann-Whitney U test."""
        self.run_test(test="mann_whitney")

    def RunTtest(self) -> None:
        """Run t-test for independent samples."""
        self.run_test(test="t_test_independent")

    def RunTtestPaired(self) -> None:
        """Run t-test for paired samples."""
        self.run_test(test="t_test_paired")

    def RunTtestSingleSample(self) -> None:
        """Run single-sample t-test against ``popmean``."""
        self.run_test(test="t_test_single_sample")

    def RunWilcoxonSingleSample(self) -> None:
        """Run Wilcoxon signed-rank test for a single sample."""
        self.run_test(test="wilcoxon_single_sample")

    def RunWilcoxon(self) -> None:
        """Run Wilcoxon signed-rank test for two paired samples."""
        self.run_test(test="wilcoxon")

    def GetResult(self) -> Optional[dict]:
        """
        Return the result dictionary from the last test run.

        Returns
        -------
        dict
            Full result dictionary if a test was run successfully.
        None
            If no test has been run yet.
        {}
            Empty dict if the test encountered an error.
        """
        if self.results is None and not self.error:
            print("No test chosen, no results to output")
            return self.results
        if not self.results and self.error:
            print("Error occured, no results to output")
            return {}
        else:
            return self.results

    def GetSummary(self) -> str:
        """
        Return the full text summary of the last test run.

        Returns the accumulated log string including normality test output,
        chosen test, and results table.
        """
        if self.results is None and not self.error:
            print("No test chosen, no summary to output")
            return self.summary
        else:
            return self.summary

    def GetTestIDs(self) -> list[str]:
        """Return the list of all valid test ID strings."""
        return self.test_ids_all

    def PrintSummary(self) -> None:
        """Print the full text summary to stdout."""
        print(self.summary)


if __name__ == "__main__":
    print("This package works as an imported module only.\nUse \"import AutoStatLib\" statement")