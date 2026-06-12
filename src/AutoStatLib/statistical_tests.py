from __future__ import annotations

from AutoStatLib._protocol import StatAnalysisProtocol

from typing import Optional

import numpy as np
import scikit_posthocs as sp
from scipy.stats import (
    f_oneway,
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Known issue: One-tailed p-value calculation is currently implemented by simply
# dividing the two-tailed p-value by 2. This approach is only valid when the
# observed effect is in the hypothesized direction. If the effect is in the opposite
# direction, the one-tailed p-value should be calculated as 1 - (p_two_tailed / 2).
# Without an `alternative` parameter to specify the expected direction of the effect,
# users may receive misleading results for one-tailed tests when the effect is in the
# opposite direction.

# Bug note:
# One-tailed p-value: no directionality check
# if self.tails == 1:
#     p_value /= 2
# Dividing a two-tailed p-value by 2 is only valid when the test statistic falls in the hypothesized direction. If the effect is in the opposite direction, the one-tailed p should be 1 - p_two_tailed/2. Without a alternative parameter exposed to the user, results for one-tailed tests where the effect direction is "wrong" will be misleading.
# Recommendation: Either expose an alternative='less'/'greater' parameter and pass it to scipy.stats directly (which handles it correctly), or document that one-tailed results are only valid when the observed effect is in the expected direction.



class StatisticalTests(StatAnalysisProtocol):
    """Statistical tests mixin."""

    def run_test_auto(self) -> None:
        """Select and run the most appropriate test based on data properties."""
        if self.n_groups == 1:
            if self.parametric:
                self.run_test_by_id("t_test_single_sample")
            else:
                self.run_test_by_id("wilcoxon_single_sample")

        elif self.n_groups == 2:
            if self.paired:
                if self.parametric:
                    self.run_test_by_id("t_test_paired")
                else:
                    self.run_test_by_id("wilcoxon")
            else:
                if self.parametric:
                    self.run_test_by_id("t_test_independent")
                else:
                    self.run_test_by_id("mann_whitney")

        elif self.n_groups >= 3:
            if self.paired:
                if self.parametric:
                    self.run_test_by_id("anova_1w_rm")
                else:
                    self.run_test_by_id("friedman")
            else:
                if self.parametric:
                    self.run_test_by_id("anova_1w_ordinary")
                else:
                    self.run_test_by_id("kruskal_wallis")

    def run_test_by_id(self, test_id: str) -> None:
        """Dispatch a test by its string ID, then store the results on self."""
        test_names_dict: dict[str, str] = {
            "anova_1w_ordinary":      "Ordinary One-Way ANOVA",
            "anova_1w_rm":            "Repeated Measures One-Way ANOVA",
            "friedman":               "Friedman test",
            "kruskal_wallis":         "Kruskal-Wallis test",
            "mann_whitney":           "Mann-Whitney U test",
            "t_test_independent":     "t-test for independent samples",
            "t_test_paired":          "t-test for paired samples",
            "t_test_single_sample":   "Single-sample t-test",
            "wilcoxon":               "Wilcoxon signed-rank test",
            "wilcoxon_single_sample": "Wilcoxon signed-rank test for single sample",
            "none":                   "No statictical tests preformed",
        }

        stat: Optional[np.float64]
        p_value: Optional[np.float64]

        match test_id:
            case "anova_1w_ordinary":      stat, p_value = self.anova_1w_ordinary()
            case "anova_1w_rm":            stat, p_value = self.anova_1w_rm()
            case "friedman":               stat, p_value = self.friedman()
            case "kruskal_wallis":         stat, p_value = self.kruskal_wallis()
            case "mann_whitney":           stat, p_value = self.mann_whitney()
            case "t_test_independent":     stat, p_value = self.t_test_independent()
            case "t_test_paired":          stat, p_value = self.t_test_paired()
            case "t_test_single_sample":   stat, p_value = self.t_test_single_sample()
            case "wilcoxon":               stat, p_value = self.wilcoxon()
            case "wilcoxon_single_sample": stat, p_value = self.wilcoxon_single_sample()
            case "none":                   stat, p_value = (None, None)

        self.paired_test_applied: bool = test_id in self.test_ids_dependent
        self.test_name: str = test_names_dict[test_id]
        self.test_id: str = test_id
        self.test_stat: Optional[np.float64] = stat
        self.p_value: Optional[np.float64] = p_value

    # ------------------------------------------------------------------ #
    # Individual test methods                                            #
    # Each returns (test_statistic, p_value).                            #
    # ------------------------------------------------------------------ #

    def anova_1w_ordinary(self) -> tuple[np.float64, np.float64]:
        """One-way ANOVA with optional Tukey's post-hoc test."""
        stat, p_value = f_oneway(*self.data)        

        # bad practice to silently rewrite users input, 
        # but this is a non-directional test so one-tailed doesn't make sense        
        self.tails = 2

        # if self.tails == 1 and p_value > 0.5:
        #     p_value /= 2
        # if self.tails == 1:
        #     p_value /= 2

        if self.posthoc:  # and p_value < 0.05:
            data_flat = np.concatenate(self.data)
            self.posthoc_name = "Tukey`s posthoc"
            group_labels = np.concatenate(
                [[f"Group_{i+1}"] * len(group) for i, group in enumerate(self.data)]
            )
            tukey_result = pairwise_tukeyhsd(data_flat, group_labels)
            tukey_pvalues: list[float] = tukey_result.pvalues.tolist()
            self.posthoc_matrix = self.list_to_matrix(tukey_pvalues, self.n_groups)

        return stat, p_value

    def anova_1w_rm(self) -> tuple[np.float64, np.float64]:
        """Repeated-measures one-way ANOVA."""
        # Parameters:
        # data: list of lists, where each sublist represents repeated measures for a subject

        
        df = self.matrix_to_dataframe(self.data)
        res = AnovaRM(df, "Value", "Row", within=["Col"]).fit()

        stat: np.float64 = res.anova_table.iloc[0][0]
        p_value: np.float64 = res.anova_table.iloc[0][3]

        # # --- Posthocs: paired t-tests ---
        # wide = df.pivot(index='Row', columns='Col', values='Value')
        # conds = wide.columns
        # pairs = list(itertools.combinations(conds, 2))

        # pvals, stats = [], []
        # for a, b in pairs:
        #     t, p = ttest_rel(wide[a], wide[b])
        #     stats.append(t)
        #     pvals.append(p)

        # # Adjust p-values
        # rej, p_corr, _, _ = multipletests(pvals, method='bonferroni')

        # print(p_corr)



        self.tails = 2
        return stat, p_value

    def friedman(self) -> tuple[np.float64, np.float64]:
        """Friedman test (non-parametric repeated-measures)."""
        stat, p_value = friedmanchisquare(*self.data)
        self.tails = 2
        return stat, p_value

    def kruskal_wallis(self) -> tuple[np.float64, np.float64]:
        """Kruskal-Wallis test with optional Dunn's post-hoc test."""
        stat, p_value = kruskal(*self.data)

        # Perform Dunn's multiple comparisons if Kruskal-Wallis is significant
        if self.posthoc:  # and p_value < 0.05:
            self.posthoc_matrix = sp.posthoc_dunn(
                self.data, p_adjust="bonferroni"
            ).values.tolist()
            self.posthoc_name = "Dunn`s posthoc"

        self.tails = 2
        return stat, p_value

    def mann_whitney(self) -> tuple[np.float64, np.float64]:
        """Mann-Whitney U test for two independent samples."""
        stat, p_value = mannwhitneyu(self.data[0], self.data[1], alternative="two-sided")
        if self.tails == 1:
            p_value /= 2
        # alternative method of one-tailed calculation
        # gives the same result:
        # stat, p_value = mannwhitneyu(
        #     self.data[0], self.data[1], alternative='two-sided' if self.tails == 2 else 'less')
        # if self.tails == 1 and p_value > 0.5:
        #     p_value = 1-p_value
        return stat, p_value

    def t_test_independent(self) -> tuple[np.float64, np.float64]:
        """Independent-samples t-test."""
        stat, p_value = ttest_ind(self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def t_test_paired(self) -> tuple[np.float64, np.float64]:
        """Paired-samples t-test."""
        stat, p_value = ttest_rel(self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def t_test_single_sample(self) -> tuple[np.float64, np.float64]:
        """Single-sample t-test against a population mean."""
        if self.popmean is None:
            self.popmean = 0
            self.AddWarning("no_pop_mean_set")
        stat, p_value = ttest_1samp(self.data[0], self.popmean)
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def wilcoxon(self) -> tuple[np.float64, np.float64]:
        """Wilcoxon signed-rank test for two paired samples."""
        stat, p_value = wilcoxon(self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def wilcoxon_single_sample(self) -> tuple[np.float64, np.float64]:
        """Wilcoxon signed-rank test for a single sample against a population mean."""
        if self.popmean is None:
            self.popmean = 0
            self.AddWarning("no_pop_mean_set")
        arr = np.asarray(self.data[0], dtype=float) - self.popmean
        stat, p_value = wilcoxon(arr)
        if self.tails == 1:
            p_value /= 2
        return stat, p_value