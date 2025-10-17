import numpy as np
import itertools
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, wilcoxon, mannwhitneyu, f_oneway, kruskal, friedmanchisquare


class StatisticalTests():
    '''
        Statistical tests mixin
    '''

    def run_test_auto(self):

        if self.n_groups == 1:
            if self.parametric:
                self.run_test_by_id('t_test_single_sample')
            else:
                self.run_test_by_id('wilcoxon_single_sample')

        elif self.n_groups == 2:
            if self.paired:
                if self.parametric:
                    self.run_test_by_id('t_test_paired')
                else:
                    self.run_test_by_id('wilcoxon')
            else:
                if self.parametric:
                    self.run_test_by_id('t_test_independent')
                else:
                    self.run_test_by_id('mann_whitney')

        elif self.n_groups >= 3:
            if self.paired:
                if self.parametric:
                    self.run_test_by_id('anova_1w_rm')
                else:
                    self.run_test_by_id('friedman')
            else:
                if self.parametric:
                    self.run_test_by_id('anova_1w_ordinary')
                else:
                    self.run_test_by_id('kruskal_wallis')

        else:
            pass

    def run_test_by_id(self, test_id):

        test_names_dict = {
            'anova_1w_ordinary': 'Ordinary One-Way ANOVA',
            'anova_1w_rm': 'Repeated Measures One-Way ANOVA',
            'friedman': 'Friedman test',
            'kruskal_wallis': 'Kruskal-Wallis test',
            'mann_whitney': 'Mann-Whitney U test',
            't_test_independent': 't-test for independent samples',
            't_test_paired': 't-test for paired samples',
            't_test_single_sample': 'Single-sample t-test',
            'wilcoxon': 'Wilcoxon signed-rank test',
            'wilcoxon_single_sample': 'Wilcoxon signed-rank test for single sample',
        }

        match test_id:
            case 'anova_1w_ordinary': stat, p_value = self.anova_1w_ordinary()
            case 'anova_1w_rm': stat, p_value = self.anova_1w_rm()
            case 'friedman': stat, p_value = self.friedman()
            case 'kruskal_wallis': stat, p_value = self.kruskal_wallis()
            case 'mann_whitney': stat, p_value = self.mann_whitney()
            case 't_test_independent': stat, p_value = self.t_test_independent()
            case 't_test_paired': stat, p_value = self.t_test_paired()
            case 't_test_single_sample': stat, p_value = self.t_test_single_sample()
            case 'wilcoxon': stat, p_value = self.wilcoxon()
            case 'wilcoxon_single_sample': stat, p_value = self.wilcoxon_single_sample()

        if test_id in self.test_ids_dependent:
            self.paired = True
        else:
            self.paired = False

        self.test_name = test_names_dict[test_id]
        self.test_id = test_id
        self.test_stat = stat
        self.p_value = p_value

    def anova_1w_ordinary(self):
        stat, p_value = f_oneway(*self.data)
        self.tails = 2
        # if self.tails == 1 and p_value > 0.5:
        #     p_value /= 2
        # if self.tails == 1:
        #     p_value /= 2

        if self.posthoc:  # and p_value < 0.05:
            data_flat = np.concatenate(self.data)
            self.posthoc_name = 'Tukey`s posthoc'
            group_labels = np.concatenate(
                [[f"Group_{i+1}"] * len(group) for i, group in enumerate(self.data)])
            # Tukey's multiple comparisons
            tukey_result = pairwise_tukeyhsd(data_flat, group_labels)

            list = tukey_result.pvalues.tolist()
            n = self.n_groups
            # prepare posthoc matrix
            self.posthoc_matrix = self.list_to_matrix(list, n)

        return stat, p_value

    def anova_1w_rm(self):
        """
        Perform repeated measures one-way ANOVA test.

        Parameters:
        data: list of lists, where each sublist represents repeated measures for a subject
        """

        df = self.matrix_to_dataframe(self.data)
        res = AnovaRM(df, 'Value', 'Row', within=['Col']).fit()
        print(res)
        stat = res.anova_table.iloc[0][0]
        p_value = res.anova_table.iloc[0][3]

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

    def friedman(self):
        stat, p_value = friedmanchisquare(*self.data)
        self.tails = 2
        return stat, p_value

    def kruskal_wallis(self):
        stat, p_value = kruskal(*self.data)

        # Perform Dunn's multiple comparisons if Kruskal-Wallis is significant
        if self.posthoc:  # and p_value < 0.05:
            self.posthoc_matrix = sp.posthoc_dunn(
                self.data, p_adjust='bonferroni').values.tolist()
            self.posthoc_name = 'Dunn`s posthoc'
        self.tails = 2
        return stat, p_value

    def mann_whitney(self):
        stat, p_value = mannwhitneyu(
            self.data[0], self.data[1], alternative='two-sided')
        if self.tails == 1:
            p_value /= 2
        # alternative method of one-tailed calculation
        # gives the same result:
        # stat, p_value = mannwhitneyu(
        #     self.data[0], self.data[1], alternative='two-sided' if self.tails == 2 else 'less')
        # if self.tails == 1 and p_value > 0.5:
        #     p_value = 1-p_value
        return stat, p_value

    def t_test_independent(self):
        stat, p_value = ttest_ind(
            self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def t_test_paired(self):
        stat, p_value = ttest_rel(
            self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def t_test_single_sample(self):
        if self.popmean == None:
            self.popmean = 0
            self.AddWarning('no_pop_mean_set')
        stat, p_value = ttest_1samp(self.data[0], self.popmean)
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def wilcoxon(self):
        stat, p_value = wilcoxon(self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        return stat, p_value

    def wilcoxon_single_sample(self):
        if self.popmean == None:
            self.popmean = 0
            self.AddWarning('no_pop_mean_set')
        data = [i - self.popmean for i in self.data[0]]
        stat, p_value = wilcoxon(data)
        if self.tails == 1:
            p_value /= 2
        return stat, p_value
