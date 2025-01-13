import numpy as np
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp, wilcoxon, mannwhitneyu, f_oneway, kruskal, friedmanchisquare, shapiro, norm


class __StatisticalTests():
    '''
        Statistical tests mixin
    '''

    def anova(self):
        stat, p_value = f_oneway(*self.data)
        if self.tails == 1 and p_value > 0.5:
            p_value /= 2
        # if self.tails == 1:
        #     p_value /= 2
        self.test_name = 'ANOVA'
        self.test_id = 'anova'
        self.paired = False
        self.test_stat = stat
        self.p_value = p_value

    def friedman_test(self):
        stat, p_value = friedmanchisquare(*self.data)
        self.test_name = 'Friedman test'
        self.test_id = 'friedman'
        self.paired = True
        self.test_stat = stat
        self.p_value = p_value

    def kruskal_wallis_test(self):
        stat, p_value = kruskal(*self.data)
        self.test_name = 'Kruskal-Wallis test'
        self.test_id = 'kruskal_wallis'
        self.paired = False
        self.test_stat = stat
        self.p_value = p_value

    def mann_whitney_u_test(self):
        stat, p_value = mannwhitneyu(
            self.data[0], self.data[1], alternative='two-sided' if self.tails == 2 else 'greater')
        # if self.tails == 1:
        #     p_value /= 2
        self.test_name = 'Mann-Whitney U test'
        self.test_id = 'mann_whitney'
        self.paired = False
        self.test_stat = stat
        self.p_value = p_value

    def t_test_independend(self):
        t_stat, t_p_value = ttest_ind(
            self.data[0], self.data[1])

        if self.tails == 1:
            t_p_value /= 2

        self.test_name = 't-test for independend samples'
        self.test_id = 't_test_independend'
        self.paired = False
        self.test_stat = t_stat
        self.p_value = t_p_value

    def t_test_paired(self):
        t_stat, t_p_value = ttest_rel(
            self.data[0], self.data[1])

        if self.tails == 1:
            t_p_value /= 2

        self.test_name = 't-test for paired samples'
        self.test_id = 't_test_paired'
        self.paired = True
        self.test_stat = t_stat
        self.p_value = t_p_value

    def t_test_single_sample(self):
        if self.popmean == None:
            self.popmean = 0
            self.AddWarning('no_pop_mean_set')
        t_stat, t_p_value = ttest_1samp(self.data[0], self.popmean)
        if self.tails == 1:
            t_p_value /= 2

        self.test_name = 'Single-sample t-test'
        self.test_id = 't_test_single_sample'
        self.paired = False
        self.test_stat = t_stat
        self.p_value = t_p_value

    def wilcoxon_single_sample(self):
        if self.popmean == None:
            self.popmean = 0
            self.AddWarning('no_pop_mean_set')
        data = [i - self.popmean for i in self.data[0]]
        w_stat, w_p_value = wilcoxon(data)
        if self.tails == 1:
            p_value /= 2

        self.test_name = 'Wilcoxon signed-rank test for single sample'
        self.test_id = 'wilcoxon_single_sample'
        self.paired = False
        self.test_stat = w_stat
        self.p_value = w_p_value

    def wilcoxon(self):
        stat, p_value = wilcoxon(self.data[0], self.data[1])
        if self.tails == 1:
            p_value /= 2
        self.test_name = 'Wilcoxon signed-rank test'
        self.test_id = 'wilcoxon'
        self.paired = True
        self.test_stat = stat
        self.p_value = p_value


class __NormalityTests():
    '''
        Normality tests mixin
    '''

    def check_normality(self, data):
        # Shapiro-Wilk Test
        sw_stat, sw_p_value = shapiro(data)
        if sw_p_value > 0.05:
            return True, 'Shapiro-Wilk'

        # Lilliefors Test (Kolmogorov-Smirnov test)
        lilliefors_stat, lilliefors_p_value = self.lilliefors_test(data)
        if lilliefors_p_value > 0.05:
            return True, 'Lilliefors'

        return False, 'Not normal'

    def lilliefors_test(self, data):
        ''' Lilliefors Test (Kolmogorov-Smirnov test) '''
        data = np.sort(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        cdf_values = norm.cdf(data, loc=mean, scale=std)
        empirical_cdf = np.arange(1, n + 1) / n
        D_plus = np.max(empirical_cdf - cdf_values)
        D_minus = np.max(cdf_values - (np.arange(n) / n))
        D = max(D_plus, D_minus)

        # Use critical values from the Lilliefors table
        # (two-sided test, approximations for large n)
        critical_values = {
            100: 0.072,
            1000: 0.024,
            5000: 0.018,
            10000: 0.015,
            np.inf: 0.01
        }

        for size, critical_value in critical_values.items():
            if n <= size:
                return D, 1 - critical_value if D < critical_value else 0

        return D, 0


class __TextFormatting():
    '''
        Text formatting mixin
    '''

    def autospace(self, elements_list, space, delimiter=' ') -> str:
        output = ''
        for i, element in enumerate(elements_list):
            if i == len(elements_list):
                output += element
            else:
                output += element + (space-len(element))*delimiter
        return output

    def print_groups(self, space=24, max_length=15):
        self.log('')
        # Get the number of groups (rows) and the maximum length of rows
        data = self.data
        num_groups = len(data)
        group_longest = max(len(row) for row in data)

        # Print the header
        header = [f'Group {i+1}' for i in range(num_groups)]
        line = [''*7]
        self.log(self.autospace(header, space))
        self.log(self.autospace(line, space))

        # Print each column with a placeholder if longer than max_length
        for i in range(group_longest):
            row_values = []
            all_values_empty = True
            for row in data:
                if len(row) > max_length:
                    if i < max_length:
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    elif i == max_length:
                        row_values.append(f'[{len(row) - max_length} more]')
                        all_values_empty = False
                    else:
                        continue
                else:
                    if i < len(row):
                        row_values.append(str(row[i]))
                        all_values_empty = False
                    else:
                        row_values.append('')
            if all_values_empty:
                break
            self.log(self.autospace(row_values, space))

    def make_stars(self) -> int:
        p = self.p_value.item()
        if p is not None:
            if p < 0.0001:
                return 4
            if p < 0.001:
                return 3
            elif p < 0.01:
                return 2
            elif p < 0.05:
                return 1
            else:
                return 0
        return 0

    def make_p_value_printed(self) -> str:
        p = self.p_value.item()
        if p is not None:
            if p > 0.99:
                return 'p>0.99'
            elif p >= 0.01:
                return f'p={p:.2g}'
            elif p >= 0.001:
                return f'p={p:.2g}'
            elif p >= 0.0001:
                return f'p={p:.1g}'
            elif p < 0.0001:
                return 'p<0.0001'
            else:
                return 'N/A'
        return 'N/A'

    def print_results(self):
        self.log('\n\nResults: \n')
        for i in self.results:
            shift = 27 - len(i)
            if i == 'Warnings':
                self.log(i, ':', ' ' * shift, len(self.results[i]))
            elif i == 'Samples':
                pass
            else:
                self.log(i, ':', ' ' * shift, self.results[i])

    def create_results_dict(self) -> dict:

        self.stars_int = self.make_stars()
        self.stars_str = '*' * self.stars_int if self.stars_int else 'ns'

        return {
            'p-value': self.make_p_value_printed(),
            'Significance(p<0.05)':  True if self.p_value.item() < 0.05 else False,
            'Stars_Printed': self.stars_str,
            'Test_Name': self.test_name,
            'Groups_Compared': self.n_groups,
            'Population_Mean': self.popmean if self.n_groups == 1 else 'N/A',
            'Data_Normaly_Distributed': self.parametric,
            'Parametric_Test_Applied': True if self.test_id in self.test_ids_parametric else False,
            'Paired_Test_Applied': self.paired,
            'Tails': self.tails,
            'p-value_exact': self.p_value.item(),
            'Stars':  self.stars_int,
            # 'Stat_Value': self.test_stat.item(),
            'Warnings': self.warnings,
            'Groups_N': [len(self.data[i]) for i in range(len(self.data))],
            'Groups_Median': [np.median(self.data[i]).item() for i in range(len(self.data))],
            'Groups_Mean': [np.mean(self.data[i]).item() for i in range(len(self.data))],
            'Groups_SD': [np.std(self.data[i]).item() for i in range(len(self.data))],
            'Groups_SE': [np.std(self.data[i]).item() / np.sqrt(len(self.data)).item() for i in range(len(self.data))],
            # actually returns list of lists of numpy dtypes of float64, next make it return regular floats:
            'Samples': self.data,
        }

    def log(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        # print(message, **kwargs)
        self.summary += '\n' + message

    def AddWarning(self, warning_id):
        message = self.warning_ids_all[warning_id]
        self.log(message)
        self.warnings.append(message)


class __InputFormatting():
    def floatify_recursive(self, data):
        if isinstance(data, list):
            # Recursively process sublists and filter out None values
            processed_list = [self.floatify_recursive(item) for item in data]
            return [item for item in processed_list if item is not None]
        else:
            try:
                # Try to convert the item to float
                return np.float64(data)
            except (ValueError, TypeError):
                # If conversion fails, replace with None
                self.warning_flag_non_numeric_data = True
                return None


class StatisticalAnalysis(__StatisticalTests, __NormalityTests, __TextFormatting, __InputFormatting):
    '''
        The main class
        *documentation placeholder*

    '''

    def __init__(self,
                 groups_list,
                 paired=False,
                 tails=2,
                 popmean=None,
                 verbose=True):
        self.results = None
        self.groups_list = groups_list
        self.paired = paired
        self.tails = tails
        self.popmean = popmean
        self.verbose = verbose
        self.n_groups = len(self.groups_list)
        self.warning_flag_non_numeric_data = False
        self.summary = ''
        self.test_ids_parametric = ['anova',
                                    't_test_independend',
                                    't_test_paired',
                                    't_test_single_sample',]
        self.test_ids_all = [  # in aplhabetical order
            'anova',
            'friedman',
            'kruskal_wallis',
            'mann_whitney',
            't_test_independend',
            't_test_paired',
            't_test_single_sample',
            'wilcoxon',
            'wilcoxon_single_sample',
        ]
        self.warning_ids_all = {
            'not-numeric':                     '\nWarning: Non-numeric data was found in input and ignored.\n         Make sure the input data is correct to get the correct results\n',
            'param_test_with_non-normal_data': '\nWarning: Parametric test was manualy chosen for Not-Normaly distributed data.\n         The results might be skewed. \n         Please, run non-parametric test or preform automatic test selection.\n',
            'non-param_test_with_normal_data': '\nWarning: Non-Parametric test was manualy chosen for Normaly distributed data.\n         The results might be skewed. \n         Please, run parametric test or preform automatic test selection.\n',
            'no_pop_mean_set':                 '\nWarning: No Population Mean was set up for single-sample test, used default 0 value.\n         The results might be skewed. \n         Please, set the Population Mean and run the test again.\n',
        }

    def __run_test(self, test='auto'):

        # reset values from previous tests
        self.results = None
        self.warnings = []
        self.normals = []
        self.methods = []
        self.test_name = None
        self.test_id = None
        self.test_stat = None
        self.p_value = None

        self.log('\n' + '-'*67)
        self.log('Statistical analysis initiated for data in {} groups\n'.format(
            len(self.groups_list)))

        # adjusting input data type
        self.data = self.floatify_recursive(self.groups_list)
        if self.warning_flag_non_numeric_data:
            self.AddWarning('not-numeric')

        # delete the empty cols from input
        self.data = [col for col in self.data if any(
            x is not None for x in col)]

        # User input assertion block
        try:
            assert self.tails in [1, 2], 'Tails parameter can be 1 or 2 only'
            assert test in self.test_ids_all or test == 'auto', 'Wrong test id choosen, ensure you called correct function'
            assert not (self.n_groups > 1
                        and (test == 't_test_single_sample'
                             or test == 'wilcoxon_single_sample')), 'Only one group of data must be given for single-group tests'
            assert all(len(
                group) > 2 for group in self.data), 'Each group must contain at least three values'
            assert not (self.paired == True and not all(len(lst) == len(
                self.data[0]) for lst in self.data)), 'Paired groups must be the same length'
            assert not (test == 'friedman' and not all(len(lst) == len(
                self.data[0]) for lst in self.data)), 'Paired groups must be the same length for Friedman Chi Square test'
            assert not (test == 't_test_paired' and not all(len(lst) == len(
                self.data[0]) for lst in self.data)), 'Paired groups must be the same length for Paired t-test'
            assert not (test == 'wilcoxon' and not all(len(lst) == len(
                self.data[0]) for lst in self.data)), 'Paired groups must be the same length for Wilcoxon signed-rank test'
            assert not (test == 'friedman' and self.n_groups <
                        3), 'At least three groups of data must be given for 3-groups tests'
            assert not ((test == 'anova'
                         or test == 'kruskal_wallis') and self.n_groups < 2), 'At least two groups of data must be given for ANOVA or Kruskal Wallis tests'
            assert not ((test == 'wilcoxon'
                         or test == 't_test_independend'
                         or test == 't_test_paired'
                         or test == 'mann_whitney')
                        and self.n_groups != 2), 'Only two groups of data must be given for 2-groups tests'
        except AssertionError as error:
            self.log('\nTest  :', test)
            self.log('Error :', error)
            self.log('-'*67 + '\n')
            print(self.summary)
            return

        # Print the data
        self.print_groups()

        # Normality tests
        self.log('\n\nNormality checked by both Shapiro-Wilk and Lilliefors tests')
        self.log('Group data is normal if at least one results is positive:\n')
        for i, data in enumerate(self.data):
            normal, method = self.check_normality(data)
            self.normals.append(normal)
            self.methods.append(method)
            self.log(
                f'        Group {i+1}: disrtibution is {"normal" if normal else "not normal"}')
        self.parametric = all(self.normals)

        # print test choosen
        self.log('\n\nInput:\n')
        self.log('Data Normaly Distributed:     ', self.parametric)
        self.log('Paired Groups:                ', self.paired)
        self.log('Groups:                       ', self.n_groups)
        self.log('Test chosen by user:          ', test)

        # Wrong test Warnings
        if not test == 'auto' and not self.parametric and test in self.test_ids_parametric:
            self.AddWarning('param_test_with_non-normal_data')
        if not test == 'auto' and self.parametric and not test in self.test_ids_parametric:
            self.AddWarning('non-param_test_with_normal_data')

        if test == 'anova':
            self.anova()
        elif test == 'friedman':
            self.friedman_test()
        elif test == 'kruskal_wallis':
            self.kruskal_wallis_test()
        elif test == 'mann_whitney':
            self.mann_whitney_u_test()
        elif test == 't_test_independend':
            self.t_test_independend()
        elif test == 't_test_paired':
            self.t_test_paired()
        elif test == 't_test_single_sample':
            self.t_test_single_sample()
        elif test == 'wilcoxon':
            self.wilcoxon()
        elif test == 'wilcoxon_single_sample':
            self.wilcoxon_single_sample()
        else:
            self.log('Automatic test selection preformed.')
            self.__auto()

        # print the results
        self.results = self.create_results_dict()
        self.print_results()
        self.log(
            '\n\nResults above are accessible as a dictionary via GetResult() method')
        self.log('-'*67 + '\n')

        # print the results to console:
        if self.verbose == True:
            print(self.summary)

    def __auto(self):

        if self.n_groups == 2:
            if self.paired:
                if self.parametric:
                    return self.t_test_paired()
                else:
                    return self.wilcoxon()
            else:
                if self.parametric:
                    return self.t_test_independend()
                else:
                    return self.mann_whitney_u_test()
        elif self.n_groups == 1:
            if self.parametric:
                return self.t_test_single_sample()
            else:
                return self.wilcoxon_single_sample()
        else:
            if self.paired:
                return self.friedman_test()
            else:
                if self.parametric:
                    return self.anova()
                else:
                    return self.kruskal_wallis_test()

    # public methods:
    def RunAuto(self):
        self.__run_test(test='auto')

    def RunManual(self, test):
        self.__run_test(test)

    def RunAnova(self):
        self.__run_test(test='anova')

    def RunFriedman(self):
        self.__run_test(test='friedman')

    def RunKruskalWallis(self):
        self.__run_test(test='kruskal_wallis')

    def RunMannWhitney(self):
        self.__run_test(test='mann_whitney')

    def RunTtest(self):
        self.__run_test(test='t_test_independend')

    def RunTtestPaired(self):
        self.__run_test(test='t_test_paired')

    def RunTtestSingleSample(self):
        self.__run_test(test='t_test_single_sample')

    def RunWilcoxonSingleSample(self):
        self.__run_test(test='wilcoxon_single_sample')

    def RunWilcoxon(self):
        self.__run_test(test='wilcoxon')

    def GetResult(self):
        if self.results: 
            return self.results
        else:
            self.__run_test(test='auto')
            return self.results

    def GetSummary(self):
        if self.results: 
            return self.summary
        else:
            self.__run_test(test='auto')
            return self.summary

    def PrintSummary(self):
        print(self.summary)


if __name__ == '__main__':
    print('\nThis package works as an imported module only\n')
