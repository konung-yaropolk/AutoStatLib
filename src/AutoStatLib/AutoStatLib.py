from AutoStatLib.statistical_tests import StatisticalTests
from AutoStatLib.normality_tests import NormalityTests
from AutoStatLib.helpers import Helpers
from AutoStatLib.text_formatting import TextFormatting


class StatisticalAnalysis(StatisticalTests, NormalityTests, TextFormatting, Helpers):
    '''
        The main class
        *documentation placeholder*

    '''

    def __init__(self,
                 groups_list,
                 paired=False,
                 tails=2,
                 popmean=None,
                 posthoc=True,
                 verbose=True):
        self.results = None
        self.error = False
        self.groups_list = groups_list
        self.paired = paired
        self.tails = tails
        self.popmean = popmean
        self.posthoc = posthoc
        self.verbose = verbose
        self.n_groups = len(self.groups_list)
        self.warning_flag_non_numeric_data = False
        self.summary = ''

        # test IDs classification:
        self.test_ids_all = [  # in aplhabetical order
            'anova_1w_ordinary',
            'anova_1w_rm',
            'friedman',
            'kruskal_wallis',
            'mann_whitney',
            't_test_independent',
            't_test_paired',
            't_test_single_sample',
            'wilcoxon',
            'wilcoxon_single_sample',
        ]
        self.test_ids_parametric = [
            'anova_1w_ordinary',
            'anova_1w_rm'
            't_test_independent',
            't_test_paired',
            't_test_single_sample',
        ]
        self.test_ids_dependent = [
            'anova_1w_rm',
            'friedman',
            't_test_paired',
            'wilcoxon',
        ]
        self.test_ids_3sample = [
            'anova_1w_ordinary',
            'anova_1w_rm',
            'friedman',
            'kruskal_wallis',
        ]
        self.test_ids_2sample = [
            'mann_whitney',
            't_test_independent',
            't_test_paired',
            'wilcoxon',
        ]
        self.test_ids_1sample = [
            't_test_single_sample',
            'wilcoxon_single_sample',
        ]
        self.warning_ids_all = {
            # 'not-numeric':                     '\nWarning: Non-numeric data was found in input and ignored.\n         Make sure the input data is correct to get the correct results\n',
            'param_test_with_non-normal_data': '\nWarning: Parametric test was manualy chosen for Not-Normaly distributed data.\n         The results might be skewed. \n         Please, run non-parametric test or preform automatic test selection.\n',
            'non-param_test_with_normal_data': '\nWarning: Non-Parametric test was manualy chosen for Normaly distributed data.\n         The results might be skewed. \n         Please, run parametric test or preform automatic test selection.\n',
            'no_pop_mean_set':                 '\nWarning: No Population Mean was set up for single-sample test, used default 0 value.\n         The results might be skewed. \n         Please, set the Population Mean and run the test again.\n',
        }

    def run_test(self, test='auto'):

        # reset values from previous tests
        self.results = None
        self.error = False
        self.warnings = []
        self.normals = []
        self.test_name = None
        self.test_id = None
        self.test_stat = None
        self.p_value = None
        self.posthoc_matrix_df = None
        self.posthoc_matrix = []
        self.posthoc_name = None

        self.log('\n' + '-'*67)
        self.log('Statistical analysis __init__iated for data in {} groups\n'.format(
            len(self.groups_list)))

        # adjusting input data type
        self.data = self.floatify_recursive(self.groups_list)
        if self.warning_flag_non_numeric_data:
            self.log(
                'Text or other non-numeric data in the input was ignored:')

        # delete the empty cols from input
        self.data = [col for col in self.data if any(
            x is not None for x in col)]

        # User input assertion block
        try:
            assert self.data, 'There is no input data'
            assert self.tails in [1, 2], 'Tails parameter can be 1 or 2 only'
            assert test in self.test_ids_all or test == 'auto', 'Wrong test id choosen, ensure you called correct function'
            assert all(len(
                group) >= 4 for group in self.data), 'Each group must contain at least four values'
            assert not (self.paired == True
                        and not all(len(lst) == len(self.data[0]) for lst in self.data)), 'Paired groups must have the same length'
            assert not (test in self.test_ids_dependent
                        and not all(len(lst) == len(self.data[0]) for lst in self.data)), 'Groups must have the same length for dependent groups test'
            assert not (test in self.test_ids_2sample
                        and self.n_groups != 2), f'Only two groups of data must be given for 2-groups tests, got {self.n_groups}'
            assert not (test in self.test_ids_1sample
                        and self.n_groups > 1), f'Only one group of data must be given for single-group tests, got {self.n_groups}'
            assert not (test in self.test_ids_3sample
                        and self.n_groups < 3), f'At least three groups of data must be given for multi-groups tests, got {self.n_groups}'
        except AssertionError as error:
            self.log('\nTest  :', test)
            self.log('Error :', error)
            self.log('-'*67 + '\n')
            self.error = True
            print(self.summary)
            return

        # Print the data
        self.print_groups()

        # Normality tests
        self.log(
            '\n\nThe group is assumed to be normally distributed if at least one')
        self.log(
            'normality test result is positive. Normality checked by tests:')
        self.log('Shapiro-Wilk, Lilliefors, Anderson-Darling, D\'Agostino-Pearson')
        self.log(
            '[+] -positive, [-] -negative, [ ] -too small group for the test\n')
        self.log('        Test   :   SW  LF  AD  AP  ')
        for i, data in enumerate(self.data):
            poll = self.check_normality(data)
            isnormal = any(poll)
            poll_print = tuple(
                '+' if x is True else '-' if x is False else ' ' if x is None else 'e' for x in poll)
            self.normals.append(isnormal)
            self.log(
                f'        Group {i+1}:    {poll_print[0]}   {poll_print[1]}   {poll_print[2]}   {poll_print[3]}   so disrtibution seems {"normal" if isnormal else "not normal"}')
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

        # run the test

        if test in self.test_ids_all:
            self.run_test_by_id(test)
        else:
            self.run_test_auto()

        # print the results
        self.results = self.create_results_dict()
        self.print_results()
        self.log(
            '\n\nResults above are accessible as a dictionary via GetResult() method')
        self.log('-'*67 + '\n')

        # print the results to console:
        if self.verbose == True:
            print(self.summary)

    # public methods:

    def RunAuto(self):
        self.run_test(test='auto')

    def RunManual(self, test):
        self.run_test(test)

    def RunOnewayAnova(self):
        self.run_test(test='anova_1w_ordinary')

    def RunOnewayAnovaRM(self):
        self.run_test(test='anova_1w_rm')

    def RunFriedman(self):
        self.run_test(test='friedman')

    def RunKruskalWallis(self):
        self.run_test(test='kruskal_wallis')

    def RunMannWhitney(self):
        self.run_test(test='mann_whitney')

    def RunTtest(self):
        self.run_test(test='t_test_independent')

    def RunTtestPaired(self):
        self.run_test(test='t_test_paired')

    def RunTtestSingleSample(self):
        self.run_test(test='t_test_single_sample')

    def RunWilcoxonSingleSample(self):
        self.run_test(test='wilcoxon_single_sample')

    def RunWilcoxon(self):
        self.run_test(test='wilcoxon')

    def GetResult(self):
        if not self.results and not self.error:
            print('No test chosen, no results to output')
            # self.run_test(test='auto')
            return self.results
        if not self.results and self.error:
            print('Error occured, no results to output')
            return {}
        else:
            return self.results

    def GetSummary(self):
        if not self.results and not self.error:
            print('No test chosen, no summary to output')
            # self.run_test(test='auto')
            return self.summary
        else:
            return self.summary

    def GetTestIDs(self):
        return self.test_ids_all

    def PrintSummary(self):
        print(self.summary)


if __name__ == '__main__':
    print('This package works as an imported module only.\nUse "import autostatlib" statement')
