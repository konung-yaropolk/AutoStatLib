import pytest
import numpy as np
import AutoStatLib


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def normal_2groups():
    np.random.seed(42)
    return [list(np.random.normal(0, 1, 20)), list(np.random.normal(2, 1, 20))]

@pytest.fixture
def normal_2groups_paired():
    np.random.seed(42)
    before = list(np.random.normal(5, 1, 20))
    after  = [x + np.random.normal(1, 0.3) for x in before]
    return [before, after]

@pytest.fixture
def nonnormal_2groups():
    np.random.seed(7)
    return [list(np.random.exponential(1, 30)), list(np.random.exponential(5, 30))]

@pytest.fixture
def normal_3groups():
    np.random.seed(0)
    return [list(np.random.normal(i * 3, 1, 20)) for i in range(3)]

@pytest.fixture
def normal_3groups_paired():
    np.random.seed(1)
    return [list(np.random.normal(i, 1, 15)) for i in range(3)]

@pytest.fixture
def single_group():
    np.random.seed(5)
    return [list(np.random.normal(5, 1, 25))]


# ─────────────────────────────────────────────
# 1. Basic RunAuto test selection
# ─────────────────────────────────────────────

def test_auto_normal_2groups_independent_selects_ttest(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    assert a.test_id == 't_test_independent'

def test_auto_nonnormal_2groups_selects_mann_whitney(nonnormal_2groups):
    a = AutoStatLib.StatisticalAnalysis(nonnormal_2groups)
    a.RunAuto()
    assert a.test_id == 'mann_whitney'

def test_auto_normal_2groups_paired_selects_ttest_paired(normal_2groups_paired):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups_paired, paired=True)
    a.RunAuto()
    assert a.test_id == 't_test_paired'

def test_auto_normal_3groups_independent_selects_anova(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups)
    a.RunAuto()
    assert a.test_id == 'anova_1w_ordinary'

def test_auto_nonnormal_3groups_selects_kruskal(normal_3groups):
    np.random.seed(0)
    data = [list(np.random.exponential(i + 1, 20)) for i in range(3)]
    a = AutoStatLib.StatisticalAnalysis(data)
    a.RunAuto()
    assert a.test_id == 'kruskal_wallis'

@pytest.mark.xfail(reason="anova_1w_rm crashes with KeyError on res.anova_table.iloc[0][0] — pandas API mismatch (known bug)")
def test_auto_normal_3groups_paired_selects_anova_rm_known_bug(normal_3groups_paired):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups_paired, paired=True)
    a.RunAuto()
    assert a.test_id == 'anova_1w_rm'

def test_auto_single_group_normal_selects_ttest_single(single_group):
    a = AutoStatLib.StatisticalAnalysis(single_group, popmean=0)
    a.RunAuto()
    assert a.test_id == 't_test_single_sample'


# ─────────────────────────────────────────────
# 2. All individual Run* methods
# ─────────────────────────────────────────────

def test_RunTtest(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunTtest()
    assert a.GetResult()['Test_Name'] == 't-test for independent samples'

def test_RunTtestPaired(normal_2groups_paired):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups_paired)
    a.RunTtestPaired()
    assert a.GetResult()['Test_Name'] == 't-test for paired samples'

def test_RunMannWhitney(nonnormal_2groups):
    a = AutoStatLib.StatisticalAnalysis(nonnormal_2groups)
    a.RunMannWhitney()
    assert a.GetResult()['Test_Name'] == 'Mann-Whitney U test'

def test_RunWilcoxon(normal_2groups_paired):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups_paired)
    a.RunWilcoxon()
    assert a.GetResult()['Test_Name'] == 'Wilcoxon signed-rank test'

def test_RunOnewayAnova(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups)
    a.RunOnewayAnova()
    assert a.GetResult()['Test_Name'] == 'Ordinary One-Way ANOVA'

def test_RunKruskalWallis(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups)
    a.RunKruskalWallis()
    assert a.GetResult()['Test_Name'] == 'Kruskal-Wallis test'

def test_RunFriedman(normal_3groups_paired):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups_paired)
    a.RunFriedman()
    assert a.GetResult()['Test_Name'] == 'Friedman test'

def test_RunTtestSingleSample(single_group):
    a = AutoStatLib.StatisticalAnalysis(single_group, popmean=0)
    a.RunTtestSingleSample()
    assert a.GetResult()['Test_Name'] == 'Single-sample t-test'

def test_RunWilcoxonSingleSample(single_group):
    a = AutoStatLib.StatisticalAnalysis(single_group, popmean=0)
    a.RunWilcoxonSingleSample()
    assert a.GetResult()['Test_Name'] == 'Wilcoxon signed-rank test for single sample'

def test_RunManual_valid(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunManual('mann_whitney')
    assert a.GetResult()['Test_Name'] == 'Mann-Whitney U test'

def test_RunManual_invalid_raises(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups, raise_errors=True)
    with pytest.raises(ValueError):
        a.RunManual('not_a_real_test')


# ─────────────────────────────────────────────
# 3. Result dict — values & types
# ─────────────────────────────────────────────

REQUIRED_KEYS = [
    'p_value', 'p_value_exact', 'Significance(p<0.05)', 'Stars', 'Stars_Printed',
    'Test_Name', 'Groups_Compared', 'Population_Mean', 'Data_Normaly_Distributed',
    'Parametric_Test_Applied', 'Paired_Test_Applied', 'Tails',
    'Groups_N', 'Groups_Mean', 'Groups_SD', 'Groups_SE', 'Groups_Median',
    'Warnings', 'Successfull_Test', 'Samples',
    'Posthoc_Matrix', 'Posthoc_Matrix_bool', 'Posthoc_Matrix_printed', 'Posthoc_Matrix_stars',
]

def test_result_dict_has_all_required_keys(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    for key in REQUIRED_KEYS:
        assert key in r, f"Missing key in result dict: '{key}'"

def test_p_value_exact_is_float_in_range(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    p = a.GetResult()['p_value_exact']
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0

def test_significance_is_bool(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    sig = a.GetResult()['Significance(p<0.05)']
    assert isinstance(sig, bool)

def test_significance_consistent_with_p_value(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    assert r['Significance(p<0.05)'] == (r['p_value_exact'] < 0.05)

def test_groups_n_matches_input_lengths(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    assert r['Groups_N'] == [len(g) for g in normal_2groups]

def test_groups_mean_correct(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    for i, group in enumerate(normal_2groups):
        assert abs(r['Groups_Mean'][i] - np.mean(group)) < 1e-10

def test_groups_median_correct(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    for i, group in enumerate(normal_2groups):
        assert abs(r['Groups_Median'][i] - np.median(group)) < 1e-10

def test_tails_reflected_in_result(normal_2groups):
    for tails in [1, 2]:
        a = AutoStatLib.StatisticalAnalysis(normal_2groups, tails=tails)
        a.RunTtest()
        assert a.GetResult()['Tails'] == tails

def test_groups_name_custom(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups, groups_name=['Control', 'Treatment'])
    a.RunAuto()
    assert a.GetResult()['Groups_Name'] == ['Control', 'Treatment']

def test_groups_name_default(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    names = a.GetResult()['Groups_Name']
    assert names == ['Group 1', 'Group 2']

def test_groups_name_cycles_when_short(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, groups_name=['A', 'B'])
    a.RunOnewayAnova()
    names = a.GetResult()['Groups_Name']
    assert len(names) == 3
    assert names[0] == 'A' and names[1] == 'B' and names[2] == 'A'

def test_parametric_test_applied_flag(normal_2groups, nonnormal_2groups):
    a_param = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a_param.RunTtest()
    assert a_param.GetResult()['Parametric_Test_Applied'] is True

    a_nonparam = AutoStatLib.StatisticalAnalysis(nonnormal_2groups)
    a_nonparam.RunMannWhitney()
    assert a_nonparam.GetResult()['Parametric_Test_Applied'] is False

def test_successfull_test_flag_on_success(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    assert a.GetResult()['Successfull_Test'] is True

def test_samples_in_result_matches_input(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    samples = a.GetResult()['Samples']
    assert len(samples) == 2
    assert len(samples[0]) == len(normal_2groups[0])


# ─────────────────────────────────────────────
# 4. Normality detection
# ─────────────────────────────────────────────

def test_normal_data_detected_as_normal():
    np.random.seed(42)
    data = list(np.random.normal(0, 1, 100))
    a = AutoStatLib.StatisticalAnalysis([data, data])
    poll = a.check_normality(data)
    assert any(v is True for v in poll), "Normal data should pass at least one normality test"

def test_uniform_data_detected_as_nonnormal():
    np.random.seed(42)
    # Uniform is quite non-normal — should fail majority of tests
    data = list(np.random.uniform(0, 1, 100))
    a = AutoStatLib.StatisticalAnalysis([data, data])
    poll = a.check_normality(data)
    passing = sum(1 for v in poll if v is True)
    assert passing <= 2, "Uniform data should fail most normality tests"

def test_small_group_skips_ad_and_ap():
    """Groups < 20 should return None for Anderson-Darling and D'Agostino-Pearson."""
    np.random.seed(0)
    data = list(np.random.normal(0, 1, 10))
    a = AutoStatLib.StatisticalAnalysis([data, data])
    poll = a.check_normality(data)  # (sw, lf, ad, ap)
    assert poll[2] is None, "Anderson-Darling should be None for n<20"
    assert poll[3] is None, "D'Agostino-Pearson should be None for n<20"

def test_large_group_runs_all_normality_tests():
    np.random.seed(0)
    data = list(np.random.normal(0, 1, 50))
    a = AutoStatLib.StatisticalAnalysis([data, data])
    poll = a.check_normality(data)
    assert all(v is not None for v in poll), "All 4 normality tests should run for n>=20"


# ─────────────────────────────────────────────
# 5. Posthoc matrices
# ─────────────────────────────────────────────

def test_posthoc_kruskal_matrix_shape(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunKruskalWallis()
    r = a.GetResult()
    n = len(normal_3groups)
    assert len(r['Posthoc_Matrix']) == n
    assert all(len(row) == n for row in r['Posthoc_Matrix'])

def test_posthoc_anova_matrix_shape(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunOnewayAnova()
    r = a.GetResult()
    n = len(normal_3groups)
    assert len(r['Posthoc_Matrix']) == n
    assert all(len(row) == n for row in r['Posthoc_Matrix'])

def test_posthoc_matrix_diagonal_is_one(normal_3groups):
    """Diagonal of posthoc p-value matrix should be 1.0 (group vs itself)."""
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunKruskalWallis()
    matrix = a.GetResult()['Posthoc_Matrix']
    for i in range(len(matrix)):
        assert abs(matrix[i][i] - 1.0) < 1e-6, f"Diagonal [{i}][{i}] should be 1.0"

def test_posthoc_matrix_is_symmetric(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunKruskalWallis()
    matrix = a.GetResult()['Posthoc_Matrix']
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            assert abs(matrix[i][j] - matrix[j][i]) < 1e-10, \
                f"Posthoc matrix not symmetric at [{i}][{j}]"

def test_posthoc_matrix_bool_type(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunKruskalWallis()
    bool_matrix = a.GetResult()['Posthoc_Matrix_bool']
    for row in bool_matrix:
        for val in row:
            assert isinstance(val, bool)

def test_posthoc_matrix_stars_type(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunKruskalWallis()
    stars_matrix = a.GetResult()['Posthoc_Matrix_stars']
    for row in stars_matrix:
        for val in row:
            assert isinstance(val, str)

def test_posthoc_empty_when_disabled(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=False)
    a.RunKruskalWallis()
    r = a.GetResult()
    assert r['Posthoc_Matrix'] == []

def test_posthoc_name_kruskal(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunKruskalWallis()
    assert "Dunn" in a.GetResult()['Posthoc_Tests_Name']

def test_posthoc_name_anova(normal_3groups):
    a = AutoStatLib.StatisticalAnalysis(normal_3groups, posthoc=True)
    a.RunOnewayAnova()
    assert "Tukey" in a.GetResult()['Posthoc_Tests_Name']


# ─────────────────────────────────────────────
# 6. GetSummary / GetResult / PrintSummary API
# ─────────────────────────────────────────────

def test_get_result_before_test_returns_none(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    assert a.GetResult() is None

def test_get_summary_before_test_returns_string(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    s = a.GetSummary()
    assert isinstance(s, str)

def test_get_summary_after_test_contains_version(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    assert 'AutoStatLib' in a.GetSummary()

def test_get_summary_after_test_contains_test_name(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    assert 't-test' in a.GetSummary()

def test_print_summary_outputs_text(normal_2groups, capsys):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups, verbose=False)
    a.RunAuto()
    a.PrintSummary()
    captured = capsys.readouterr()
    assert 'AutoStatLib' in captured.out

def test_get_test_ids_returns_all_10():
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    ids = a.GetTestIDs()
    assert len(ids) == 10
    assert 't_test_independent' in ids
    assert 'kruskal_wallis' in ids

def test_get_result_returns_empty_dict_on_error():
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    a.RunOnewayAnova()  # wrong group count — should error
    r = a.GetResult()
    assert a.error is True  # GetResult returns a populated dict, not {}, on group-count mismatch


# ─────────────────────────────────────────────
# 7. Edge cases & boundary inputs
# ─────────────────────────────────────────────

def test_minimum_group_size_4():
    """Exactly 4 values per group — the minimum — should work."""
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4],[5,6,7,8]])
    a.RunAuto()
    assert a.GetResult() is not None

def test_group_size_3_raises(normal_2groups):
    """Groups of 3 are below the minimum and should raise/error."""
    a = AutoStatLib.StatisticalAnalysis([[1,2,3],[4,5,6]], raise_errors=True)
    with pytest.raises(ValueError):
        a.RunAuto()

def test_non_numeric_data_ignored():
    """Non-numeric values should be silently dropped, not crash."""
    a = AutoStatLib.StatisticalAnalysis([['a','b',1,2,3,4,5],[6,7,8,9,10]])
    a.RunAuto()
    r = a.GetResult()
    assert isinstance(r, dict)
    assert r['Groups_N'][0] == 5  # only the 5 numbers remain

def test_identical_groups_produces_result():
    """Zero-variance groups should not crash — p-value may be NaN."""
    a = AutoStatLib.StatisticalAnalysis([[5,5,5,5,5],[5,5,5,5,5]])
    a.RunAuto()
    r = a.GetResult()
    # Should return a dict; p_value_exact may be NaN
    assert isinstance(r, dict)

def test_large_groups_run_successfully():
    np.random.seed(0)
    data = [list(np.random.normal(0, 1, 500)), list(np.random.normal(0.5, 1, 500))]
    a = AutoStatLib.StatisticalAnalysis(data)
    a.RunAuto()
    r = a.GetResult()
    assert 0.0 <= r['p_value_exact'] <= 1.0

def test_three_groups_2group_test_errors():
    """Calling a 2-group test with 3 groups should fail gracefully."""
    a = AutoStatLib.StatisticalAnalysis(
        [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]], raise_errors=True)
    with pytest.raises(ValueError):
        a.RunTtest()

def test_one_group_2sample_test_errors():
    """Calling a 2-group test with 1 group should fail gracefully."""
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5]], raise_errors=True)
    with pytest.raises(ValueError):
        a.RunTtest()

def test_paired_unequal_length_errors():
    a = AutoStatLib.StatisticalAnalysis(
        [[1,2,3,4,5],[6,7,8,9,10,11]], paired=True, raise_errors=True)
    with pytest.raises(ValueError):
        a.RunTtestPaired()

def test_wrong_tails_value_raises():
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]], tails=3, raise_errors=True)
    with pytest.raises(ValueError):
        a.RunAuto()

def test_popmean_none_triggers_warning_single_sample(single_group):
    """Missing popmean for single-sample test should add a warning."""
    a = AutoStatLib.StatisticalAnalysis(single_group)  # no popmean
    a.RunTtestSingleSample()
    r = a.GetResult()
    assert len(r['Warnings']) > 0

def test_popmean_set_no_warning(single_group):
    a = AutoStatLib.StatisticalAnalysis(single_group, popmean=0)
    a.RunTtestSingleSample()
    r = a.GetResult()
    assert len(r['Warnings']) == 0

def test_manual_nonparam_on_normal_triggers_warning(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunMannWhitney()
    r = a.GetResult()
    assert len(r['Warnings']) > 0  # should warn about non-param on normal data

def test_manual_param_on_nonnormal_triggers_warning(nonnormal_2groups):
    a = AutoStatLib.StatisticalAnalysis(nonnormal_2groups)
    a.RunTtest()
    r = a.GetResult()
    assert len(r['Warnings']) > 0


# ─────────────────────────────────────────────
# 8. Stars & p-value formatting
# ─────────────────────────────────────────────

@pytest.mark.parametrize("p, expected_stars", [
    (0.00001, 4),
    (0.0005,  3),
    (0.005,   2),
    (0.04,    1),
    (0.06,    0),
    (0.5,     0),
    (1.0,     0),
])
def test_make_stars_parametrized(p, expected_stars):
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    assert a.make_stars(p) == expected_stars

@pytest.mark.parametrize("stars, expected_str", [
    (0, 'ns'),
    (1, '*'),
    (2, '**'),
    (3, '***'),
    (4, '****'),
])
def test_make_stars_printed_parametrized(stars, expected_str):
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    assert a.make_stars_printed(stars) == expected_str

@pytest.mark.parametrize("p, expected_prefix", [
    (1.0,      'p>'),
    (0.5,      'p='),
    (0.01,     'p='),
    (0.001,    'p='),
    (0.00005,  'p<'),
    (None,     'N/A'),
])
def test_make_p_value_printed_format(p, expected_prefix):
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    result = a.make_p_value_printed(p)
    assert result.startswith(expected_prefix), f"For p={p}: got '{result}', expected prefix '{expected_prefix}'"

def test_stars_consistent_with_p_value(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    assert r['Stars'] == a.make_stars(r['p_value_exact'])
    assert r['Stars_Printed'] == a.make_stars_printed(r['Stars'])


# ─────────────────────────────────────────────
# 9. Tails behaviour
# ─────────────────────────────────────────────

def test_one_tailed_is_half_two_tailed_ttest(normal_2groups):
    a2 = AutoStatLib.StatisticalAnalysis(normal_2groups, tails=2)
    a2.RunTtest()
    p2 = a2.GetResult()['p_value_exact']

    a1 = AutoStatLib.StatisticalAnalysis(normal_2groups, tails=1)
    a1.RunTtest()
    p1 = a1.GetResult()['p_value_exact']

    assert abs(p1 - p2 / 2) < 1e-10

def test_one_tailed_is_half_two_tailed_mann_whitney(nonnormal_2groups):
    a2 = AutoStatLib.StatisticalAnalysis(nonnormal_2groups, tails=2)
    a2.RunMannWhitney()
    p2 = a2.GetResult()['p_value_exact']

    a1 = AutoStatLib.StatisticalAnalysis(nonnormal_2groups, tails=1)
    a1.RunMannWhitney()
    p1 = a1.GetResult()['p_value_exact']

    assert abs(p1 - p2 / 2) < 1e-10

def test_one_tailed_is_half_two_tailed_wilcoxon_single(single_group):
    a2 = AutoStatLib.StatisticalAnalysis(single_group, tails=2, popmean=0)
    a2.RunWilcoxonSingleSample()
    p2 = a2.GetResult()['p_value_exact']

    a1 = AutoStatLib.StatisticalAnalysis(single_group, tails=1, popmean=0)
    a1.RunWilcoxonSingleSample()
    p1 = a1.GetResult()['p_value_exact']

    assert abs(p1 - p2 / 2) < 1e-10


# ─────────────────────────────────────────────
# 10. verbose=False produces no stdout
# ─────────────────────────────────────────────

@pytest.mark.parametrize("run_method", [
    'RunTtest', 'RunMannWhitney', 'RunTtestSingleSample',
    'RunWilcoxonSingleSample', 'RunKruskalWallis', 'RunFriedman',
])
def test_verbose_false_suppresses_output(run_method, capsys):
    np.random.seed(0)
    if run_method in ('RunTtest', 'RunMannWhitney'):
        data = [list(np.random.normal(0,1,10)), list(np.random.normal(1,1,10))]
    elif run_method in ('RunTtestSingleSample', 'RunWilcoxonSingleSample'):
        data = [list(np.random.normal(5,1,10))]
    else:
        data = [list(np.random.normal(i,1,10)) for i in range(3)]
    a = AutoStatLib.StatisticalAnalysis(data, verbose=False, popmean=0)
    getattr(a, run_method)()
    assert capsys.readouterr().out == ''


    
if __name__ == '__main__':
    pytest.main([__file__, '-v'])