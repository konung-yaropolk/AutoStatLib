# tests/test_autostatlib.py
import pytest
import numpy as np
import AutoStatLib


# --- Fixtures ---
@pytest.fixture
def normal_2groups():
    np.random.seed(42)
    return [list(np.random.normal(0, 1, 20)), list(np.random.normal(1, 1, 20))]

@pytest.fixture
def nonnormal_2groups():
    np.random.seed(42)
    return [list(np.random.exponential(1, 20)), list(np.random.exponential(2, 20))]


# --- Basic functionality ---
def test_run_auto_returns_result(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    assert isinstance(r, dict)
    assert 'p_value_exact' in r
    assert 0.0 <= r['p_value_exact'] <= 1.0

def test_run_auto_selects_ttest_for_normal(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    assert a.test_id == 't_test_independent'

def test_run_auto_selects_mann_whitney_for_nonnormal(nonnormal_2groups):
    a = AutoStatLib.StatisticalAnalysis(nonnormal_2groups)
    a.RunAuto()
    assert a.test_id == 'mann_whitney'

def test_verbose_false_no_print(normal_2groups, capsys):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups, verbose=False)
    a.RunAuto()
    captured = capsys.readouterr()
    assert captured.out == ''


# --- Result dict completeness ---
def test_result_dict_keys(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    required_keys = [
        'p_value', 'p_value_exact', 'Significance(p<0.05)', 'Stars',
        'Stars_Printed', 'Test_Name', 'Groups_N', 'Groups_Mean',
        'Groups_SD', 'Groups_SE', 'Groups_Median', 'Warnings',
    ]
    for key in required_keys:
        assert key in r, f"Missing key: {key}"

def test_se_calculation_correct(normal_2groups):
    """SE = std / sqrt(n) per group, not std / sqrt(num_groups)."""
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    r = a.GetResult()
    for i, group in enumerate(normal_2groups):
        expected_se = np.std(group, ddof=1) / np.sqrt(len(group))
        assert abs(r['Groups_SE'][i] - expected_se) < 0.01, \
            f"SE for group {i} is wrong: {r['Groups_SE'][i]} vs {expected_se}"


# --- Error handling ---
def test_raises_on_too_few_samples():
    a = AutoStatLib.StatisticalAnalysis([[1, 2, 3], [4, 5, 6]], raise_errors=True)
    with pytest.raises(ValueError):
        a.RunAuto()

def test_empty_result_on_wrong_group_count():
    """3-group test requested with 2 groups should fail gracefully."""
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    a.RunOnewayAnova()
    assert a.GetResult() == {} or a.error

def test_non_numeric_data_filtered():
    a = AutoStatLib.StatisticalAnalysis([['x', 'y', 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    a.RunAuto()
    r = a.GetResult()
    assert isinstance(r, dict)


# --- Single-sample tests ---
def test_single_sample_ttest():
    data = [list(np.random.normal(5, 1, 30))]
    a = AutoStatLib.StatisticalAnalysis(data, popmean=0)
    a.RunTtestSingleSample()
    r = a.GetResult()
    assert r['Significance(p<0.05)'] is True  # mean ~5 vs popmean=0 should be significant

def test_no_popmean_triggers_warning():
    data = [list(np.random.normal(1, 1, 20))]
    a = AutoStatLib.StatisticalAnalysis(data)
    a.RunTtestSingleSample()
    r = a.GetResult()
    assert len(r['Warnings']) > 0


# --- Paired tests ---
def test_paired_ttest_equal_length_required():
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10,11]], paired=True, raise_errors=True)
    with pytest.raises(ValueError):
        a.RunTtestPaired()


# --- Posthoc ---
def test_posthoc_kruskal():
    np.random.seed(0)
    data = [list(np.random.normal(i, 1, 20)) for i in range(3)]
    a = AutoStatLib.StatisticalAnalysis(data, posthoc=True)
    a.RunKruskalWallis()
    r = a.GetResult()
    assert len(r['Posthoc_Matrix']) == 3
    assert len(r['Posthoc_Matrix'][0]) == 3


# --- Stars ---
@pytest.mark.parametrize("p,expected", [
    (0.001, 3), (0.01, 2), (0.04, 1), (0.1, 0), (0.00001, 4)
])
def test_make_stars(p, expected):
    a = AutoStatLib.StatisticalAnalysis([[1,2,3,4,5],[6,7,8,9,10]])
    assert a.make_stars(p) == expected


# --- Tails ---
def test_one_tailed_p_less_than_two_tailed(normal_2groups):
    a2 = AutoStatLib.StatisticalAnalysis(normal_2groups, tails=2)
    a2.RunTtest()
    p2 = a2.GetResult()['p_value_exact']

    a1 = AutoStatLib.StatisticalAnalysis(normal_2groups, tails=1)
    a1.RunTtest()
    p1 = a1.GetResult()['p_value_exact']

    assert abs(p1 - p2 / 2) < 1e-10


# --- GetSummary ---
def test_get_summary_contains_version(normal_2groups):
    a = AutoStatLib.StatisticalAnalysis(normal_2groups)
    a.RunAuto()
    assert 'AutoStatLib' in a.GetSummary()