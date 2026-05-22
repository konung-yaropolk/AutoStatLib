"""
tests/test_statplots.py
Full test suite for AutoStatLib.StatPlots

Run with:
    pytest tests/test_statplots.py -v
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before any other plt import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from AutoStatLib.StatPlots import (
    BaseStatPlot,
    BarStatPlot,
    BoxStatPlot,
    Helpers,
    ScatterStatPlot,
    SwarmStatPlot,
    SwarmStatPlot_subgrouping_betta,
    ViolinStatPlot,
)

ALL_PLOT_CLASSES = [
    BarStatPlot,
    ViolinStatPlot,
    BoxStatPlot,
    ScatterStatPlot,
    SwarmStatPlot,
]


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to avoid resource warnings."""
    yield
    plt.close("all")


@pytest.fixture
def data2():
    np.random.seed(42)
    return [list(np.random.normal(0, 1, 20)), list(np.random.normal(2, 1, 20))]


@pytest.fixture
def data3():
    np.random.seed(0)
    return [list(np.random.normal(i * 3, 1, 20)) for i in range(3)]


@pytest.fixture
def data4():
    np.random.seed(0)
    return [list(np.random.normal(i * 3, 1, 20)) for i in range(4)]


@pytest.fixture
def data5():
    np.random.seed(0)
    return [list(np.random.normal(i * 3, 1, 20)) for i in range(5)]


@pytest.fixture
def posthoc3():
    return [[1.0, 0.01, 0.001], [0.01, 1.0, 0.5], [0.001, 0.5, 1.0]]


@pytest.fixture
def posthoc4():
    return [
        [1.0, 0.01, 0.001, 0.0001],
        [0.01, 1.0, 0.5, 0.1],
        [0.001, 0.5, 1.0, 0.3],
        [0.0001, 0.1, 0.3, 1.0],
    ]


@pytest.fixture
def posthoc5():
    m = [[0.05] * 5 for _ in range(5)]
    for i in range(5):
        m[i][i] = 1.0
    return m


# ─────────────────────────────────────────────
# 1. Helpers — utility methods
# ─────────────────────────────────────────────

class TestHelpers:

    @pytest.fixture
    def h(self):
        return Helpers()

    # make_stars
    @pytest.mark.parametrize("p, expected", [
        (0.00001, 4),
        (0.0005,  3),
        (0.005,   2),
        (0.03,    1),
        (0.1,     0),
        (1.0,     0),
        (None,    0),
    ])
    def test_make_stars(self, h, p, expected):
        assert h.make_stars(p) == expected

    # make_stars_printed
    @pytest.mark.parametrize("n, expected", [
        (0, "ns"),
        (1, "*"),
        (2, "**"),
        (3, "***"),
        (4, "****"),
    ])
    def test_make_stars_printed(self, h, n, expected):
        assert h.make_stars_printed(n) == expected

    # make_p_value_printed
    @pytest.mark.parametrize("p, expected", [
        (1.0,       "p>0.99"),
        (0.5,       "p=0.5"),
        (0.05,      "p=0.05"),
        (0.005,     "p=0.005"),
        (0.0005,    "p=0.0005"),
        (0.00005,   "p<0.0001"),
        (0.0000001, "p<0.0001"),
        (None,      "N/A"),
    ])
    def test_make_p_value_printed(self, h, p, expected):
        assert h.make_p_value_printed(p) == expected

    def test_make_p_value_printed_returns_str(self, h):
        assert isinstance(h.make_p_value_printed(0.03), str)

    # expand_counts
    def test_expand_counts_basic(self, h):
        assert h.expand_counts([3, 2, 1]) == [0, 0, 0, 1, 1, 2]

    def test_expand_counts_zeros_filtered(self, h):
        assert h.expand_counts([0, 0, 2]) == [0, 0]

    def test_expand_counts_empty(self, h):
        assert h.expand_counts([]) == [0]

    def test_expand_counts_single(self, h):
        assert h.expand_counts([4]) == [0, 0, 0, 0]

    # transpose
    def test_transpose_square(self, h):
        assert h.transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]

    def test_transpose_rectangular(self, h):
        assert h.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

    def test_transpose_roundtrip(self, h):
        data = [[1, 2, 3], [4, 5, 6]]
        assert h.transpose(h.transpose(data)) == data

    # colors_to_rgba
    def test_colors_to_rgba_alpha(self, h):
        rgba = h.colors_to_rgba(["red", "blue"], alpha=0.5)
        assert all(c[3] == 0.5 for c in rgba)

    def test_colors_to_rgba_default_alpha(self, h):
        rgba = h.colors_to_rgba(["red"])
        assert rgba[0][3] == 0.35

    def test_colors_to_rgba_length(self, h):
        rgba = h.colors_to_rgba(["red", "green", "blue"])
        assert len(rgba) == 3

    def test_colors_to_rgba_values(self, h):
        rgba = h.colors_to_rgba(["red"])
        assert rgba[0][0] == pytest.approx(1.0)  # red channel
        assert rgba[0][1] == pytest.approx(0.0)  # green channel
        assert rgba[0][2] == pytest.approx(0.0)  # blue channel

    # get_colors
    def test_get_colors_default_length(self, h):
        edge, fill = h.get_colors(None)
        assert len(edge) == 10  # 9 Set1 + 1 black prepended
        assert len(fill) == 10

    def test_get_colors_default_fill_alpha(self, h):
        _, fill = h.get_colors(None)
        assert all(c[3] == pytest.approx(0.35) for c in fill)

    def test_get_colors_custom(self, h):
        edge, fill = h.get_colors(["red", "blue"])
        assert len(edge) == 2
        assert len(fill) == 2

    def test_get_colors_invalid_entry_becomes_black(self, h):
        edge, _ = h.get_colors(["red", "notacolor", "blue"])
        assert edge[1] == "k"

    def test_get_colors_valid_entries_preserved(self, h):
        edge, _ = h.get_colors(["red", "blue"])
        assert edge[0] == "red"
        assert edge[1] == "blue"


# ─────────────────────────────────────────────
# 2. BaseStatPlot — initialisation & attributes
# ─────────────────────────────────────────────

class TestBaseStatPlotInit:

    def test_n_groups_correct(self, data2):
        p = BarStatPlot(data2)
        assert p.n_groups == 2

    def test_n_list_correct(self, data2):
        p = BarStatPlot(data2)
        assert p.n == [20, 20]

    def test_mean_length(self, data2):
        p = BarStatPlot(data2)
        assert len(p.mean) == 2

    def test_mean_values_correct(self, data2):
        p = BarStatPlot(data2)
        for i, group in enumerate(data2):
            assert abs(p.mean[i] - np.mean(group)) < 1e-10

    def test_median_values_correct(self, data2):
        p = BarStatPlot(data2)
        for i, group in enumerate(data2):
            assert abs(p.median[i] - np.median(group)) < 1e-10

    def test_sd_uses_ddof1(self, data2):
        p = BarStatPlot(data2)
        for i, group in enumerate(data2):
            assert abs(p.sd[i] - np.std(group, ddof=1)) < 1e-10

    def test_sem_correct(self, data2):
        p = BarStatPlot(data2)
        for i, group in enumerate(data2):
            expected = np.std(group, ddof=1) / np.sqrt(len(group))
            assert abs(p.sem[i] - expected) < 1e-10

    def test_y_max_correct(self, data2):
        p = BarStatPlot(data2)
        assert abs(p.y_max - max(max(g) for g in data2)) < 1e-10

    def test_p_printed_format(self, data2):
        p = BarStatPlot(data2, p_value_exact=0.03)
        assert p.p_printed == "p=0.03"

    def test_stars_printed_significant(self, data2):
        p = BarStatPlot(data2, p_value_exact=0.03)
        assert p.stars_printed == "*"

    def test_no_p_value_gives_na(self, data2):
        p = BarStatPlot(data2)
        assert p.p_printed == "N/A"
        assert p.stars_printed == "ns"

    def test_error_false_on_valid_data(self, data2):
        p = BarStatPlot(data2)
        assert p.error is False

    def test_groups_name_default(self, data2):
        p = BarStatPlot(data2)
        assert p.groups_name == [""]

    def test_groups_name_custom(self, data2):
        p = BarStatPlot(data2, Groups_Name=["Control", "Treatment"])
        assert p.groups_name == ["Control", "Treatment"]

    def test_dependent_default_false(self, data2):
        p = BarStatPlot(data2)
        assert p.dependent is False

    def test_dependent_true(self, data2):
        p = BarStatPlot(data2, Paired_Test_Applied=True)
        assert p.dependent is True

    def test_subgrouping_default(self, data2):
        p = BarStatPlot(data2)
        assert p.subgrouping == [0]

    def test_subgrouping_custom(self, data2):
        p = BarStatPlot(data2, subgrouping=[3, 2])
        assert p.subgrouping_arrange == [0, 0, 0, 1, 1]

    def test_posthoc_matrix_default_empty(self, data2):
        p = BarStatPlot(data2)
        assert p.posthoc_matrix == []

    def test_posthoc_matrix_stored(self, data3, posthoc3):
        p = BarStatPlot(data3, Posthoc_Matrix=posthoc3)
        assert p.posthoc_matrix == posthoc3

    def test_colormap_default_flag(self, data2):
        p = BarStatPlot(data2)
        assert p.colormap_default is True

    def test_colormap_custom_flag(self, data2):
        p = BarStatPlot(data2, colormap=["red", "blue"])
        assert p.colormap_default is False

    def test_empty_data_fallback(self):
        """Empty groups should not crash — fallback to dummy data."""
        p = BarStatPlot([[], []])
        assert p.error is False

    def test_test_name_stored(self, data2):
        p = BarStatPlot(data2, Test_Name="t-test")
        assert p.testname == "t-test"

    def test_posthoc_name_stored(self, data2):
        p = BarStatPlot(data2, Posthoc_Tests_Name="Dunn")
        assert p.posthoc_name == "Dunn"

    def test_figure_scale_factor_stored(self, data2):
        p = BarStatPlot(data2, figure_scale_factor=2.0)
        assert p.figure_scale_factor == 2.0


# ─────────────────────────────────────────────
# 3. All plot classes — smoke tests
# ─────────────────────────────────────────────

@pytest.mark.parametrize("PlotClass", ALL_PLOT_CLASSES)
class TestAllPlotClassesSmoke:

    def test_instantiates_without_error(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03)
        assert p.error is False

    def test_plot_does_not_crash(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03)
        p.plot()

    def test_plot_with_3_groups(self, PlotClass, data3):
        p = PlotClass(data3, p_value_exact=0.001)
        p.plot()

    def test_plot_no_p_value(self, PlotClass, data2):
        p = PlotClass(data2)
        p.plot()

    def test_plot_with_groups_name(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03, Groups_Name=["A", "B"])
        p.plot()

    def test_plot_with_custom_colormap(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03, colormap=["red", "blue"])
        p.plot()

    def test_plot_with_scale_factor(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03, figure_scale_factor=1.5)
        p.plot()

    def test_plot_paired(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03, Paired_Test_Applied=True)
        p.plot()

    def test_plot_custom_figure_size(self, PlotClass, data2):
        p = PlotClass(data2, figure_w=8, figure_h=6)
        p.plot()

    def test_show_does_not_crash(self, PlotClass, data2):
        """show() with Agg backend should not raise."""
        p = PlotClass(data2, p_value_exact=0.03)
        p.plot()
        p.show()

    def test_close_does_not_crash(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03)
        p.plot()
        p.close()

    def test_save_creates_file(self, PlotClass, data2):
        p = PlotClass(data2, p_value_exact=0.03)
        p.plot()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            p.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
            p.close()


# ─────────────────────────────────────────────
# 4. Significance bars — p-value label combos
# ─────────────────────────────────────────────

class TestSignificanceBars:

    @pytest.mark.parametrize("print_p, print_s", [
        (True,  True),
        (True,  False),
        (False, True),
        (False, False),
    ])
    def test_label_combos_do_not_crash(self, data2, print_p, print_s):
        p = BarStatPlot(data2, p_value_exact=0.03,
                        print_p_label=print_p, print_stars=print_s)
        p.plot()

    def test_no_bars_when_no_p_and_no_posthoc(self, data2):
        """No significance bar drawn when p=None and no posthoc matrix."""
        p = BarStatPlot(data2)
        p.plot()  # should not crash

    def test_single_bar_2groups(self, data2):
        p = BarStatPlot(data2, p_value_exact=0.001)
        p.plot()

    def test_posthoc_3groups(self, data3, posthoc3):
        p = BarStatPlot(data3, Posthoc_Matrix=posthoc3,
                        Posthoc_Tests_Name="Dunn")
        p.plot()

    def test_posthoc_4groups(self, data4, posthoc4):
        p = BarStatPlot(data4, Posthoc_Matrix=posthoc4)
        p.plot()

    def test_posthoc_5groups(self, data5, posthoc5):
        p = BarStatPlot(data5, Posthoc_Matrix=posthoc5)
        p.plot()

    def test_posthoc_6groups_fallback_to_single_bar(self):
        """6-group posthoc has no hardcoded layout — falls back to one overall bar."""
        np.random.seed(0)
        data6 = [list(np.random.normal(i, 1, 15)) for i in range(6)]
        posthoc6 = [[1.0 if i == j else 0.05 for j in range(6)] for i in range(6)]
        p = BarStatPlot(data6, p_value_exact=0.001, Posthoc_Matrix=posthoc6)
        p.plot()


# ─────────────────────────────────────────────
# 5. Descriptive statistics — correctness
# ─────────────────────────────────────────────

class TestDescriptiveStats:

    def test_mean_correct_3groups(self, data3):
        p = BarStatPlot(data3)
        for i, group in enumerate(data3):
            assert abs(p.mean[i] - np.mean(group)) < 1e-10

    def test_median_correct_3groups(self, data3):
        p = BarStatPlot(data3)
        for i, group in enumerate(data3):
            assert abs(p.median[i] - np.median(group)) < 1e-10

    def test_sd_ddof1_correct(self, data3):
        p = BarStatPlot(data3)
        for i, group in enumerate(data3):
            assert abs(p.sd[i] - np.std(group, ddof=1)) < 1e-10

    def test_sem_formula_correct(self, data3):
        p = BarStatPlot(data3)
        for i, group in enumerate(data3):
            expected = np.std(group, ddof=1) / np.sqrt(len(group))
            assert abs(p.sem[i] - expected) < 1e-10

    def test_n_matches_group_lengths(self, data3):
        p = BarStatPlot(data3)
        assert p.n == [len(g) for g in data3]

    def test_y_max_is_global_max(self, data3):
        p = BarStatPlot(data3)
        expected = max(max(g) for g in data3)
        assert abs(p.y_max - expected) < 1e-10

    def test_unequal_group_sizes(self):
        np.random.seed(0)
        data = [list(np.random.normal(0, 1, 10)), list(np.random.normal(1, 1, 30))]
        p = BarStatPlot(data)
        assert p.n == [10, 30]
        assert abs(p.sem[0] - np.std(data[0], ddof=1) / np.sqrt(10)) < 1e-10
        assert abs(p.sem[1] - np.std(data[1], ddof=1) / np.sqrt(30)) < 1e-10


# ─────────────────────────────────────────────
# 6. save() — format options
# ─────────────────────────────────────────────

class TestSave:

    @pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
    def test_save_formats(self, data2, fmt):
        p = BarStatPlot(data2, p_value_exact=0.03)
        p.plot()
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            path = f.name
        try:
            p.save(path, format=fmt)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
            p.close()

    def test_save_custom_dpi(self, data2):
        p = BarStatPlot(data2)
        p.plot()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            p.save(path, dpi=300)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
            p.close()

    def test_error_state_skips_save(self):
        """When error=True, save() should do nothing (no crash, no file)."""
        p = object.__new__(BarStatPlot)
        p.error = True
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        os.unlink(path)
        p.save(path)
        assert not os.path.exists(path)

    def test_error_state_skips_show(self):
        p = object.__new__(BarStatPlot)
        p.error = True
        p.show()  # should not raise

    def test_error_state_skips_close(self):
        p = object.__new__(BarStatPlot)
        p.error = True
        p.close()  # should not raise


# ─────────────────────────────────────────────
# 7. BaseStatPlot.plot() raises NotImplementedError
# ─────────────────────────────────────────────

def test_base_plot_raises_not_implemented():
    b = object.__new__(BarStatPlot)
    b.error = False
    with pytest.raises(NotImplementedError):
        BaseStatPlot.plot(b)


# ─────────────────────────────────────────────
# 8. x-label / title / y-label rendering
# ─────────────────────────────────────────────

class TestLabelsAndTitles:

    def test_x_label(self, data2):
        p = BarStatPlot(data2, x_label="Time (s)")
        p.plot()

    def test_y_label(self, data2):
        p = BarStatPlot(data2, y_label="Amplitude")
        p.plot()

    def test_plot_title(self, data2):
        p = BarStatPlot(data2, plot_title="My Experiment")
        p.plot()

    def test_no_x_labels(self, data2):
        p = BarStatPlot(data2, print_x_labels=False)
        p.plot()

    def test_groups_name_cycling(self):
        np.random.seed(0)
        data4 = [list(np.random.normal(i, 1, 15)) for i in range(4)]
        p = BarStatPlot(data4, Groups_Name=["A", "B"])
        p.plot()  # should cycle A, B, A, B without crashing


# ─────────────────────────────────────────────
# 9. SwarmStatPlot_subgrouping_betta (experimental)
# ─────────────────────────────────────────────

class TestSwarmSubgrouping:

    def test_instantiates(self, data2):
        p = SwarmStatPlot_subgrouping_betta(data2)
        assert p.error is False

    def test_plot_does_not_crash(self, data2, capsys):
        p = SwarmStatPlot_subgrouping_betta(data2, subgrouping=[1, 2, 1, 2])
        p.plot()

    def test_no_subgrouping_falls_back(self, data2, capsys):
        p = SwarmStatPlot_subgrouping_betta(data2)
        p.plot()


# ─────────────────────────────────────────────
# 10. Large groups & edge values
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_large_groups(self):
        np.random.seed(0)
        data = [list(np.random.normal(i, 1, 300)) for i in range(2)]
        p = BarStatPlot(data, p_value_exact=0.001)
        p.plot()

    def test_single_value_per_group(self):
        """Minimum possible data — 1 point per group."""
        p = BarStatPlot([[5.0], [10.0]])
        assert p.n == [1, 1]
        assert p.mean == [5.0, 10.0]

    def test_very_small_p_value(self, data2):
        p = BarStatPlot(data2, p_value_exact=1e-10)
        assert p.p_printed == "p<0.0001"
        assert p.stars_printed == "****"

    def test_p_value_exactly_at_boundary(self, data2):
        p = BarStatPlot(data2, p_value_exact=0.05)
        assert p.make_stars(0.05) == 0
        assert p.stars_printed == "ns"

    def test_p_value_just_below_boundary(self, data2):
        p = BarStatPlot(data2, p_value_exact=0.049)
        assert p.make_stars(0.049) == 1

    def test_all_identical_values(self):
        """Zero-variance data should not crash."""
        data = [[5.0] * 10, [5.0] * 10]
        p = BarStatPlot(data)
        p.plot()

    def test_negative_values(self):
        data = [[-3.0, -2.0, -1.0], [-6.0, -5.0, -4.0]]
        p = BarStatPlot(data)
        assert p.y_max == pytest.approx(-1.0)

    def test_colormap_longer_than_groups(self, data2):
        """Providing more colors than groups should not crash."""
        p = BarStatPlot(data2, colormap=["red", "blue", "green", "purple"])
        p.plot()

    def test_invalid_colormap_all_entries(self, data2):
        """All invalid color entries should all fall back to 'k'."""
        p = BarStatPlot(data2, colormap=["notacolor", "alsoinvalid"])
        assert all(c == "k" for c in p.colors_edge)

    def test_many_groups(self):
        """8 groups — beyond posthoc layout hardcoding — falls back gracefully."""
        np.random.seed(0)
        data8 = [list(np.random.normal(i, 1, 10)) for i in range(8)]
        posthoc8 = [[1.0 if i == j else 0.05 for j in range(8)] for i in range(8)]
        p = BarStatPlot(data8, p_value_exact=0.001, Posthoc_Matrix=posthoc8)
        p.plot()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])