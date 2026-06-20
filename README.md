# AutoStatLib - python library for automated statistical analysis

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/AutoStatLib/tests.yml?label=Tests)](https://github.com/konung-yaropolk/AutoStatLib/actions/workflows/tests.yml)
[![pypi_version](https://img.shields.io/pypi/v/AutoStatLib?label=PyPI&color=green)](https://pypi.org/project/AutoStatLib)
[![GitHub Release](https://img.shields.io/github/v/release/konung-yaropolk/AutoStatLib?label=GitHub&color=green&link=https%3A%2F%2Fgithub.com%2Fkonung-yaropolk%2FAutoStatLib)](https://github.com/konung-yaropolk/AutoStatLib)
[![PyPI - License](https://img.shields.io/pypi/l/AutoStatLib)](https://pypi.org/project/AutoStatLib)
[![Python](https://img.shields.io/badge/Python-v3.10%5E-green?logo=python)](https://pypi.org/project/AutoStatLib)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/AutoStatLib?label=PyPI%20stats&color=blue)](https://pypi.org/project/AutoStatLib)

AutoStatLib picks the right statistical test for your data and runs it — automatically, or manually if you prefer to choose. It checks normality, decides between parametric and non-parametric tests, accounts for paired vs. independent groups, and (optionally) runs post-hoc pairwise comparisons. Results come back as a plain Python dictionary, and a companion plotting module turns that same dictionary straight into a publication-style figure.

## Table of contents

- [Install](#install)
- [Quick start](#quick-start)
- [How the API works](#how-the-api-works)
  - [1. Creating an analysis](#1-creating-an-analysis)
  - [2. Running a test](#2-running-a-test)
  - [3. How automatic test selection works](#3-how-automatic-test-selection-works)
  - [4. Reading the results](#4-reading-the-results)
  - [5. Warnings](#5-warnings)
  - [6. Text summary](#6-text-summary)
- [Plotting results](#plotting-results)


---

## Install

```bash
pip install AutoStatLib
```

Requires Python ≥ 3.10.

---

## Quick start

```python
import numpy as np
import AutoStatLib

# two independent, normally-distributed groups
group_A = list(np.random.normal(loc=4.0, scale=1.0, size=30))
group_B = list(np.random.normal(loc=6.5, scale=2.0, size=30))

analysis = AutoStatLib.StatisticalAnalysis([group_A, group_B],
                                            groups_name=['Control', 'Drug'],
                                            paired=False, 
                                            verbose=False,
                                            posthoc=True)

analysis.RunAuto()
result = analysis.GetResult()

print(f"{result['Test_Name']} \np-value: {result['p_value']} \nSignificance: {result['Significance(p<0.05)']}")

fig = AutoStatLib.StatPlots.BarStatPlot(result['Samples'], **result)
fig.plot()
fig.show()
fig.save('result.png')

```

More runnable examples live in the [`/demo`](./demo) directory of this repo.

---

## How the API works

The library is built around one class, **`StatisticalAnalysis`**. You construct it with your data and configuration, call one of its `Run*()` methods to actually perform a test, then pull the outcome out with `GetResult()`.

```
StatisticalAnalysis(data, ...)  →  .RunAuto() / .RunTtest() / ...  →  .GetResult()
                                                                   →  .GetSummary()
```

### 1. Creating an analysis

```python
analysis = AutoStatLib.StatisticalAnalysis(
    groups_list,
    paired=False,
    tails=2,
    posthoc=True,
    verbose=True,
    raise_errors=False,
    groups_name=None,
    subgrouping=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups_list` | `list[list[float]]` | — | Your data: a list of groups, each group a list of numbers. 1 group → single-sample test, 2 groups → two-sample test, 3+ groups → multi-group test. Non-numeric values inside a group (strings, `None`, etc.) are silently dropped. Each group needs at least 4 valid values. |
| `paired` | `bool` | `False` | Set `True` if the groups are dependent / repeated-measures (e.g. before–after on the same subjects). Determines whether paired tests (t-test paired, Wilcoxon, repeated-measures ANOVA, Friedman) or independent tests are eligible. |
| `tails` | `int` (`1` or `2`) | `2` | One-tailed or two-tailed p-value. |
| `popmean` | `float \| None` | `None` | Population mean to compare against — only used for single-sample tests (`RunTtestSingleSample`, `RunWilcoxonSingleSample`). If left `None` for a single-sample test, AutoStatLib defaults to `0` and adds a warning to the result. |
| `posthoc` | `bool` | `False` | If `True` and you run a 3+ group test, a post-hoc pairwise comparison matrix is computed (Tukey's HSD after ANOVA, Dunn's test after Kruskal-Wallis). |
| `verbose` | `bool` | `True` | If `True`, a full human-readable analysis log (data table, normality results, chosen test, results table) is printed to stdout every time you run a test. Set `False` to run silently and just use `GetResult()` / `GetSummary()`. |
| `raise_errors` | `bool` | `False` | If `True`, invalid input (e.g. wrong number of groups for the chosen test, unequal lengths for a paired test) raises a `ValueError` instead of printing a message and returning an empty/error result. |
| `groups_name` | `list[str] \| None` | `None` | Labels for your groups, used in the printed summary and available for plotting. If shorter than the number of groups it is cycled (`['A', 'B']` over 4 groups → `A, B, A, B`). Defaults to `Group 1`, `Group 2`, … |
| `subgrouping` | `list \| None` | `None` | Optional metadata passed through unchanged into the result dict and the plotting module, for experiments with within-group categories (sex, batch, etc.). |

### 2. Running a test

Call **`RunAuto()`** to let AutoStatLib decide, or call one of the manual methods if you already know which test you want:

| Method | Test performed | Groups required |
|---|---|---|
| `RunAuto()` | Automatically selects the test below based on normality, `paired`, and number of groups | any |
| `RunTtest()` | Independent-samples t-test | 2, independent |
| `RunMannWhitney()` | Mann-Whitney U test | 2, independent |
| `RunTtestPaired()` | Paired-samples t-test | 2, paired |
| `RunWilcoxon()` | Wilcoxon signed-rank test | 2, paired |
| `RunOnewayAnova()` | Ordinary one-way ANOVA (+ Tukey's post-hoc if `posthoc=True`) | 3+, independent |
| `RunKruskalWallis()` | Kruskal-Wallis test (+ Dunn's post-hoc if `posthoc=True`) | 3+, independent |
| `RunOnewayAnovaRM()` | Repeated-measures one-way ANOVA | 3+, paired |
| `RunFriedman()` | Friedman test | 3+, paired |
| `RunTtestSingleSample()` | Single-sample t-test against `popmean` | 1 |
| `RunWilcoxonSingleSample()` | Wilcoxon signed-rank test against `popmean` | 1 |
| `RunManual(test_id)` | Run any test above by its string ID | — |

`GetTestIDs()` returns the list of valid IDs for `RunManual()`, e.g.:

```python
analysis.GetTestIDs()
# ['anova_1w_ordinary', 'anova_1w_rm', 'friedman', 'kruskal_wallis',
#  'mann_whitney', 't_test_independent', 't_test_paired',
#  't_test_single_sample', 'wilcoxon', 'wilcoxon_single_sample']

analysis.RunManual('mann_whitney')   # equivalent to analysis.RunMannWhitney()
```

Calling a manual method on data it isn't suited for (e.g. a parametric test on non-normal data, or a 2-group test on 3 groups) doesn't crash — it either runs anyway and adds a warning to the result (wrong distribution / wrong pairing), or fails validation and returns an error result (wrong number of groups, mismatched paired lengths). See [Warnings](#5-warnings) below.

### 3. How automatic test selection works

`RunAuto()` follows this decision logic:

1. **Check normality of every group.** Each group is run through up to four normality tests — Shapiro-Wilk, Lilliefors, Anderson-Darling, and D'Agostino-Pearson (the latter two are skipped for groups smaller than n=20, where they're unreliable). A group is treated as **normally distributed if at least one test says so** — this is intentionally lenient, since requiring unanimous agreement makes the parametric branch too hard to reach with realistic sample sizes.
2. **Data is "parametric" only if every group passed.** One non-normal group is enough to route everything to the non-parametric branch.
3. **Branch by group count and `paired`:**

| Groups | Paired | Parametric | Test |
|---|---|---|---|
| 1 | — | yes | Single-sample t-test |
| 1 | — | no | Wilcoxon signed-rank (single sample) |
| 2 | No | yes | Independent t-test |
| 2 | No | no | Mann-Whitney U |
| 2 | Yes | yes | Paired t-test |
| 2 | Yes | no | Wilcoxon signed-rank |
| 3+ | No | yes | One-way ANOVA |
| 3+ | No | no | Kruskal-Wallis |
| 3+ | Yes | yes | Repeated-measures ANOVA |
| 3+ | Yes | no | Friedman |

The full normality breakdown (which test passed/failed for each group) is always printed in the verbose log, and the overall `True`/`False` verdict is available in the result dict as `Data_Normaly_Distributed`.

> **Note on one-tailed tests:** when `tails=1`, AutoStatLib currently halves the two-tailed p-value. This is only strictly correct when the observed effect is in your hypothesized direction — if you need directional certainty, treat one-tailed results with that caveat in mind, or verify the effect direction yourself before reporting.

### 4. Reading the results

```python
result = analysis.GetResult()
```

`GetResult()` returns:
- a **populated `dict`** if a test ran successfully,
- `{}` (empty dict) if the test failed validation (and prints an explanatory message),
- `None` if you haven't called any `Run*()` method yet.

Full key reference:

```python
{
    'p_value':                      str,                 # human-formatted, e.g. "p=0.03" or "p<0.0001"
    'p_value_exact':                float,               # raw numeric p-value
    'Significance(p<0.05)':         bool,
    'Stars':                        int,                 # 0-4, number of significance stars
    'Stars_Printed':                str,                 # 'ns', '*', '**', '***', '****'
    'Test_Name':                    str,                 # e.g. "t-test for independent samples"
    'Groups_Compared':              int,
    'Population_Mean':              float | str,         # value of `popmean`, or 'N/A' for multi-group tests
    'Data_Normaly_Distributed':     bool,
    'Parametric_Test_Applied':      bool,
    'Paired_Test_Applied':          bool,
    'Tails':                        int,                 # 1 or 2, echoes the input

    'Groups_Name':                  list[str],
    'Groups_N':                     list[int],
    'Groups_Mean':                  list[float],
    'Groups_Median':                list[float],
    'Groups_SD':                    list[float],         # sample SD, ddof=1
    'Groups_SE':                    list[float],         # standard error of the mean
    'Samples':                      list[list[float]],   # your cleaned input data, by group

    'Posthoc_Tests_Name':           str,                 # '' if posthoc=False
    'Posthoc_Matrix':               list[list[float]],   # pairwise p-values, [] if posthoc=False
    'Posthoc_Matrix_bool':          list[list[bool]],    # pairwise significance
    'Posthoc_Matrix_printed':       list[list[str]],     # human-formatted p-values
    'Posthoc_Matrix_stars':         list[list[str]],     # star notation per pair

    'Warnings':                     list[str],           # see below
    'Successfull_Test':             bool,
    'subgrouping':                  list,                # echoes the constructor input
}
```

If `posthoc=True` was set and you ran a 3+ group test, `Posthoc_Matrix` is an `N×N` matrix (where `N` is the number of groups) of pairwise comparison p-values, with `1.0` on the diagonal.

### 5. Warnings

`result['Warnings']` is a list of human-readable strings describing anything questionable about the analysis — it's empty (`[]`) for a clean run. Warnings are added when:

- A parametric test was manually run on data that isn't normally distributed (or vice versa)
- A paired test was manually run on data declared independent (or vice versa)
- A single-sample test was run without `popmean` set (defaults to `0`)

These don't stop the test from running — they flag results that might be misleading so you can decide whether to trust them or re-run with the correct configuration / `RunAuto()`.

### 6. Text summary

For a copy-paste-ready writeup of the whole analysis (data table, normality breakdown, test chosen, full results):

```python
analysis.PrintSummary()        # prints directly to stdout
text = analysis.GetSummary()   # same content, as a string
```

If `verbose=True` (the default), this same summary is printed automatically every time you call a `Run*()` method.

---

## Plotting results

The `AutoStatLib.StatPlots` module turns a finished analysis directly into a matplotlib figure. Each plot class takes your raw data plus (optionally) the exact dictionary returned by `GetResult()`, unpacked as keyword arguments — so the test name, p-value, and post-hoc matrix flow straight from the analysis into the plot without you re-typing anything.

```python
from AutoStatLib.StatPlots import BarStatPlot

result = analysis.GetResult()

fig = BarStatPlot(result['Samples'], **result)
fig.plot()
fig.show()                       # or fig.save('figure.png')
```

Available plot types — all share the same constructor signature and workflow:

| Class | Plot style |
|---|---|
| `BarStatPlot` | Bar chart of group means with SD error bars and swarm overlay |
| `ViolinStatPlot` | Violin plot with SD error bars and swarm overlay |
| `BoxStatPlot` | Box-and-whisker plot with swarm overlay |
| `ScatterStatPlot` | Mean/median markers with jittered scatter points (lines connect paired data) |
| `SwarmStatPlot` | Mean/median markers with a non-overlapping swarm plot |
| `SwarmStatPlot_subgrouping_betta` | Swarm plot with per-subgroup coloring (experimental) |

You don't need a `StatisticalAnalysis` result to use these — you can also build a plot directly from raw data and your own values:

```python
from AutoStatLib.StatPlots import BoxStatPlot

fig = BoxStatPlot(
    [group_A, group_B, group_C],
    p_value_exact=0.012,
    Test_Name='Kruskal-Wallis test',
    Groups_Name=['Control', 'Drug A', 'Drug B'],
    plot_title='Effect on firing rate',
    y_label='Hz',
)
fig.plot()
fig.save('result.png', dpi=300)
```

Useful customization parameters (all optional):

| Parameter | Description |
|---|---|
| `plot_title`, `x_label`, `y_label` | Figure text |
| `print_x_labels` | Show/hide group labels on the x-axis |
| `print_p_label`, `print_stars` | Toggle p-value text and/or star notation on significance bars |
| `colormap` | List of colors, one per group (falls back to a default palette) |
| `figure_scale_factor`, `figure_h`, `figure_w` | Figure sizing |

Output methods on every plot object: `.plot()` to render, `.show()` to display, `.save(path, format='png', dpi=150)` to export, `.close()` to free the figure.

---

## TODO

The project is in alpha dev status. Here is much work to do:

-- Anova: posthocs  
-- Anova: add 2-way anova and 3-way anova  
-- onevay Anova: add repeated measures (for normal dependent values) with and without Gaisser-Greenhouse correction  
-- onevay Anova: add Brown-Forsithe and Welch (for normal independent values with unequal SDs between groups)  
-- paired T-test: add ratio-paired t-test (ratios of paired values are consistent)  
-- add Welch test (for norm data unequal variances)  
-- add Kolmogorov-smirnov test (unpaired nonparametric 2 sample, compare cumulative distributions)  
-- add independent t-test with Welch correction (do not assume equal SDs in groups)  
-- add correlation test, correlation diagram  
-- add linear regression, regression diagram  
-- add QQ plot  
-- n-sample tests: add onetail option  

✅ done -- detailed normality test results  
✅ done -- added posthoc: Kruskal-Wallis Dunn's multiple comparisons  

tests check:  
1-sample:  
✅ok --Wilcoxon 2,1 tails  
✅ok --t-tests 2,1 tails  

2-sample:  
✅ok --Wilcoxon 2,1 tails  
✅ok --Mann-whitney 2,1 tails  
✅ok --t-tests 2,1 tails  

n-sample:  
✅ok --Kruskal-Wallis 2 tail  
✅ok --Dunn's multiple comparisons  
✅ok --Friedman 2 tail  
✅ok --one-way ANOVA 2-tailed  
✅ok --Tukey`s multiple comparisons  


---
