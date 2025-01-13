# AutoStatLib - automated statistical analysis library

[![pypi_version](https://img.shields.io/pypi/v/AutoStatLib?label=PyPI&color=green)](https://pypi.org/project/AutoStatLib)
[![GitHub Release](https://img.shields.io/github/v/release/konung-yaropolk/AutoStatLib?label=GitHub&color=green&link=https%3A%2F%2Fgithub.com%2Fkonung-yaropolk%2FDiaModality)](https://github.com/konung-yaropolk/AutoStatLib)
[![PyPI - License](https://img.shields.io/pypi/l/AutoStatLib)](https://pypi.org/project/AutoStatLib)
[![Python](https://img.shields.io/badge/Python-v3.10%5E-green?logo=python)](https://pypi.org/project/AutoStatLib)  
[![PyPI - Downloads](https://img.shields.io/pypi/dm/AutoStatLib?label=PyPI%20stats&color=blue)](https://pypi.org/project/AutoStatLib)


### To install run the command:
```bash
pip install autostatlib
```


### Example use case:
See the /demo directory on Git repo or  
use the following example:


```python
import numpy as np
import AutoStatLib

# generate random data:
groups = 2
n = 30

# normal data
data_norm = [list(np.random.normal(.5*i + 4, abs(1-.2*i), n))
        for i in range(groups)]

# non-normal data
data_uniform = [list(np.random.uniform(i+3, i+1, n)) for i in range(groups)]


# set the parameters:
paired = False     # is groups dependend or not
tails = 2          # two-tailed or one-tailed result
popmean = 0        # population mean - only for single-sample tests needed

# initiate the analysis
analysis = AutoStatLib.StatisticalAnalysis(
    data_norm, paired=paired, tails=tails, popmean=popmean)
```

now you can preform automated statistical test selection:
```python
analysis.RunAuto()
```

or you can choose specific tests:
```python
# 2 groups independend:
analysis.RunTtest()
analysis.RunMannWhitney()

# 2 groups paired"
analysis.RunTtestPaired()
analysis.RunWilcoxon()

# 3 and more independed groups comparison:
analysis.RunAnova()
analysis.RunKruskalWallis()

# 3 and more depended groups comparison:
analysis.RunFriedman()

# single group tests"
analysis.RunTtestSingleSample()
analysis.RunWilcoxonSingleSample()
```

Test summary will be printed to the console.
You can also get it as a python string via GetSummary() method.

---
Test results are accessible as a dictionary via GetResult() method:
```python
results = analysis.GetResult()
```

The results dictionary keys with representing value types:
```
{
    'p-value':                     String
    'Significance(p<0.05)':        Boolean
    'Stars_Printed':               String
    'Test_Name':                   String
    'Groups_Compared':             Integer
    'Population_Mean':             Float   (taken from the input)
    'Data_Normaly_Distributed':    Boolean
    'Parametric_Test_Applied':     Boolean
    'Paired_Test_Applied':         Boolean
    'Tails':                       Integer (taken from the input)
    'p-value_exact':               Float
    'Stars':                       Integer
    'Warnings':                    String
    'Groups_N':                    List of integers
    'Groups_Median':               List of floats
    'Groups_Mean':                 List of floats
    'Groups_SD':                   List of floats
    'Groups_SE':                   List of floats
    'Samples':                     List of input values by groups
                                           (taken from the input)
}
```






---
## Pre-Alpha dev status.

### TODO:

--Kruskal-Wallis test - add Dunn's multiple comparisons
--Anova: add 2-way anova and 3-way(?)

check:
--Wilcoxon signed-rank test and Mann-whitney - check mechanism of one-tailed calc, looks like it works wrong


checked tests:
--Wilcoxon 2 tail - ok
--Mann-whitney 2 tail - ok



