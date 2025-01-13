# statlib - a simple statistical analysis library

[![pypi_version](https://img.shields.io/pypi/v/statlib?label=PyPI&color=green)](https://pypi.org/project/statlib)
[![GitHub Release](https://img.shields.io/github/v/release/konung-yaropolk/statlib?label=GitHub&color=green&link=https%3A%2F%2Fgithub.com%2Fkonung-yaropolk%2FDiaModality)](https://github.com/konung-yaropolk/statlib)
[![PyPI - License](https://img.shields.io/pypi/l/statlib)](https://pypi.org/project/statlib)
[![Python](https://img.shields.io/badge/Python-v3.10%5E-green?logo=python)](https://pypi.org/project/statlib)  
[![PyPI - Downloads](https://img.shields.io/pypi/dm/statlib?label=PyPI%20stats&color=blue)](https://pypi.org/project/statlib)


### To install run the code:
```bash
pip install statlib
```


### Example of use:
See the /demo directory on Git repo or  
use the following example:


```python
import numpy as np
import statlib

# generate random normal data:
groups = 2
n = 30
data = [list(np.random.normal(.5*i + 4, abs(1-.2*i), n))
        for i in range(groups)]


# set the parameters:
paired = False     # is groups dependend or not
tails = 2          # two-tailed or one-tailed result
popmean = 0        # population mean - only for single-sample tests needed

# initiate the analysis
analysis = statlib.StatisticalAnalysis(
    data, paired=paired, tails=tails, popmean=popmean)
```

now you can preform automatically statistical test selection:
```python
analysis.RunAuto()
```

ih the other hand you can reform scecific tests:
```python
# 2 groups independend:
analysis.RunTtest()
analysis.RunMannWhitney()

# 2 groups paired"
analysis.RunTtestPaired()
analysis.RunWilcoxon()

# 3 and more indepennded groups comparison:
analysis.RunAnova()
analysis.RunKruskalWallis()

# 3 and more paired groups comparison:
analysis.RunFriedman()

# single group tests"
analysis.RunTtestSingleSample()
analysis.RunWilcoxonSingleSample()
```

Summary will be printed to the console.
Results are accessible as a dictionary via GetResult() method:
```python
results = analysis.GetResult()
```

The results dictionary keys with representing value types:
```
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
```







## Pre-Alpha dev status.

### TODO:
*

-Kruskal-Wallis test - add Dunn's multiple comparisons
-Anova: add 2-way anova and 3-way(?)

check:
-Wilcoxon signed-rank test and Mann-whitney - check mechanism of one-tailed calc, looks like it works wrong


checked tests:
-Wilcoxon 2 tail - ok
-Mann-whitney 2 tail - ok

*

