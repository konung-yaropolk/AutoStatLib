import numpy as np
import AutoStatLib


# Example usage:

# %%# generate random normal data:
groups = 3
n = 20
data = [list(np.random.normal(.5*i + 4, abs(1-.2*i), n))
        for i in range(groups)]

# %%# generate random non-normal data:
# groups = 2
# n = 30
# data = [list(np.random.uniform(i+3, i+1, n)) for i in range(groups)]


# %%# set the parameters:
paired = False     # is groups dependent or not
tails = 2          # two-tailed or one-tailed result
popmean = 0        # population mean - only for single-sample tests needed

# %%# initiate the analysis
analysis = AutoStatLib.StatisticalAnalysis(
    data, paired=paired, tails=tails, popmean=popmean, posthoc=True)

# %%# Preform auto-selected test
# analysis.RunAuto()


# %%# Preform specific tests:

# # 2 groups independent:
# analysis.RunTtest()
# analysis.RunMannWhitney()

# # 2 groups paired"
# analysis.RunTtestPaired()
# analysis.RunWilcoxon()

# # 3 and more independed groups comparison:
analysis.RunOnewayAnova()
# analysis.RunKruskalWallis()

# # 3 and more depended groups comparison:
# analysis.RunOnewayAnovaRM()
# analysis.RunFriedman()

# # single group tests"
# analysis.RunTtestSingleSample()
# analysis.RunWilcoxonSingleSample()


# %%# Get the results dictionary for future processing
results = analysis.GetResult()


plot = AutoStatLib.StatPlots.BarStatPlot(results['Samples'], **results
                                         #    dependent=dependent,
                                         #    y_label=y_label,
                                         #    x_manual_tick_labels=x_manual_tick_labels,
                                         )

plot.plot()
plot.show()

# %%
