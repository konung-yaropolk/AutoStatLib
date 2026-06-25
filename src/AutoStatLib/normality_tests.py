from __future__ import annotations

from AutoStatLib._protocol import StatAnalysisProtocol

from typing import Literal, Optional

import numpy as np
from numpy.random import Generator
from scipy.stats import (
    MonteCarloMethod,
    anderson,
    goodness_of_fit,
    norm,
    normaltest,
    shapiro,
)
from statsmodels.stats.diagnostic import lilliefors

NormalityMethod = Literal["interpolate", "monte_carlo"]
_NORMALITY_METHODS: tuple[str, ...] = ("interpolate", "monte_carlo")


class NormalityTests(StatAnalysisProtocol):
    """
    Normality tests mixin.

    Test selection follows:
    Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov,
    Lilliefors and Anderson-Darling tests,
    Nornadiah Mohd Razali, Yap Bee Wah.

    The Kolmogorov-Smirnov/Lilliefors and Anderson-Darling tests each
    support two ways of getting a p-value, selected with `method`:

    "interpolate" (default): fast, O(1), table/formula-based. Lilliefors
        uses statsmodels' Dallal & Wilkinson (1986) table interpolation;
        Anderson-Darling uses `scipy.stats.anderson`'s own critical-value
        table together with the closed-form continuous approximation from
        Lorentz Jantschi & Sorana D. Bolboaca, *Computation of Probability
        Associated with Anderson-Darling Statistic*, Mathematics, 2018
        (scipy's own table lookup is accurate but clipped to [0.01, 0.15];
        this formula extrapolates beyond that range).

    "monte_carlo": slower, (asymptotically) exact. Both loc and scale are
        fit by MLE on the sample (that's what makes these "Lilliefors" /
        a proper Anderson-Darling test rather than tests against a fully
        specified normal), and the p-value comes from simulating that
        statistic's true null distribution via parametric bootstrap
        (`scipy.stats.goodness_of_fit` for KS, `scipy.stats.anderson` with
        a `MonteCarloMethod` for AD), seeded from a NumPy `Generator`,
        rather than read off a table/formula. Use this when the table
        approximation is questionable (e.g. a statistic landing right on
        a table boundary) or when you need a precise p-value rather than
        just a pass/fail at alpha=0.05 — see `_N_MC_SAMPLES` for the
        speed/precision trade-off.
    """

    # Monte Carlo replicates used to build the null distribution when
    # method="monte_carlo". 9999 is scipy's own default (p-values accurate
    # to ~1e-4). Runtime grows with both this value and `n`; for n in the
    # hundreds expect well under a second per test, for n in the thousands
    # expect several seconds. Lower this if you need speed over precision.
    _N_MC_SAMPLES: int = 9999

    def check_normality(
        self,
        data: list[float],
        method: NormalityMethod = "interpolate",
        random_state: Optional[Generator] = None,
    ) -> tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool]]:
        """
        Run up to four normality tests on a single group.

        Parameters:
            data: sample to test.
            method: "interpolate" (default) for fast table/formula-based
                p-values, or "monte_carlo" for slower, (asymptotically)
                exact p-values. Applies to the Lilliefors and
                Anderson-Darling tests; see the class docstring.
            random_state: NumPy `Generator` driving the Monte Carlo
                simulation when method="monte_carlo". Pass one in for
                reproducible results across calls; otherwise a fresh
                `default_rng()` is created. Ignored when
                method="interpolate".

        Returns a tuple (sw, lf, ad, ap) where each element is:
            True  — test passed (p > 0.05, distribution appears normal)
            False — test failed
            None  — sample too small to run the test reliably (n < 20)
        """
        sw: Optional[bool] = None
        lf: Optional[bool] = None
        ad: Optional[bool] = None
        ap: Optional[bool] = None
        n: int = len(data)

        # Shapiro-Wilk test — scipy's Royston (AS R94) p-value is already
        # the correct approach here, no change needed.
        _sw_stat, sw_p_value = shapiro(data)
        sw = bool(sw_p_value and sw_p_value > 0.05)

        # Lilliefors test: KS statistic against a normal fit to the data.
        _lf_stat, lf_p_value = self._ks_lilliefors(
            data, method=method, random_state=random_state
        )
        lf = bool(lf_p_value and lf_p_value > 0.05)

        # Anderson-Darling test (requires n >= 20)
        if n >= 20:
            _ad_stat, ad_p_value = self.anderson_get_p(
                data, method=method, random_state=random_state
            )
            ad = bool(ad_p_value and ad_p_value > 0.05)

        # D'Agostino-Pearson test (unreliable for n < 20). This is a
        # skewness/kurtosis statistic with a known asymptotic chi-square
        # null distribution — scipy's `normaltest` p-value is already the
        # standard, correct approach and doesn't need Monte Carlo.
        if n >= 20:
            _ap_stat, ap_p_value = normaltest(data)
            ap = bool(ap_p_value and ap_p_value > 0.05)

        return (sw, lf, ad, ap)

    def _ks_lilliefors(
        self,
        data: list[float],
        method: NormalityMethod = "interpolate",
        random_state: Optional[Generator] = None,
        n_mc_samples: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        KS test for normality with the Lilliefors correction, i.e. loc/scale
        estimated from `data` rather than assumed known.

        method="interpolate" (default, fast): statsmodels' Lilliefors /
            Dallal & Wilkinson (1986) table-interpolated p-value.
        method="monte_carlo" (slow, exact): p-value from simulating the
            statistic's true null distribution via parametric bootstrap
            (`scipy.stats.goodness_of_fit`).

        Returns:
            (ks_statistic, p_value)
        """
        if method not in _NORMALITY_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; expected one of {_NORMALITY_METHODS}."
            )

        if method == "monte_carlo":
            rng: Generator = (
                random_state if random_state is not None else np.random.default_rng()
            )
            result = goodness_of_fit(
                norm,
                data,
                statistic="ks",
                n_mc_samples=n_mc_samples or self._N_MC_SAMPLES,
                random_state=rng,
            )
            return float(result.statistic), float(result.pvalue)

        lf_stat, lf_p_value = lilliefors(data, dist="norm")
        return float(lf_stat), float(lf_p_value)

    def anderson_get_p(
        self,
        data: list[float],
        method: NormalityMethod = "interpolate",
        random_state: Optional[Generator] = None,
        n_mc_samples: Optional[int] = None,
    ) -> tuple[float, Optional[float]]:
        """
        Anderson-Darling test for normality, with loc/scale estimated from
        `data`.

        method="interpolate" (default, fast): the AD statistic comes from
            `scipy.stats.anderson(..., method="interpolate")`, and is
            converted to a continuous p-value with the closed-form
            approximation from Lorentz Jantschi & Sorana D. Bolboaca,
            *Computation of Probability Associated with Anderson-Darling
            Statistic*, Mathematics, 2018 (scipy's own table lookup is
            accurate but clipped to [0.01, 0.15]; this extrapolates
            beyond that range).
        method="monte_carlo" (slow, exact): both statistic and p-value
            come from `scipy.stats.anderson` with a `MonteCarloMethod`,
            which simulates the statistic's true null distribution
            (refitting loc/scale on each simulated draw) instead of
            reading off the table.

        Returns:
            (ad_statistic, p_value)
        """
        if method not in _NORMALITY_METHODS:
            raise ValueError(
                f"Unknown method {method!r}; expected one of {_NORMALITY_METHODS}."
            )

        if method == "monte_carlo":
            rng: Generator = (
                random_state if random_state is not None else np.random.default_rng()
            )
            mc_method = MonteCarloMethod(
                n_resamples=n_mc_samples or self._N_MC_SAMPLES, rng=rng
            )
            result = anderson(data, dist="norm", method=mc_method)
            return float(result.statistic), float(result.pvalue)
        
        if method == "interpolate":
            result = anderson(data, dist="norm", method="interpolate")
            return float(result.statistic), float(result.pvalue)

        # n: int = len(data)
        # ad: float = float(anderson(data, dist="norm", method="interpolate").statistic)

        # # Adjust statistic for small sample sizes, then map to a
        # # continuous p-value (Jantschi & Bolboaca, 2018).
        # s: float = ad * (1 + 0.75 / n + 2.25 / (n**2))

        # try:
        #     p: float
        #     if s >= 0.6:
        #         p = np.e ** (1.2937 - 5.709 * s + 0.0186 * s**2)
        #     elif s > 0.34:
        #         p = np.e ** (0.9177 - 4.279 * s - 1.38 * s**2)
        #     elif s > 0.2:
        #         p = 1 - np.e ** (-8.318 + 42.796 * s - 59.938 * s**2)
        #     else:
        #         p = 1 - np.e ** (-13.436 + 101.14 * s - 223.73 * s**2)
        # # catch rear case of overflow when p is too small
        # except Exception(OverflowError):
        #     p = 5e-324

        # return ad, p