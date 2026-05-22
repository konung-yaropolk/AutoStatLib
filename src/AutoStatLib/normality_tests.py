from __future__ import annotations

from AutoStatLib._protocol import StatAnalysisProtocol

from typing import Optional

import numpy as np
from scipy.stats import shapiro, normaltest, anderson
from statsmodels.stats.diagnostic import lilliefors


class NormalityTests(StatAnalysisProtocol):
    """
    Normality tests mixin.

    See the article about minimal sample size for tests:
    Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov,
    Lilliefors and Anderson-Darling tests,
    Nornadiah Mohd Razali, Yap Bee Wah.
    """

    def check_normality(
        self, data: list[float]
    ) -> tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool]]:
        """
        Run up to four normality tests on a single group.

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

        # Shapiro-Wilk test
        _sw_stat, sw_p_value = shapiro(data)
        sw = bool(sw_p_value and sw_p_value > 0.05)

        # Lilliefors test
        _lf_stat, lf_p_value = lilliefors(data, dist="norm")
        lf = bool(lf_p_value and lf_p_value > 0.05)

        # Anderson-Darling test (requires n >= 20)
        if n >= 20:
            _ad_stat, ad_p_value = self.anderson_get_p(data, dist="norm")
            ad = bool(ad_p_value and ad_p_value > 0.05)

        # D'Agostino-Pearson test (unreliable for n < 20)
        if n >= 20:
            _ap_stat, ap_p_value = normaltest(data)
            ap = bool(ap_p_value and ap_p_value > 0.05)

        return (sw, lf, ad, ap)

    def anderson_get_p(
        self, data: list[float], dist: str = "norm"
    ) -> tuple[float, Optional[float]]:
        """
        Calculate the p-value for the Anderson-Darling test using the method described in:
        *Computation of Probability Associated with Anderson-Darling Statistic*,
        Lorentz Jantschi and Sorana D. Bolboaca, Mathematics, 2018.

        Returns:
            (ad_statistic, p_value)
        """
        e: float = 2.718281828459045
        n: int = len(data)

        ad, _critical_values, _significance_levels = anderson(data, dist=dist)

        # Adjust statistic for small sample sizes
        s: float = ad * (1 + 0.75 / n + 2.25 / (n**2))

        p: Optional[float]
        if s >= 0.6:
            p = e ** (1.2937 - 5.709 * s + 0.0186 * s**2)
        elif s > 0.34:
            p = e ** (0.9177 - 4.279 * s - 1.38 * s**2)
        elif s > 0.2:
            p = 1 - e ** (-8.318 + 42.796 * s - 59.938 * s**2)
        elif s <= 0.2:
            p = 1 - e ** (-13.436 + 101.14 * s - 223.73 * s**2)
        else:
            p = None

        return ad, p
