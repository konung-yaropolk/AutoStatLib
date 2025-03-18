from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro, normaltest, anderson


class NormalityTests():
    '''
        Normality tests mixin

        see the article about minimal sample size for tests:
        Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov,
        Lilliefors and Anderson-Darling tests, Nornadiah Mohd Razali1, Yap Bee Wah1
    '''

    def check_normality(self, data):
        sw = None
        lf = None
        ad = None
        ap = None
        n = len(data)

        # Shapiro-Wilk test
        sw_stat, sw_p_value = shapiro(data)
        if sw_p_value and sw_p_value > 0.05:
            sw = True
        else:
            sw = False

        # Lilliefors test
        lf_stat, lf_p_value = lilliefors(
            data, dist='norm')
        if lf_p_value and lf_p_value > 0.05:
            lf = True
        else:
            lf = False

        # Anderson-Darling test
        if n >= 20:
            ad_stat, ad_p_value = self.anderson_get_p(
                data, dist='norm')
            if ad_p_value and ad_p_value > 0.05:
                ad = True
            else:
                ad = False

        # D'Agostino-Pearson test
        # test result is skewed if n<20
        if n >= 20:
            ap_stat, ap_p_value = normaltest(data)
            if ap_p_value and ap_p_value > 0.05:
                ap = True
            else:
                ap = False

        return (sw, lf, ad, ap)

    def anderson_get_p(self, data, dist='norm'):
        '''
            calculating p-value for Anderson-Darling test using the method described here:
            Computation of Probability Associated with Anderson-Darling Statistic
            Lorentz Jantschi and Sorana D. Bolboaca, 2018 - Mathematics

        '''
        e = 2.718281828459045
        n = len(data)

        ad, critical_values, significance_levels = anderson(
            data, dist=dist)

        # adjust ad_stat for small sample sizes:
        s = ad*(1 + 0.75/n + 2.25/(n**2))

        if s >= 0.6:
            p = e**(1.2937 - 5.709*s + 0.0186*s**2)
        elif s > 0.34:
            p = e**(0.9177 - 4.279*s - 1.38*s**2)
        elif s > 0.2:
            p = 1 - e**(-8.318 + 42.796*s - 59.938*s**2)
        elif s <= 0.2:
            p = 1 - e**(-13.436 + 101.14*s - 223.73*s**2)
        else:
            p = None

        return ad, p
