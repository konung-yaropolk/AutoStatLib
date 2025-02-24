import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp

def kruskal_wallis_dunn(data_groups):
    # Perform the Kruskal-Wallis test
    h_stat, p_val = stats.kruskal(*data_groups)
    print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_val:.4f}")
    
    # Perform Dunn's multiple comparisons if Kruskal-Wallis is significant
    if p_val < 0.05:
        data_flat = np.concatenate(data_groups)
        groups = np.concatenate([[i] * len(group) for i, group in enumerate(data_groups)])
        # Pairwise post-hoc comparisons using Dunn's test
        dunn_result = sp.posthoc_dunn(data_groups, p_adjust='bonferroni')
        print("\nDunn's multiple comparisons result:")
        print(dunn_result)
    else:
        print("Kruskal-Wallis test is not significant; no need for multiple comparisons.")

# Sample data for each group
group1 = np.random.normal(loc=10, scale=2, size=20)
group2 = np.random.normal(loc=12, scale=2, size=20)
group3 = np.random.normal(loc=14, scale=2, size=20)
group4 = np.random.normal(loc=16, scale=2, size=20)
group5 = np.random.normal(loc=18, scale=2, size=20)

data_groups = [group1, group2, group3, group4, group5]

# Run the test
kruskal_wallis_dunn(data_groups)
