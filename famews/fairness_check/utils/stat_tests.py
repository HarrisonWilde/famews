from typing import Union

import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats._mannwhitneyu import MannwhitneyuResult
from scipy.stats.contingency import chi2_contingency


def run_mann_whitney_u(
    df: pd.DataFrame,
    var: str,
    hue: str,
    group1: Union[str, bool, int],
    group2: Union[str, bool, int],
    hyp="two-sided",
) -> MannwhitneyuResult:
    """Run Mann-Whitney U test. Compare the distribution of var in group 1 with the distribution of var in group 2.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the distribution values
    var : str
        name of variable whose distribution is compared
    hue : str
        column name in df where the groups are defined
    group1 : Union[str, bool, int]
        name of first group
    group2 : Union[str, bool, int]
        name of second group
    hyp : str, optional
        Test hypothesis, by default "two-sided"

    Returns
    -------
    MannwhitneyuResult
        Result of statistical test
    """

    data1 = df[df[hue] == group1][var]
    data2 = df[df[hue] == group2][var]
    return mannwhitneyu(data1, data2, alternative=hyp)


def run_chi_square_independence_test(
    df: pd.DataFrame, var1: str, var2: str, filter1: list = None, filter2: list = None
):
    """Run Chi-Square test of independence.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the distribution values
    var1: str
        First categorical variable
    var2 : str
        Second categorical variable
    filter1: list, optional
        List of categories of var1 to include, by default None.
    filter2: list, optional
        List of categories of var2 to include, by default None.
    """
    if filter1:
        df = df[df[var1].isin(filter1)]
    if filter2:
        df = df[df[var2].isin(filter2)]
    table = pd.crosstab(df[var1], df[var2])
    return chi2_contingency(table)
