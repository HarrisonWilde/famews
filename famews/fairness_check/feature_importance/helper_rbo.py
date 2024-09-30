from typing import Dict, List, Tuple

from famews.fairness_check.feature_importance.rbo import RankingSimilarity


def get_critical_rbo_value(
    feat_ranking_all: List[str],
    feat_ranking_random: Dict[str, Dict[str, List[str]]],
    k: int = 20,
) -> float:
    """Compute the critical RBO value from synthetic feature rankings.

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_random : Dict[str, Dict[str, List[str]]]
        Map of synthetic feature rankings
    k : int, optional
        Size of the ranking to focus on, by default 20

    Returns
    -------
    float
        Critical RBO value
    """
    critical_rbo = 1
    for dict_feat_ranking in feat_ranking_random.values():
        for random_rk in dict_feat_ranking.values():
            rbo_random = RankingSimilarity(feat_ranking_all, random_rk).rbo(
                p=0.935, k=k
            )
            if rbo_random < critical_rbo:
                critical_rbo = rbo_random
    return critical_rbo


def compare_ranking_rbo(
    feat_ranking_all: List[str],
    feat_ranking_cat: List[str],
    critical_rbo: float,
    k: int = 20,
) -> Tuple[float, bool]:
    """_summary_

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_cat : List[str]
        Feature ranking for a cohort
    critical_rbo : float
        Critical RBO value
    k : int, optional
        Size of the ranking to focus on, by default 20

    Returns
    -------
    Tuple[float, bool]
        RBO(feat_ranking_all, feat_ranking_cat), boolean assessing whether feat_ranking_cat is significantly different from feat_ranking_all
    """
    rbo_value = RankingSimilarity(feat_ranking_all, feat_ranking_cat).rbo(p=0.935, k=k)
    return rbo_value, rbo_value < critical_rbo


def compare_ranking_per_group(
    feat_ranking_all: List[str],
    feat_ranking_per_group: Dict[str, Dict[str, List[str]]],
    critical_rbo: float,
    k: int = 20,
) -> Dict[str, Dict[str, Tuple[float, bool]]]:
    """Compute the RBO of each cohort and assess whether the cohort feature ranking is significantly different.

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_per_group : Dict[str, Dict[str, List[str]]]
        Feature ranking for each cohort of each grouping
    critical_rbo : float
        Critical RBO value
    k : int, optional
        Size of the ranking to focus on, by default 20

    Returns
    -------
    Dict[str, Dict[str, Tuple[float, bool]]]
        Map each cohort of each grouping to the RBO value and boolean assessing whether the cohort feature ranking is significantly different.
    """
    res_comparison = {group_name: {} for group_name in feat_ranking_per_group.keys()}
    for group_name, feat_ranking_group in feat_ranking_per_group.items():
        for cat, feat_ranking_cat in feat_ranking_group.items():
            res_comparison[group_name][cat] = compare_ranking_rbo(
                feat_ranking_all, feat_ranking_cat, critical_rbo, k
            )
    return res_comparison


def get_invdelta_value(rk_all: int, rk_group: int) -> float:
    """Compute the delta of inverse rank.

    Parameters
    ----------
    rk_all : int
        Global rank
    rk_group : int
       Rank from the cohort feature ranking

    Returns
    -------
    float
        Delta of inverse rank value: |1/(global rank) - 1/(cohort rank)|
    """
    return abs(1 / (rk_all + 1) - 1 / (rk_group + 1))


def get_critical_invdelta_value(
    rk_all: int,
    feat_ranking_random_index: Dict[str, Dict[str, Dict[str, int]]],
    feat_name: str,
) -> float:
    """Compute the critical delta of inverse rank value from synthetic feature rankings for a specific feature.

    Parameters
    ----------
    rk_all : int
        Global rank
    feat_ranking_random_index : Dict[str, Dict[str, Dict[str, int]]]
        Map of synthetic feature rankings
    feat_name : str
        Feature name

    Returns
    -------
    float
        Critical inverse delta value
    """
    critical_invdelta = 0
    for dict_feat_ranking_index in feat_ranking_random_index.values():
        for random_rk_index in dict_feat_ranking_index.values():
            rk_random = random_rk_index[feat_name]
            random_invdelta = get_invdelta_value(rk_all, rk_random)
            if random_invdelta > critical_invdelta:
                critical_invdelta = random_invdelta
    return critical_invdelta


def get_critical_invdelta_features(
    feat_ranking_all: List[str], feat_ranking_random: Dict[str, Dict[str, List[str]]]
) -> Dict[str, float]:
    """Compute the critical delta of inverse rank value from synthetic feature rankings for all features.

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_random : Dict[str, Dict[str, List[str]]]
        Map of synthetic feature rankings

    Returns
    -------
    Dict[str, float]
        Map feature name to its critical delta of inverse rank value.
    """
    feat_ranking_random_index = {
        group_name: {
            cat: {feat_name: rk for rk, feat_name in enumerate(list_feat)}
            for cat, list_feat in feat_ranking_random_gp.items()
        }
        for group_name, feat_ranking_random_gp in feat_ranking_random.items()
    }
    critical_invdelta_features = {}
    for rk, feat_name in enumerate(feat_ranking_all):
        critical_invdelta_features[feat_name] = get_critical_invdelta_value(
            rk, feat_ranking_random_index, feat_name
        )
    return critical_invdelta_features
