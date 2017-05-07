import unittest
from collections import defaultdict

import numpy as np
import scipy.stats as stats


class UncertaintyMetric(object):
    """Modeling different uncertainty metrics.

    Attributes:
        criterion: str,
            individual/independent uncertainty of a single feature
        weighted: Boolean, default=False,
            weighted by global/prior uncertainty information
        correlated: Boolean, default=False,
            infer a feature's uncertainty by other highly correlated features
    """
    def __init__(self, criterion, weighted=False, correlated=False):
        self.criterion = criterion
        self.weighted = weighted
        self.correlated = correlated

    def __repr__(self):
        metric_str = self.criterion
        metric_str += '_weighted' if self.weighted else ''
        metric_str += '_correlated' if self.weighted else ''
        return metric_str

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
                and self.criterion == other.criterion \
                and self.weighted == other.weighted \
                and self.correlated == other.correlated

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.criterion, self.weighted, self.correlated))


class UncertaintyBook(object):
    """Keep track feature's uncertainty.
    Support multiple uncertainty metrics

    Attributes:
        star_rank: int (default=2),
            number of star levels
        feature_count: int,
            number of features
        criterion: string, default='expected_rating_var'
            uncertainty metric
        weighted: Boolean, default=False
            weighted uncertainty metric using prior/global ratings
        correlated: Boolean, default=False
            consider a feature's uncertainty using correlated features
        dataset_profile: SimulationStats object, default=None
            dataset's profile
        ratings: 2d numpy array, shape: (feature_count, star_rank)
            each row records ratings of a feature
        co_ratings: 4d numpy array,
            shape: (feature_count, feature_count, star_rank, star_rank)
            2 first indices define a matrix of 2 features co-ocurrence ratings
        independent_uncertainties: 1d np array, shape: (feature_count, )
            individual feature's uncertainty values
        uncertainties: 1d np array, shape: (feature_count, )
            feature's uncertainties after weighted, correlated
    """
    uncertainty_metrics = [
            UncertaintyMetric('dirichlet_var_sum', weighted=False,
                              correlated=False),
            UncertaintyMetric('dirichlet_var_sum', weighted=True,
                              correlated=False),
            UncertaintyMetric('expected_rating_var', weighted=False,
                              correlated=True),
            UncertaintyMetric('expected_rating_var', weighted=False,
                              correlated=False),
            UncertaintyMetric('expected_rating_var', weighted=True,
                              correlated=False),
            UncertaintyMetric('expected_rating_var', weighted=True,
                              correlated=True),
            UncertaintyMetric('confidence_interval_len', weighted=False,
                              correlated=False)
            ]

    optimization_goals = uncertainty_metrics[:2]

    def __init__(self, star_rank, feature_count,
                 criterion='expected_rating_var',
                 weighted=False,
                 correlated=False,
                 dataset_profile=None,
                 confidence_level=0.95):
        if star_rank < 2 or feature_count < 1:
            raise ValueError('Invalid values of star_rank (>= 2) or '
                             'feature_count (>= 1)')

        self.star_rank = star_rank
        self.feature_count = feature_count
        self.criterion = criterion
        self.correlated = correlated
        self.weighted = weighted
        self.dataset_profile = dataset_profile
        self.confidence_level = confidence_level
        if dataset_profile:
            self.prior_rating_count = \
                dataset_profile.feature_rating_count_average
            ratings_uncertainties = np.apply_along_axis(
                globals()[criterion], 1,
                np.array(dataset_profile.feature_ratings))
            self.prior_uncertainty = np.average(ratings_uncertainties)
            self.prior_uncertainty_total = self.prior_rating_count * \
                self.prior_uncertainty

        self.independent_uncertainties = np.zeros(feature_count)
        self.uncertainties = np.zeros(feature_count)

        self.ratings = np.zeros((feature_count, star_rank))
        self.co_ratings = np.zeros((feature_count, feature_count,
                                    star_rank, star_rank))

    def refresh_uncertainty(self):
        """Refresh (independent) uncertainties to reflect latest ratings.

        Calculate based on self.criterion, self.weighted, self.correlated
        """
        self.independent_uncertainties, self.uncertainties = \
            self.compute_uncertainty(self.criterion, self.weighted,
                                     self.correlated)

    def compute_uncertainty(self, uncertainty_func_name,
                            weighted, correlated):
        """Compute uncertainty using different criteria.

        Args:
            uncertainty_func_name: str
                function name of this module, e.g. 'dirichlet_var_sum',
                'expected_rating_var', 'pearson_cor', 'confidence_interval_len'
            weighted: Boolean
            correlated: Boolean
        Returns:
            (indept_uncertainties, cor_uncertainties)
        """
        indept_uncertainties = np.apply_along_axis(
            globals()[uncertainty_func_name], 1, self.ratings)
        if weighted:
            rating_counts = np.sum(self.ratings, axis=1)
            indept_uncertainties = (indept_uncertainties * rating_counts
                                    + self.prior_uncertainty_total) / (
                                    rating_counts + self.prior_rating_count)

        if not correlated:
            cor_uncertainties = np.copy(indept_uncertainties)
        else:
            correlations = np.apply_along_axis(
                pearson_cor_on_flatten, 2,
                self.co_ratings.reshape(self.feature_count, self.feature_count,
                                        self.star_rank * self.star_rank))
            np.fill_diagonal(correlations, 1)
            correlated_var = indept_uncertainties.reshape(
                self.feature_count, 1) / np.abs(correlations)
            cor_uncertainties = np.min(correlated_var, axis=1)
        return (indept_uncertainties, cor_uncertainties)

    def report_uncertainty(self):
        report = UncertaintyReport()
        for metric in self.uncertainty_metrics:
            report.add_uncertainty(metric, self.uncertainty_total(metric))

        return report

    def uncertainty_total(self, metric):
        """Total uncertainties of all features.

        Args:
            metric: UncertaintyMetric
        Returns:
            sum of all features' uncertainties, float
        """
        _, cor_uncertainties = self.compute_uncertainty(metric.criterion,
                                                        metric.weighted,
                                                        metric.correlated)
        return cor_uncertainties.sum()

    def rate_feature(self, feature, star, count=1):
        """ Rate a single feature."""
        if star < 1 or star > self.star_rank:
            raise IndexError('Wrong star rating (>=1 and <={})'.format(
                self.star_rank))
        self.ratings[feature.idx, star - 1] += count

    def rate_2features(self, feature1, star1, feature2, star2):
        """Update co-rating of 2 features that appear in the same review."""
        if star1 < 1 or star1 > self.star_rank or \
                star2 < 1 or star2 > self.star_rank:
            raise IndexError('Wrong star rating (>=1 and <={})'.format(
                self.star_rank))
        self.co_ratings[feature1.idx, feature2.idx, star1 - 1, star2 - 1] += 1
        self.co_ratings[feature2.idx, feature1.idx, star2 - 1, star1 - 1] += 1


class UncertaintyReport(object):

    def __init__(self, **kargs):
        self.uncertainty_totals = defaultdict(float)

    def add_uncertainty(self, metric, uncertainty_total):
        self.uncertainty_totals[metric] = uncertainty_total

    def __str__(self):
        strs = ['{:50s}: {:.3f}'.format(str(key), value)
                for key, value in self.uncertainty_totals.items()]
        return '\n'.join(strs)

    def metrics(self):
        return list(self.uncertainty_totals.keys())

    def get_metric(self, metric):
        return self.uncertainty_totals[metric]


def weighted_uncertainty(uncertainty, rating_count,
                         prior_count, prior_uncertainty):
    """Weighted uncertainty using global/prior ratings."""
    weighted_value = (rating_count * uncertainty + prior_count *
                      prior_uncertainty) / (rating_count + prior_count)
    return weighted_value


def dirichlet_var_sum(ratings):
    """Sum of feature's Dirichlet variance.
        sum(Var(star = i))
    Args:
        ratings: list, numpy array of star
    """
    dirichlet_params = np.array(ratings) + 1
    return sum(stats.dirichlet.var(dirichlet_params))


def expected_rating_var(ratings):
    """Variance of feature's expected rating.
        Var(r|x) = Var(E[\#stars])
    Args:
        ratings: list, numpy array of star
    """
    dirichlet_params = np.array(ratings) + 1
    cov_matrix = dirich_cov(dirichlet_params)

    d = dirichlet_params.shape[0]
    stars = np.linspace(1, d, d)
    star_weights = np.outer(stars, stars.T)

    feature_var = sum(sum(star_weights * cov_matrix))
    return feature_var


def pearson_cor_on_flatten(flatten_count_table):
    """Similar to pearson_cor except that the argument is flatten array.
    Args:
        count_table: 1-d numpy.array, reshape of a square matrix (2-d array)
    """
    d = int(np.sqrt(flatten_count_table.shape[0]))
    return pearson_cor(flatten_count_table.reshape(d, d))


def pearson_cor(count_table):
    """Pearson correlation of 2 features (with vectorization).

    Args:
        count_table: numpy.array (2-d). For example:
            |          | Cost-1* | Cost-2* | Cost-3* |
            |Screen-1* |    3    |    2    |    0    | 5
            |Screen-2* |    1    |    5    |    2    | 8
            |Screen-3* |    1    |    4    |    7    | 12
            |          |    5    |    11   |    9    |
    Returns:
        a real number
    """
    d = count_table.shape[0]    # number of star levels
    param_table = count_table + 1

    fat_dirich_params = param_table.flatten()
    fat_covs = np.array(dirich_cov(fat_dirich_params))    # dim: d^2 * d^2
    # cell-cell-covariance
    cc_covs = fat_covs.reshape(d, d, d, d)

    row_row_var = cc_covs.sum(axis=3).sum(axis=1)
    col_col_var = cc_covs.sum(axis=2).sum(axis=0)
    row_col_var = cc_covs.sum(axis=2).sum(axis=1)

    stars = np.linspace(1, d, d)
    star_weights = np.outer(stars, stars.T)
    rvar = (row_row_var * star_weights).sum().sum()
    cvar = (col_col_var * star_weights).sum().sum()
    cov = (row_col_var * star_weights).sum().sum()

    return cov / np.sqrt(rvar * cvar)


def confidence_interval_len(rating_counts, confidence_level=0.95):
    """Confidence interval length using Student distribution
    Args:
        rating_counts: numpy array of star count
        confidence_level: float, default=0.95
    """
    n = np.sum(rating_counts + 1)

    stars = np.arange(1, rating_counts.shape[0] + 1)
    mean = rating_counts.dot(stars) / n
    spread = stars - mean
    sd = np.sqrt(rating_counts.dot(spread * spread) / n)
    if sd == 0:
        return 0
    lower, upper = stats.t.interval(confidence_level, n - 1,
                                    loc=mean, scale=sd / np.sqrt(n))
    return upper - lower


def simple_pearson_cor(count_table):
    """Simple Pearson correlation of 2 features (without vectorization).

    Args:
        count_table: numpy.array (2-d). For example:
            |          | Cost-1* | Cost-2* | Cost-3* |
            |Screen-1* |    3    |    2    |    0    | 5
            |Screen-2* |    1    |    5    |    2    | 8
            |Screen-3* |    1    |    4    |    7    | 12
            |          |    5    |    11   |    9    |
    Returns:
        a real number
    """
    def fat_idx(row_idx, col_idx, dim):
        """Flatten (i, j) index (2-d) to 1-d index."""
        return row_idx * dim + col_idx

    d = count_table.shape[0]    # number of star levels
    param_table = count_table + 1

    # Flattten statistics from Dirichlet distribution
    fat_dirich_params = param_table.flatten()
    fat_covs = np.array(dirich_cov(fat_dirich_params))    # dim: d^2 * d^2

    # row/column variance - e.g., Var(p^s_1)
    #     p^s_1 = p^s_11 + p^s_12 + p^s_13
    row_var = np.zeros((d, 1))
    col_var = np.zeros((d, 1))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                row_var[i, 0] += fat_covs[fat_idx(i, j, d), fat_idx(i, k, d)]
                col_var[i, 0] += fat_covs[fat_idx(j, i, d), fat_idx(k, i, d)]

    # row-row/col-col covariance - e.g., Cov(p^s_1, p^s_3), Cov(p^c_1, p^c_2)
    rr_cov = np.zeros((d, d))
    cc_cov = np.zeros((d, d))
    for i in range(d):
        for k in range(d):
            if i == k:
                rr_cov[i, k] = row_var[i, 0]
                cc_cov[i, k] = col_var[i, 0]
            else:
                for j in range(d):
                    for t in range(d):
                        rr_cov[i, k] += fat_covs[fat_idx(i, j, d),
                                                 fat_idx(k, t, d)]
                        cc_cov[i, k] += fat_covs[fat_idx(j, i, d),
                                                 fat_idx(t, k, d)]

    # row-column covariance - e.g., Cov(p^s_1, p^c_3)
    rc_cov = np.zeros((d, d))
    for i in range(d):
        for k in range(d):
            for j in range(d):
                for t in range(d):
                    row_idx = fat_idx(i, j, d)
                    col_idx = fat_idx(t, k, d)
                    rc_cov[i, k] += fat_covs[row_idx, col_idx]

    rvar = 0
    cvar = 0
    for i in range(d):
        for j in range(d):
            rvar += (i + 1) * (j + 1) * rr_cov[i, j]
            cvar += (i + 1) * (j + 1) * cc_cov[i, j]

    cov = 0
    for i in range(d):
        for k in range(d):
            cov += (i + 1) * (k + 1) * rc_cov[i, k]

    return cov / np.sqrt(rvar * cvar)


def dirich_cov(alphas):
    """Dirichlet distribution's covariances matrix.

    Diagonal is filled by variance instead.
    Args:
        alphas: list, numpy array of Dirichlet distribution parameters
    Returns:
        cov - numpy matrix (narray), covariance matrix
    """
    alphas = np.array(alphas)
    if np.min(alphas) <= 0:
        raise ValueError("All Dirichlet parameters must be greater than 0")
    dim = alphas.shape[0]
    a0 = sum(alphas)
    denom = a0 ** 2 * (a0 + 1)
    cov = - alphas.reshape(dim, 1).dot(alphas.reshape(1, dim)) / denom
    np.fill_diagonal(cov, alphas * (a0 - alphas) / denom)

    return cov


class UncertaintyTester(unittest.TestCase):

    def testPearsonCor(self):
        count_tables = []
        count_tables.append(np.array([[3, 3, 0], [1, 7, 2], [1, 4, 7]]))
        count_tables.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        count_tables.append(np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]))
        count_tables.append(np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]]))

        for count_table in count_tables:
            self.assertEqual(pearson_cor(count_table),
                             simple_pearson_cor(count_table))


if __name__ == '__main__':
    count_tables = []
    for i in range(0, 50, 5):
        count_tables.append(np.array([[i, 0, 0, 0, 0],
                                      [0, i, 0, 0, 0],
                                      [0, 0, i, 0, 0],
                                      [0, 0, 0, i, 0],
                                      [0, 0, 0, 0, i]]))
        count_tables.append(np.array([[i, 0, 0],
                                      [0, i, 0],
                                      [0, 0, i]]))
        count_tables.append(np.array([[0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, i]]))
        count_tables.append(np.array([[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, i]]))

    for table in count_tables:
        print(table, ' ---> ', pearson_cor(table))
    unittest.main()

    for table in count_tables:
        print(table, ' ---> ', pearson_cor(table))
    unittest.main()
