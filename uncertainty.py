import unittest
import itertools
from collections import OrderedDict, defaultdict
import copy

import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, psi
from scipy.spatial.distance import pdist, squareform


class UncertaintyMetric(object):
    """Modeling different uncertainty metrics.

    Attributes:
        criterion: str,
            individual/independent uncertainty of a single feature
        correlated: bool, default=False,
            infer a feature's uncertainty by other highly correlated features
        corr_threshold: float, default=0.5
            correlation threshold to consider 2 aspects as fully correlated
            0.5 is the default due to 'Correlation Coefficient Rule of Thumb'
            by Krehbiel, T. C. (2004)
        rated_prob: bool, default=False
            consider probability that the asked aspect will be answered (rated)
            The specific probability values  will be derived from history.
            This is only used for optimization goal.
        aggregate: aggregate function, default=np.max
            select from list of feature's uncertainty
    """
    __metrics = []

    def __init__(self,
                 criterion,
                 correlated=False,
                 corr_threshold=0.5,
                 rated_prob=False,
                 aggregate=np.max,
                 ratio=False,
                 ):
        self.criterion = criterion
        self.correlated = correlated
        self.corr_threshold = corr_threshold
        self.rated_prob = rated_prob
        self.aggregate = aggregate
        self.ratio = ratio      # hack: high_confidence_ratio

    def __repr__(self):
        if self.ratio:
            return self.show()
        return self.show(show_aggregate=True)

    def show(self, show_aggregate=False):
        metric_str = self.criterion

        # temporary hack to shorten this metric for pretty plot
        if metric_str == "expected_rating_var":
            metric_str = "variance"
        if show_aggregate:
            if self.aggregate.__name__ == "amax":
                metric_str = 'max_' + metric_str
            else:
                metric_str = self.aggregate.__name__ + '_' + metric_str
        metric_str += '_correlated' if self.correlated else ''
        if self.correlated and self.corr_threshold != 1.0:
            metric_str += '_corr_threshold_{}'.format(self.corr_threshold)
        metric_str += '_with_rated_prob' if self.rated_prob else ''
        return metric_str

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and self.criterion == other.criterion \
            and self.correlated == other.correlated \
            and self.corr_threshold == other.corr_threshold \
            and self.rated_prob == other.rated_prob \
            and self.aggregate == other.aggregate

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.criterion, self.correlated, self.corr_threshold,
                     self.rated_prob, str(self.aggregate)))

    @classmethod
    def metrics_standard(cls):
        """List of standard, un-correlated metrics."""
        return [
                UncertaintyMetric('expected_rating_var'),
                UncertaintyMetric('confidence_interval_len'),
                UncertaintyMetric(
                    'high_confidence_ratio', aggregate=np.average, ratio=True),
                ]

    @classmethod
    def metrics(cls):
        if not cls.__metrics:
            cls.__metrics.append(cls('dirichlet_var_sum', correlated=False))
            cls.__metrics.append(cls('expected_rating_var', correlated=False))
            cls.__metrics.append(cls('expected_rating_var', correlated=True))

            cls.__metrics.append(cls('confidence_interval_len',
                                     aggregate=np.average))
            cls.__metrics.append(cls('confidence_interval_len',
                                     aggregate=np.max))
            cls.__metrics.append(cls('confidence_region_vol',
                                     aggregate=np.average))
            cls.__metrics.append(cls('confidence_region_vol',
                                     aggregate=np.max))
        return cls.__metrics

    @classmethod
    def optm_goals(cls):
        return cls.metrics()[:3]


class UncertaintyBook(object):
    """Keep track feature's uncertainty.
    Support multiple uncertainty metrics

    Attributes:
        star_rank: int (default=2),
            number of star levels
        feature_count: int,
            number of features
        rating_truth_dists: list of star_dist (list) or 2d np.array
            truth distribution used for generating simulated rating
        optm_goal: UncertaintyMetric
        aspect_to_rated_prob: dict, data_model.Feature -> float, default=None,
            aspect probability of being rated
        ratings: 2d numpy array, shape: (feature_count, star_rank)
            each row records ratings of a feature
        co_ratings: 4d numpy array, bootstraped by one for every slot.
            shape: (feature_count, feature_count, star_rank, star_rank)
            2 first indices define a matrix of 2 features co-ocurrence ratings
        independent_uncertainties: 1d np array, shape: (feature_count, )
            individual feature's uncertainty values
        uncertainties: 1d np array, shape: (feature_count, )
            feature's uncertainties after correlated
    """

    def __init__(self,
                 star_rank,
                 feature_count,
                 rating_truth_dists=None,
                 optm_goal=None,
                 aspect_to_rated_prob=None,
                 co_ratings_prior=None,
                 ):
        if star_rank < 2 or feature_count < 1:
            raise ValueError('Invalid values of star_rank (>= 2) or '
                             'feature_count (>= 1)')

        self.star_rank = star_rank
        self.feature_count = feature_count
        self.optm_goal = optm_goal
        self.rating_truth_dists = rating_truth_dists

        self.independent_uncertainties = np.zeros(feature_count)
        self.uncertainties = np.zeros(feature_count)
        self.criterion_to_cache_unc = {}
        self.correlations_cache = None

        self.rated_probs = np.ones(feature_count)
        if not aspect_to_rated_prob:
            aspect_id_to_rated_prob = {
                aspect.idx: prob
                for aspect, prob in aspect_to_rated_prob.items()}
            self.rated_probs = np.array(
                [aspect_id_to_rated_prob[idx] for idx in range(feature_count)])

        self.ratings = np.ones((feature_count, star_rank))
        self.co_ratings = np.copy(co_ratings_prior) \
            if co_ratings_prior is not None \
            else np.ones((feature_count, feature_count, star_rank, star_rank))

        self.prev_ratings = np.ones((feature_count, star_rank))
        self.vars = [[0, 1] for i in range(feature_count)]

    def refresh_uncertainty(self):
        """Refresh (independent) uncertainties to reflect latest ratings.

        Calculate based on self.criterion, self.correlated
        """
        self.criterion_to_cache_unc = {}
        self.correlations_cache = None

        # Update uncertainties for the next pick in simulation
        if self.optm_goal:
            self.independent_uncertainties, self.uncertainties = \
                self.compute_uncertainty(self.optm_goal)

    def compute_uncertainty(self, metric):
        """Compute uncertainty using different criteria.

        Args:
            metric: UncertaintyMetric
        Returns:
            (indept_uncertainties, cor_uncertainties)
        """
        criterion = metric.criterion

        if criterion == "confidence_region_vol":
            if criterion not in self.criterion_to_cache_unc:
                self.criterion_to_cache_unc[criterion] = np.apply_along_axis(
                    globals()[criterion], 2, self.co_ratings.reshape(
                        self.feature_count, self.feature_count,
                        self.star_rank * self.star_rank))
            confid_region_vols = self.criterion_to_cache_unc[criterion]
            cor_uncertainties = metric.aggregate(confid_region_vols, axis=1)
            return (cor_uncertainties, cor_uncertainties)

        if criterion == "distribution_change" or criterion == "var_change":
            if criterion not in self.criterion_to_cache_unc:
                self.criterion_to_cache_unc[criterion] = \
                        globals()[criterion](self.ratings, self.prev_ratings)
            indept_uncertainties = self.criterion_to_cache_unc[criterion]
            return (indept_uncertainties, indept_uncertainties)

        if criterion == "high_confidence_ratio":
            # Count the number of features that have crediable interval width
            # smaller than a threshold, e.g. 1 star
            # z = 1.96    # 95% confidence
            width_threshold = 1
            # width_threshold = self.star_rank / 5
            base_criterion = "confidence_interval_len"
            # base_criterion = "expected_rating_var"
            if base_criterion not in self.criterion_to_cache_unc:
                self.criterion_to_cache_unc[base_criterion] = \
                        np.apply_along_axis(globals()[base_criterion], 1,
                                            self.ratings)
            aspect_vars = np.copy(self.criterion_to_cache_unc[base_criterion])
            # interval_lens = aspect_vars * z
            interval_lens = aspect_vars
            indept_uncertainties = interval_lens <= width_threshold
            return (indept_uncertainties, indept_uncertainties)

        if criterion == "kl_divergence":
            self.criterion_to_cache_unc[criterion] = np.array(
                    [kl_divergence(truth_dist, self.ratings[i, :])
                     for i, truth_dist in enumerate(self.rating_truth_dists)])

        if criterion not in self.criterion_to_cache_unc:
            self.criterion_to_cache_unc[criterion] = np.apply_along_axis(
                globals()[criterion], 1, self.ratings)
        indept_uncertainties = np.copy(self.criterion_to_cache_unc[criterion])

        if metric.rated_prob:
            indept_uncertainties *= self.rated_probs
        if not metric.correlated:
            cor_uncertainties = np.copy(indept_uncertainties)
        else:
            if self.correlations_cache is None:
                self.correlations_cache = get_feature_correlations(
                        self.co_ratings)
            cor_uncertainties = correlated_uncertainty(
                    indept_uncertainties, self.correlations_cache,
                    corr_threshold=metric.corr_threshold)

        return (indept_uncertainties, cor_uncertainties)

    def report_uncertainty(self, metrics):
        """Report current uncertainty.

        Args:
            metrics: list of UncertaintyMetric objects
        Returns:
            report: UncertaintyReport
        """
        metric_to_total = OrderedDict()
        for metric in metrics:
            metric_to_total[metric] = self.uncertainty_total(metric)

        base_criterion = UncertaintyMetric.metrics()[0].criterion
        if base_criterion in self.criterion_to_cache_unc:
            criterion_to_uncertainties = {
                base_criterion: self.criterion_to_cache_unc[base_criterion]}
        else:
            criterion_to_uncertainties = {}

        report = UncertaintyReport(
            metric_to_total=metric_to_total,
            ratings=np.copy(self.ratings - 1),
            criterion_to_uncertainties=criterion_to_uncertainties,
            correlations=self.correlations_cache)
        return report

    def uncertainty_total(self, metric):
        """Aggregate uncertainties of all features.

        Args:
            metric: UncertaintyMetric
        Returns:
            Aggregate by all features' uncertainties, float
        """
        _, cor_uncertainties = self.compute_uncertainty(metric)
        return metric.aggregate(cor_uncertainties)

    def get_rating_count(self):
        """Get the number of rating per feature.

        Returns:
            1d numpy array, number of elements = feature_count
        """
        return np.sum(self.ratings, axis=1)

    def rate_feature(self, feature, star, count=1):
        """ Rate a single feature."""
        if star < 1 or star > self.star_rank:
            raise IndexError('Wrong star rating (>=1 and <={})'.format(
                self.star_rank))
        self.prev_ratings[feature.idx, :] = \
            np.copy(self.ratings[feature.idx, :])
        self.ratings[feature.idx, star - 1] += count
        self.vars[feature.idx].append(
                expected_rating_var(self.ratings[feature.idx]))

    def rate_2features(self, feature1, star1, feature2, star2):
        """Update co-rating of 2 features that appear in the same review."""
        if star1 < 1 or star1 > self.star_rank or \
                star2 < 1 or star2 > self.star_rank:
            raise IndexError('Wrong star rating (>=1 and <={})'.format(
                self.star_rank))
        self.co_ratings[feature1.idx, feature2.idx, star1 - 1, star2 - 1] += 1
        if feature1.idx != feature2.idx:
            self.co_ratings[feature2.idx, feature1.idx,
                            star2 - 1, star1 - 1] += 1


class UncertaintyReport(object):

    def __init__(self,
                 metric_to_total=OrderedDict(),
                 metric_to_std=OrderedDict(),
                 ratings=None,
                 criterion_to_uncertainties=None,
                 correlations=None):
        """Uncertainty Report at a specific point of simulation.

        Attributes:
            metric_to_total: dict
                metric -> uncertainty total of all features
            metric_to_std: dict
                metric -> standard deviation of different uncertainty totals
                over multiple experimental runs
            ratings: 2-d np array,
                ratings of all features, from UncertaintyBook.ratings
            criterion_to_uncertainties: dict,
                base criterion -> independent uncertainties,
                base metric are the one without correlated but not
                confidence based. So far, base criterion is
                'expected_rating_var'
            correlations: 2-d np array,
        """
        self.metric_to_total = metric_to_total
        self.metric_to_std = metric_to_std
        self.ratings = ratings
        self.criterion_to_uncertainties = criterion_to_uncertainties
        self.correlations = correlations

    def add_uncertainty(self, metric, uncertainty_total):
        self.metric_to_total[metric] = uncertainty_total

    def __str__(self):
        strs = ['{:50s}: {:.3f}'.format(str(key), value)
                for key, value in self.metric_to_total.items()]
        return '\n'.join(strs)

    def metrics(self):
        return list(self.metric_to_total.keys())

    def get_uncertainty_total(self, metric):
        return self.metric_to_total[metric]

    @classmethod
    def average_reports(cls, reports, ignore_rating=False):
        """Average multiple UncertaintyReport.
        Reports can be from different products.

        Args:
            reports: list of UncertaintyReport
            ignore_rating: bool, default=False,
                ignore averaging rating of all products
        """
        if not reports or len(reports) < 1:
            raise ValueError('Empty reports')

        metric_to_total_average = OrderedDict()
        metric_to_std = OrderedDict()
        for metric in reports[0].metrics():
            uncrtnty_totals = np.array([report.get_uncertainty_total(metric)
                                        for report in reports])
            metric_to_total_average[metric] = np.average(uncrtnty_totals)

            # Standard Deviation of different products should be just averaged
            if reports[0].metric_to_std:
                metric_to_std[metric] = np.average(np.array(
                    [report.metric_to_std[metric] for report in reports]))

        ratings_average = None
        if not ignore_rating:
            ratings_average = sum([report.ratings for report in reports])
            ratings_average = ratings_average / len(reports)

        return cls(metric_to_total=metric_to_total_average,
                   metric_to_std=metric_to_std,
                   ratings=ratings_average)

    @classmethod
    def average_same_product_reports(cls, reports):
        """Average multiple UncertaintyReport of the same product.

        This product is simulated multiple times
        Args:
            reports: list of UncertaintyReport
        """
        report_count = len(reports)
        report_average = cls.average_reports(reports)

        # Standard deviation of multiple runs
        for metric in reports[0].metrics():
            uncrtnty_totals = np.array([report.get_uncertainty_total(metric)
                                        for report in reports])
            report_average.metric_to_std[metric] = np.std(uncrtnty_totals)

        cor_average = None
        if reports and reports[0].correlations is not None:
            cor_average = sum([report.correlations for report in reports])
            cor_average = cor_average / report_count

        criterion_to_uncertainties_average = defaultdict(int)
        for report in reports:
            for criterion, uncertainties \
                    in report.criterion_to_uncertainties.items():
                criterion_to_uncertainties_average[criterion] += uncertainties
        for criterion, uncertainties in \
                criterion_to_uncertainties_average.items():
            criterion_to_uncertainties_average[criterion] = \
                    uncertainties / report_count

        report_average.correlations = cor_average
        report_average.criterion_to_uncertainties = \
            criterion_to_uncertainties_average
        return report_average


def get_feature_correlations(co_ratings):
    """Compute feature's correlations matrix from co_ratings.

    Correlation is calculated by function 'pearson_cor_on_flatten'
    Args:
        co_ratings: 4-d np array,
            from UncertaintyBook.co_ratings
            shape: (feature_count, feature_count, star_rank, star_rank)
            2 first indices define a matrix of 2 features co-ocurrence ratings
    Returns:
        correlations: 2-d np array, shape: feature_count * feature_count
            each cell represents 2 corresponding features' correlation
    """
    feature_count, star_rank = co_ratings.shape[1:3]
    correlations = np.apply_along_axis(
        pearson_cor_on_flatten, 2,
        co_ratings.reshape(feature_count, feature_count, star_rank * star_rank)
        )

    np.fill_diagonal(correlations, 1)
    return correlations


def correlated_uncertainty(indept_uncertainties, correlations,
                           corr_threshold=0.5):
    """Compute feature's correlated uncertainty.

    Formula: cor_uncertainty(feature_Xi)
             = min {independent_uncertainty(feature_Xj) / corr_factor(Xi, Yj)}
    Currently correlations_factor func uses a threshold to trigger correlating.
    That is when corr(Xi, Xj) > threshold, corr_factor(Xi, Xj) = 1, so maybe
    switch to other aspect uncertainty value.
    """
    feature_count = indept_uncertainties.shape[0]
    correlated_var = indept_uncertainties.reshape(feature_count, 1) / \
        correlations_factor(correlations, corr_threshold)
    cor_uncertainties = np.min(correlated_var, axis=0)
    return cor_uncertainties


def correlations_factor(correlations, corr_threshold):
    """Currently employ Heaviside/Unit step function style."""
    corrs = np.abs(correlations)
    corrs[corrs >= corr_threshold] = 1.0
    corrs[corrs < corr_threshold] = 0.01        # not 0 to avoid divide by zero
    return corrs


def dirichlet_var_sum(ratings):
    """Sum of feature's Dirichlet variance.
        sum(Var(star = i))
    Args:
        ratings: list, numpy array of star
    """
    dirichlet_params = np.array(ratings) + 1
    return sum(stats.dirichlet.var(dirichlet_params))


def distribution_change(ratings, prev_ratings):
    """Uncertainty by considering distribution change.
    Idea: if one more rating doesn't change aspect's distribution much, then
    that aspect is quite stable.
    Change is computed by Euclidean distance between the ratings and the
    previous updated ratings.

    Args:
        ratings: 2d np array
            expect from UncertaintyBook.ratings
        prev_ratings: 2d np array
            same shape as ratings
    Returns:
        uncertainties: 1d np array
    """
    def _distance(curr, prev):
        return (stats.entropy(curr, prev) + stats.entropy(prev, curr)) / 2

    uncertainties = np.array([_distance(curr, prev_ratings[i, :])
                              for i, curr in enumerate(ratings)])
    uncertainties[np.all(ratings == prev_ratings, axis=1)] = \
        np.max(uncertainties)
    return uncertainties


def var_change(ratings, prev_ratings):
    """Uncertainty by considering distribution change.
    Idea: if one more rating doesn't change aspect's distribution much, then
    that aspect is quite stable.
    Change is computed by Variance change between the ratings and the
    previous updated ratings.

    Args:
        ratings: 2d np array
            expect from UncertaintyBook.ratings
        prev_ratings: 2d np array
            same shape as ratings
    Returns:
        uncertainties: 1d np array
    """
    base_criterion = "expected_rating_var"
    curr_uncertainties = np.apply_along_axis(globals()[base_criterion], 1,
                                             ratings)
    prev_uncertainties = np.apply_along_axis(globals()[base_criterion], 1,
                                             prev_ratings)
    var_diff = np.abs(curr_uncertainties - prev_uncertainties)
    var_diff[np.all(ratings == prev_ratings, axis=1)] = np.max(var_diff) + 0.01
    return var_diff


def naive_var(ratings):
    """Naive (dicrete) variance of aspect without Bayesian inference.
    Args:
        ratings: list, numpy array of star
    """
    stars = []
    for i in range(len(ratings)):
        stars.extend([i + 1] * int(ratings[i]))
    return np.var(stars)


def expected_rating_var(ratings):
    """Variance of feature's expected rating.
            Var(r|x) = Var(E[#stars])
    Variance of linear combination has quaratic form matrix solution
            Var(aX) = a'∑a
        in which a is column vector, a' means a's transpose, and ∑ is
        covariance matrix of random vector X
    Args:
        ratings: list, numpy array of star
    """
    dirichlet_params = np.array(ratings) + 1
    cov_matrix = dirich_cov(dirichlet_params)

    d = dirichlet_params.shape[0]
    stars = np.linspace(1, d, d)

    # star_weights = np.outer(stars, stars.T)
    # feature_var = sum(sum(star_weights * cov_matrix))
    feature_var = stars.dot(cov_matrix).dot(stars)
    return feature_var


def expected_uncertainty_drop(ratings, base_criterion=expected_rating_var):
    """Expected Uncertainty Drop after the next user's answer.

    Given the current "ratings", estimate the drop of "expected_rating_var" or
    any uncertainty metrics when we get a new rating after asking a new user.
    The probability of getting s_i stars from the next user is estimated by
    utilizing the current rating distribution.
        E[E[Var[r | n + 1]]] = sum(beta_i / beta_0 * Var[r | n, s_i stars])

    Args:
        ratings: list, numpy array of star
    Returns:
        real number
    """
    dirichlet_params = np.array(ratings) + 1
    beta0 = dirichlet_params.sum()
    beta = dirichlet_params / beta0

    next_ratings = ratings + np.identity(dirichlet_params.shape[0])
    next_unertainties = np.apply_along_axis(base_criterion, 1, next_ratings)

    curr_uncertainty = base_criterion(ratings)
    expected_next_uncertainty = np.sum(beta * next_unertainties)
    expected_drop = curr_uncertainty - expected_next_uncertainty
    return expected_drop


def entropy(ratings):
    """Differential entropy of Dirichlet distribution.

    Can be negative, maxima when Dirichlet distribution is uniform.
    """
    return stats.dirichlet.entropy(ratings + 1)


def shannon_entropy(ratings):
    return stats.entropy(ratings)


def kl_divergence(truth_dist, ratings):
    """Kullback-Leibler divergence of expected rating posterior to the truth.

    Note: scipy.stats.entropy returns KL instead of normal entropy if 2
    arguments are served.
    Args:
        truth_dist: 1d numpy array
            the truth distribution used for generating star rating
        ratings: 1d numpy array of rating counts
    Return:
        Kullback-Leibler divergence of the dirichet mean posterior from truth
            assume ratings has a dirichlet posterior that has posterior mean
                        KL(truth_dist | posterior_mean)
    """
    dirich_params = ratings + 1
    posterior_mean = dirich_params / dirich_params.sum()
    return stats.entropy(truth_dist, posterior_mean)


def kl_of_dirs(alpha, beta):
    """Kullback-Leibler divergence between two Dirichlet distribution.
    """
    alpha_0 = np.sum(alpha)
    beta_0 = np.sum(beta)
    kl = gammaln(alpha_0) - gammaln(beta_0) - \
        np.sum(gammaln(alpha)) + np.sum(gammaln(beta)) + \
        np.sum((alpha - beta) * (psi(alpha) - psi(alpha_0)))
    return kl


def sym_kl_of_dirs(alpha, beta):
    m = (alpha + beta) / 2
    return (kl_of_dirs(alpha, m) + kl_of_dirs(beta, m)) / 2


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
            |Screen-1* |    3    |    2    |    0    |
            |Screen-2* |    1    |    5    |    2    |
            |Screen-3* |    1    |    4    |    7    |
        where d = 3
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


def convert_count_table_to_pairs(count_table):
    d = count_table.shape[0]    # number of star levels
    rows = []
    cols = []
    for i in range(d):
        for j in range(d):
            for _ in range(int(count_table[i, j])):
                rows.append(i + 1)
                cols.append(j + 1)
    return rows, cols


def pearson_traditional(count_table):
    rows, cols = convert_count_table_to_pairs(count_table)
    return stats.pearsonr(rows, cols)


def spearman_rank_cor(count_table):
    rows, cols = convert_count_table_to_pairs(count_table)
    return stats.spearmanr(rows, cols)


def distcorr(Xval, Yval, pval=True, nruns=500):
    """Compute the distance correlation function, returning the p-value.
    Test with Distance Correlation but computational expensive and not worth.

    https://gist.github.com/wladston/c931b1495184fbb99bec
    Based on Satra/distcorr.py (gist aa3d19a12b74e9ab7941)
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    (0.76267624241686671, 0.404)
    """
    X = np.atleast_1d(Xval)
    Y = np.atleast_1d(Yval)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        for i in range(nruns):
            Y_r = copy.copy(Yval)
            np.random.shuffle(Y_r)
            if distcorr(Xval, Y_r, pval=False) >= dcor:
                greater += 1
        return (dcor, greater / float(nruns))
    else:
        return dcor


def confidence_interval_len(rating_counts, confidence_level=0.95):
    """Confidence interval length using Student distribution
    Args:
        rating_counts: numpy array of star count
        confidence_level: float, default=0.95
    """
    n = np.sum(rating_counts)

    stars = np.arange(1, rating_counts.shape[0] + 1)
    mean = rating_counts.dot(stars) / n
    spread = stars - mean
    sd = np.sqrt(rating_counts.dot(spread * spread) / n)
    if sd == 0:
        return 0
    lower, upper = stats.t.interval(confidence_level, n - 1,
                                    loc=mean, scale=sd / np.sqrt(n))
    if upper > len(rating_counts):
        upper = len(rating_counts)
    if lower < 1:
        lower = 1
    return upper - lower


def confidence_region_vol(co_rating_samples_flatten, confidence_level=0.95):
    """Confidence region volumn of feature pair using Hotelling distribution.
        Google keyword: confidence region vector mean
        Ellipse axes follow:
            sqrt(lambda_i) * sqrt((p(n - 1) / n(n - p)) * F(alpha, p, n - p))
            * e_i
            lambda_i, e_i are eigenvalue, eigenvector of covariance matrix
            p: dimension of random vector
            n: number of observations/samples
            alpha: confidence level (normally: 0.95, i.e. 95%)
            F(alpha, p, n - p): follow F distribution
    Args:
        co_rating_samples_flatten: flatten co_ratings matrix of a single
            feature pair's in UncertaintyBook.co_ratings
    Returns:
        vol: volumn of confidence region (ellipse).
    """
    co_ratings_raw = _convert_co_rating_flatten_to_raw(
        co_rating_samples_flatten)
    p = 2
    n = co_ratings_raw.shape[1]
    cov_matrix = np.cov(co_ratings_raw)
    w, v = np.linalg.eig(cov_matrix)

    f_value = stats.f.ppf(confidence_level, p, n - p)
    fixed_factor = np.sqrt((p * (n - 1) * f_value) / (n * (n - p)))
    ellipse_axes = fixed_factor * np.sqrt(np.abs(w)) * v
    ellipse_axes_len = np.linalg.norm(ellipse_axes, axis=0)
    vol = np.prod(ellipse_axes_len) * np.pi
    return vol


def _convert_co_rating_flatten_to_raw(co_rating_table_flatten):
    """
    Returns:
        co_ratings_raw: numpy array of 2 features and their observations.
            each row is a feature, each column is an observation. This is
            compitable with numpy.linalg.cov function
            Note: this argument is assumed to be bootstrapped with ones if
            there is no information.
    """
    d = int(np.sqrt(co_rating_table_flatten.shape[0]))
    co_rating_table = co_rating_table_flatten.reshape(d, d)
    co_ratings_raw = np.zeros((2, int(np.sum(co_rating_table_flatten))))
    count_total = 0
    for i, j in itertools.product(range(d), repeat=2):
        count = int(co_rating_table[i, j])
        if count > 0:
            co_ratings_raw[:, count_total:count_total + count] = np.tile(
                np.array([[i + 1], [j + 1]]), count)
            count_total += count
    return co_ratings_raw


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
    rates = np.array([
        [5, 5, 5, 5, 5],
        [1, 1, 1, 1, 21]
        ])
    raw_stars = []
    for i in range(rates.shape[0]):
        counts = rates[i, :]
        stars = []
        for i, count in enumerate(counts):
            stars.extend([i + 1] * count)
        print("std: {}".format(np.std(stars)))
        raw_stars.append(stars)
    print(raw_stars)
    print(np.apply_along_axis(expected_rating_var, 1, rates))

    count_tables = []
    for i in range(0, 51, 10):
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

    x = np.array([[10, 0, 0],
                  [0, 10, 0],
                  [0, 0, 10]])
    print(confidence_region_vol(x.flatten(), confidence_level=0.95))
    unittest.main()
