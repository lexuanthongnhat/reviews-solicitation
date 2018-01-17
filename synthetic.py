import logging
import pickle
import random

import numpy as np
import scipy as sp

from data_model import Review
from edmunds import EdmundsReviewSolicitation


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


class SyntheticReview(Review):
    seed_features = []
    dup_scenario_features = []
    BETA_BINO_PARAMS = np.array([
            [1, 1], [20, 1],
            [0.1, 0.1], [100, 1], [3, 60],
            [0.2, 0.25], [2, 2.5],
            [0.1, 0.7], [0.1, 10], [50, 50],
            [0.01, 0.01], [1, 1.5], [6, 8], [3, 20], [15, 4],
            [0.5, 1], [ 3, 2.5], [7, 0.2], [5, 25], [5.5, 9]
        ])

    @classmethod
    def import_dataset(cls, dataset_path, star_rank=6, duplicate=False,
                       feature_count=2, randomize=False):
        reviews = []
        for i in range(feature_count):
            feature = "feature_" + str(i)
            cls.seed_features.append(feature)

            alpha, beta = cls.BETA_BINO_PARAMS[i, :] if not randomize \
                                                     else random_alpha_beta()
            logger.debug("alpha, beta: {}, {}".format(alpha, beta))
            star_dist = np.array(beta_binomial(alpha, beta, star_rank - 1))
            star_counts = np.ceil(star_dist * 5 * star_rank)

            for star, count in enumerate(star_counts):
                feature_to_stars = {feature: [star + 1] * int(count)}
                review = cls(feature_to_stars, star_rank=star_rank)
                reviews.append(review)

        product_to_reviews = {"Synthetic Product": reviews}
        with open("output/synthetic_data.pickle", "wb") as result_file:
            pickle.dump(cls.probe_synthetic(product_to_reviews), result_file)

        return product_to_reviews

    @classmethod
    def probe_synthetic(cls, product_to_reviews):
        product_to_aspect_stars = {}
        for product, reviews in product_to_reviews.items():
            aspect_to_star_counts = {}
            for review in reviews:
                for feature, stars in review.feature_to_stars.items():
                    if feature not in aspect_to_star_counts:
                        aspect_to_star_counts[feature] = {}
                    for star in stars:
                        if star not in aspect_to_star_counts[feature]:
                            aspect_to_star_counts[feature][star] = 0
                        aspect_to_star_counts[feature][star] += 1
            product_to_aspect_stars[product] = aspect_to_star_counts
        return product_to_aspect_stars


class SyntheticReviewSolicitation(EdmundsReviewSolicitation):
    """Synthetic reviews have a fixed set of features that make the
    simulation much simpler.
    """


def random_alpha_beta():
    """Randomly create alpha, beta for Beta_Binomial distribution.

    Range: 0.01 -> 100
    Return:
        (alpha, beta): tuple
    """
    top = 50
    alpha, beta = random.randint(1, top), random.randint(1, top)
    if random.random() >= 0.5: alpha /= top
    if random.random() >= 0.5: beta /= top
    return (alpha, beta)


def beta_binomial(alpha, beta, n):
    """Beta Binomial Distribution generator.

    Args:
        alpha: float,
            parameter of prior Beta distribution
        beta: float,
            parameter of prior Beta distribution
        n: int,
            number of binomial trial
    Returns:
        dist: list of probability of getting k (k = 0...n)
    """
    return [beta_binom_pmf(alpha, beta, n, k) for k in range(n + 1)]


def beta_binom_pmf(alpha, beta, n, k):
    """Beta Binomial Probability Mass Function.

    Using the logarithm trick to avoid numerical limitation.
    Args:
        alpha: float,
            parameter of prior Beta distribution
        beta: float,
            parameter of prior Beta distribution
        n: int,
            number of binomial trial
        k: int,
            number of successes
    """
    part1 = np.log(sp.special.comb(n, k))
    part2 = sp.special.betaln(k + alpha, n - k + beta)
    part3 = sp.special.betaln(alpha, beta)
    return np.exp(part1 + part2 - part3)


def beta_binom_var(alpha, beta, n):
    var = (n * alpha * beta * (alpha + beta + n))
    var /= (alpha + beta) * (alpha + beta) * (alpha + beta + 1)
    return var


if __name__ == "__main__":
    alpha = 2
    beta = 2
    n = 10
    print(beta_binom(alpha, beta, n))
