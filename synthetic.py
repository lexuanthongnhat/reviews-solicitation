import logging
import pickle

import numpy as np
import scipy as sp

from data_model import Review
from edmunds import EdmundsReviewSolicitation
from uncertainty import expected_rating_var


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


class SyntheticReview(Review):
    seed_features = []
    dup_scenario_features = []

    @classmethod
    def import_dataset(cls, dataset_path, star_rank=6, duplicate=False,
                       feature_count=2):
        reviews = []

        alpha_betas = np.array([
            [2, 2], [20, 1],
            [0.1, 0.1], [100, 1], [3, 60],
            [0.2, 0.25], [2, 2], [1, 1],
            [0.1, 0.7], [0.1, 10], [50, 50],
            [0.01, 0.01]
            ])

        for i in range(feature_count):
            feature = "feature_" + str(i)
            cls.seed_features.append(feature)

            alpha, beta = alpha_betas[i, :]
            star_dist = np.array([beta_binom_pmf(alpha, beta, star_rank - 1, k)
                                  for k in range(star_rank)])
            star_counts = np.ceil(star_dist * 5 * star_rank)

            if i == 6:
                # star_counts = np.array([10, 2, 2, 1, 9, 10, 1,  2, 2, 10])
                star_counts = np.array([10, 2, 2, 1, 9, 11, 1, 2, 2, 10,
                                        11, 1, 3, 1, 10, 9, 2, 1, 2, 10])

            # Randomize generate
            # count_total = star_rank * 9
            # star_counts = np.zeros(star_rank + 1)
            # star_counts[-1] = count_total
            # star_counts[1:-1] = np.sort(np.random.randint(
                # 1, high=count_total, size=star_rank - 1))
            # star_counts = np.diff(star_counts)
            # star_counts += 1

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
    for k in range(11):
        print(beta_binom_pmf(alpha, beta, n, k))