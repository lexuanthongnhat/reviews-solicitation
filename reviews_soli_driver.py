import logging
import argparse

import data_model
from reviews_soli import ReviewsSolicitation
from edmunds import EdmundsReview
from edmunds_soli import EdmundsReviewSolicitation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s-%(name)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


dataset_to_review_and_sim_cls = {
        'edmunds': (EdmundsReview, EdmundsReviewSolicitation)
        }


def simulate_reviews_soli(file_path, star_rank=5,
                          criterion='weighted_sum_dirichlet_variances',
                          dataset='edmunds'):
    """Simulate the asking process
    Args:
        file_path (string)
        star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
    """
    review_cls, review_soli_sim_cls = dataset_to_review_and_sim_cls[dataset]

    product_to_reviews = review_cls.import_csv(file_path, star_rank=star_rank) 
    dataset_profile = data_model.Review.profile_dataset(product_to_reviews)

    product_to_reviews = {key: value
                          for key, value in product_to_reviews.items()
                          if len(value) >= 970}
    product_to_result_stats = {}
    for product, reviews in product_to_reviews.items():
        product_to_result_stats[product] = simulate_reviews_soli_per_product(
                reviews, review_soli_sim_cls,
                num_polls=100,
                seed_features=review_cls.seed_features,
                criterion=criterion)

    return (product_to_reviews, product_to_result_stats, dataset_profile)


def simulate_reviews_soli_per_product(
        reviews, review_soli_sim_cls,
        num_polls=-1, seed_features=[],
        criterion='weighted_sum_dirichlet_variances'):
    """
    Args:
        reviews: list of Review
        review_soli_sim_cls: ReviewSolicitation class,
            e.g. EdmundsReviewSolicitation
        num_polls: integer of the number of times (reviews) to ask customers
            (default: -1, means the number of reviews available for simulation)
        seed_features: list of product's features, if know upfront
    Returns:
        (greedy_stats, random_stats): tuple of SimulationStats
    """
    sim_stats = []
    for ask_method in ReviewsSolicitation.ask_methods:
        reviews_soli_sim = review_soli_sim_cls(
                reviews,
                num_polls=num_polls,
                seed_features=seed_features,
                criterion=criterion)
        sim_stat = getattr(reviews_soli_sim, ask_method)()
        logger.info(sim_stat.stats_str(ask_method))

    return sim_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reviews Solicitation")
    parser.add_argument("--input", help="dataset input path")
    args = parser.parse_args()
    logger.debug("args: {}".format(args))

    simulate_reviews_soli(args.input)
