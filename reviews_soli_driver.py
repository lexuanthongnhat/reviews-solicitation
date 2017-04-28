import logging
import argparse

import data_model
from reviews_soli import ReviewsSolicitation
from edmunds import EdmundsReview
from edmunds_soli import EdmundsReviewSolicitation


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s-%(name)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


dataset_to_review_and_sim_cls = {
    'edmunds': (EdmundsReview, EdmundsReviewSolicitation)
}


def simulate_reviews_soli(file_path,
                          star_rank=5,
                          dataset='edmunds',
                          num_polls=-1,
                          num_questions=1,
                          lower_num_reviews=200,
                          criterion='weighted_sum_dirichlet_variances',
                          prior_count=None,
                          prior_cost=None):
    """Simulate the asking process
    Args:
        file_path: string
        star_rank: int
            e.g. 5 means 1, 2, 3, 4 and 5 stars system
        dataset: string, default='edmunds'
        num_polls: int, default=-1 (i.e. number of reviews of the product)
            Number of polls (customers) to ask
        num_questions: int, default=1
            Number of questions to ask a customer
        lower_num_reviews: int, default=200
            Only consider products with more than this lower bound into
        criterion: string, default='weighted_sum_dirichlet_variances'
        prior_count: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
        prior_cost: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
    Returns:
        product_to_result_stats: dict
            product -> sim_stats (list of SimulationStats, corresponding to
            ReviewsSolicitation.ask_methods)
    """
    review_cls, review_soli_sim_cls = dataset_to_review_and_sim_cls[dataset]

    product_to_reviews = review_cls.import_csv(file_path, star_rank=star_rank)
    product_to_reviews = {key: value
                          for key, value in product_to_reviews.items()
                          if len(value) >= lower_num_reviews}
    logger.info('# products simulated: {}'.format(len(product_to_reviews)))

    product_to_result_stats = {}
    for product, reviews in product_to_reviews.items():
        product_to_result_stats[product] = simulate_reviews_soli_per_product(
            reviews, review_soli_sim_cls,
            num_polls=num_polls,
            num_questions=num_questions,
            seed_features=review_cls.seed_features,
            criterion=criterion)

    return product_to_result_stats


def simulate_reviews_soli_per_product(
        reviews, review_soli_sim_cls,
        num_polls=-1,
        num_questions=1,
        seed_features=[],
        criterion='weighted_sum_dirichlet_variances',
        prior_count=None,
        prior_cost=None):
    """
    Args:
        reviews: list of Review
        review_soli_sim_cls: ReviewSolicitation class,
            e.g. EdmundsReviewSolicitation
        num_polls: integer of the number of times (reviews) to ask customers
            (default: -1, means the number of reviews available for simulation)
        num_questions: int, default=1,
            Number of questions to ask a customer
        seed_features: list of product's features, if know upfront
        criterion: string, default='weighted_sum_dirichlet_variances'
        prior_count: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
        prior_cost: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
    Returns:
        sim_stats: list of SimulationStats, corresponding to
                   ReviewsSolicitation.ask_methods
    """
    sim_stats = []
    for ask_method in ReviewsSolicitation.ask_methods:
        reviews_soli_sim = review_soli_sim_cls(
            reviews,
            num_polls=num_polls,
            num_questions=num_questions,
            seed_features=seed_features,
            criterion=criterion)
        sim_stat = reviews_soli_sim.simulate(ask_method)
        sim_stats.append(sim_stat)
        logger.debug(sim_stat.stats_str(ask_method))

    return sim_stats


def profile_dataset(file_path, star_rank=5, dataset='edmunds'):
    """Profiling the dataset
    Args:
        file_path (string)
        star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
    Returns:
        dataset_profile: data_model.DatasetProfile object
    """
    review_cls, _ = dataset_to_review_and_sim_cls[dataset]

    product_to_reviews = review_cls.import_csv(file_path, star_rank=star_rank)
    dataset_profile = data_model.Review.profile_dataset(product_to_reviews)
    return dataset_profile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reviews Solicitation")
    parser.add_argument("--input", help="dataset input path")
    parser.add_argument(
            "--star-rank", type=int, default=5,
            help="Number of different star levels (default=5)")
    parser.add_argument(
            "--dataset", default="edmunds",
            help="Dataset name (default='edmunds')")
    parser.add_argument(
            "--num-polls", type=int, default=-1,
            help="Number of polls (customers) to ask (default=-1, i.e. number "
            "of reviews of the product)")
    parser.add_argument(
            "--num-questions", type=int, default=1,
            help="Number of questions to ask a customer (default=1)")
    parser.add_argument(
            "--lower-num-reviews", type=int, default=200,
            help="Only consider products with more than this lower bound into "
            " experiment (default=200)")
    parser.add_argument(
            "--loglevel", default='WARN',
            help="log level (default='WARN')")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug("args: {}".format(args))

    dataset_profile = profile_dataset(args.input)
    logger.info('# products: {}'.format(dataset_profile.num_products))

    simulate_reviews_soli(
        args.input,
        star_rank=args.star_rank,
        dataset=args.dataset,
        num_polls=args.num_polls,
        num_questions=args.num_questions,
        lower_num_reviews=args.lower_num_reviews,
        prior_count=dataset_profile.ave_num_feature_ratings_per_product,
        prior_cost=dataset_profile.global_ave_sum_variances)
