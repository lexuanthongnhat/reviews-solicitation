import logging
import argparse
import itertools
import pickle
from collections import OrderedDict
import cProfile
import pstats
from timeit import default_timer

from data_model import Review
from reviews_soli import ReviewsSolicitation, SimulationStats
from edmunds import EdmundsReview
from edmunds_soli import EdmundsReviewSolicitation
from uncertainty import UncertaintyBook


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s-%(name)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


dataset_to_review_and_sim_cls = {
    'edmunds': (EdmundsReview, EdmundsReviewSolicitation)
}


def main(args):
    """
    Attributes:
        args: Namespace object, return by ArgumentParser.parse_args()
    """
    review_cls, _ = dataset_to_review_and_sim_cls[args.dataset]
    product_to_reviews = review_cls.import_csv(args.input,
                                               star_rank=args.star_rank,
                                               duplicate=args.duplicate)

    dataset_profile = Review.probe_dataset(product_to_reviews)
    logger.info('Number of products: {}'.format(dataset_profile.product_count))

    optim_goal_to_product_result_stats = OrderedDict()
    for optim_goal in UncertaintyBook.optimization_goals:
        logger.debug('Optimization goal: "{}"'.format(optim_goal))
        product_to_result_stats = simulate_reviews_soli(
                product_to_reviews,
                star_rank=args.star_rank,
                dataset=args.dataset,
                poll_count=args.poll_count,
                question_count=args.question_count,
                review_count_lowbound=args.review_count_lowbound,
                criterion=optim_goal.criterion,
                weighted=optim_goal.weighted,
                correlated=optim_goal.correlated,
                cor_norm_factor=optim_goal.cor_norm_factor,
                dataset_profile=dataset_profile,
                confidence_level=args.confidence_level,
                duplicate=args.duplicate)
        optim_goal_to_product_result_stats[optim_goal] = \
            product_to_result_stats

    with open(args.output, 'wb') as result_file:
        pickle.dump(optim_goal_to_product_result_stats, result_file)


def simulate_reviews_soli(product_to_reviews,
                          star_rank=5,
                          dataset='edmunds',
                          poll_count=-1,
                          question_count=1,
                          review_count_lowbound=200,
                          criterion='expected_rating_var',
                          weighted=False,
                          correlated=False,
                          cor_norm_factor=1.0,
                          dataset_profile=None,
                          **kwargs):
    """Simulate the asking process
    Args:
        product_to_reviews: dict, product -> list of data_model.Review
        star_rank: int
            e.g. 5 means 1, 2, 3, 4 and 5 stars system
        dataset: string, default='edmunds'
        poll_count: int, default=-1 (i.e. number of reviews of the product)
            Number of polls (customers) to ask
        question_count: int, default=1
            Number of questions to ask a customer
        review_count_lowbound: int, default=200
            Only consider products with more than this lower bound into
        criterion: string, default='expected_rating_var'
            uncertainty metric
        weighted: Boolean, default=False
            weighted uncertainty metric using prior/global ratings
        correlated: Boolean, default=False
            consider a feature's uncertainty using correlated features
        dataset_profile: SimulationStats object, default=None
            dataset's profile
    Returns:
        product_to_result_stats: dict
            product -> sim_stats (list of SimulationStats, corresponding to
            ReviewsSolicitation.pick_methods/answer_methods)
    """
    product_to_reviews = {key: value
                          for key, value in product_to_reviews.items()
                          if len(value) >= review_count_lowbound}
    logger.info('# products simulated: {}'.format(len(product_to_reviews)))

    review_cls, review_soli_sim_cls = dataset_to_review_and_sim_cls[dataset]
    seed_features = review_cls.dup_scenario_features if kwargs['duplicate'] \
        else review_cls.seed_features
    product_to_result_stats = {}
    for product, reviews in product_to_reviews.items():
        logger.debug('feature ratings: {}'.format(
            dataset_profile.product_to_feature_ratings[product]))
        product_to_result_stats[product] = simulate_reviews_soli_per_product(
            reviews, review_soli_sim_cls,
            poll_count=poll_count,
            question_count=question_count,
            seed_features=seed_features,
            weighted=weighted,
            criterion=criterion,
            correlated=correlated,
            cor_norm_factor=cor_norm_factor,
            dataset_profile=dataset_profile,
            **kwargs)

    return product_to_result_stats


def simulate_reviews_soli_per_product(
        reviews, review_soli_sim_cls,
        poll_count=-1,
        question_count=1,
        seed_features=[],
        criterion='weighted_sum_dirichlet_variances',
        weighted=False,
        correlated=False,
        cor_norm_factor=1.0,
        dataset_profile=None,
        **kwargs):
    """
    Args:
        reviews: list of Review
        review_soli_sim_cls: ReviewSolicitation class,
            e.g. EdmundsReviewSolicitation
        poll_count: int, default=-1 (i.e. number of reviews of the product)
            Number of polls (customers) to ask
        question_count: int, default=1
            Number of questions to ask a customer
        review_count_lowbound: int, default=200
            Only consider products with more than this lower bound into
        criterion: string, default='expected_rating_var'
            uncertainty metric
        weighted: Boolean, default=False
            weighted uncertainty metric using prior/global ratings
        correlated: Boolean, default=False
            consider a feature's uncertainty using correlated features
        dataset_profile: SimulationStats object, default=None
            dataset's profile
    Returns:
        pick_answer_to_sim_stats: dict,
            tuple (pick_method, answer_method) -> SimulationStats
            from ReviewsSolicitation.pick_methods/answer_methods
    """
    pick_answer_to_sim_stats = OrderedDict()
    for answer_method, pick_method in itertools.product(
            ReviewsSolicitation.answer_methods,
            ReviewsSolicitation.pick_methods):
        reviews_soli_sim = review_soli_sim_cls(
                reviews,
                poll_count=poll_count,
                question_count=question_count,
                seed_features=seed_features,
                criterion=criterion,
                weighted=weighted,
                correlated=correlated,
                cor_norm_factor=cor_norm_factor,
                dataset_profile=dataset_profile,
                **kwargs)
        sim_stat = reviews_soli_sim.simulate(pick_method, answer_method)
        pick_answer_to_sim_stats[(pick_method, answer_method)] = sim_stat
        logger.debug(sim_stat.stats_str(pick_method + ' - ' + answer_method))

    return pick_answer_to_sim_stats


def summary_sim_stats(product_to_result_stats):
    """Summary simulation statistics of multiple products.
    Args:
        product_to_result_stats: dict, product -> pick_answer_to_sim_stats
            pick_answer_to_sim_stats is a dict returned by
            simulate_reviews_soli_per_product
    """
    pick_answer_to_sim_statses = OrderedDict()
    for pick_answer_to_sim_stats in product_to_result_stats.values():
        for pick_answer, sim_stats in pick_answer_to_sim_stats.items():
            if pick_answer not in pick_answer_to_sim_statses:
                pick_answer_to_sim_statses[pick_answer] = []
            pick_answer_to_sim_statses[pick_answer].append(sim_stats)

    pick_answer_to_sim_stats_average = OrderedDict()
    for pick_answer, sim_statses in pick_answer_to_sim_statses.items():
        pick_answer_to_sim_stats_average[pick_answer] = \
                SimulationStats.sim_stats_average(sim_statses)
    return pick_answer_to_sim_stats_average


def summary_optim_goal_ratings(optim_goal_to_product_result_stats):
    """Summary multiple optimization goals.
    TODO - should summary base on same poll or same cost (uncertainty)?

    Args:
        optim_goal_to_product_result_stats: dict,
            optim_goal(UncertaintyMetric) -> product_to_result_stats (output of
            simulate_reviews_soli function)
    """
    poll_to_optim_goal_ratings = OrderedDict()
    for optim_goal, product_to_result_stats in \
            optim_goal_to_product_result_stats.items():
        for pick_answer_to_sim_stats in product_to_result_stats.values():
            for pick_answer, sim_stats in pick_answer_to_sim_stats.items():
                pass

    return poll_to_optim_goal_ratings


def probe_dataset(file_path, star_rank=5, dataset='edmunds'):
    """Profiling the dataset
    Args:
        file_path (string)
        star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
        edmunds: string, specify the dataset
    Returns:
        dataset_profile: data_model.DatasetProfile object
    """
    review_cls, _ = dataset_to_review_and_sim_cls[dataset]

    product_to_reviews = review_cls.import_csv(file_path, star_rank=star_rank)
    dataset_profile = Review.probe_dataset(product_to_reviews)
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
            "--poll-count", type=int, default=-1,
            help="Number of polls (customers) to ask (default=-1, i.e. number "
            "of reviews of the product)")
    parser.add_argument(
            "--question-count", type=int, default=1,
            help="Number of questions to ask a customer (default=1)")
    parser.add_argument(
            "--confidence-level", type=float, default=0.95,
            help="Confidence level for confidence interval metric (default=1)")
    parser.add_argument(
            "--review-count-lowbound", type=int, default=200,
            help="Only consider products with more than this lower bound into "
            " experiment (default=200)")
    parser.add_argument(
            "--output", default="output",
            help="output directory (default='output/result.pickle')")
    parser.add_argument(
            "--loglevel", default='WARN',
            help="log level (default='WARN')")
    parser.add_argument(
            "--profile", action="store_true",
            help="Profile the program")
    parser.add_argument(
            "--duplicate", action="store_true",
            help="Duplicate scenario for experimentation: 3 features, "
                 "2 are duplicate, ask 2 question per poll")
    args = parser.parse_args()
    if args.duplicate:
        args.question_count = 2

    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug("args: {}".format(args))

    if args.profile:
        profile = cProfile.Profile()
        profile.runcall(main, args)
        stats = pstats.Stats(profile).sort_stats('cumulative')
        stats.print_stats()
    else:
        start_time = default_timer()
        main(args)
        end_time = default_timer()
        logger.info("Simulation finished in {} seconds".format(
                end_time - start_time))
