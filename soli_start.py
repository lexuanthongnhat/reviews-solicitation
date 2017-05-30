import logging
import argparse
import pickle
from collections import OrderedDict
import cProfile
import pstats
from timeit import default_timer

from data_model import Review
from reviews_soli import SimulationStats, SoliConfig
from edmunds import EdmundsReview, EdmundsReviewSolicitation
from bliu import BliuReview, BliuReviewSolicitation


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


dataset_to_review_and_sim_cls = {
    'edmunds': (5, EdmundsReview, EdmundsReviewSolicitation),
    'bliu': (6, BliuReview, BliuReviewSolicitation)
}


def simulate_reviews_soli(product_to_reviews,
                          star_rank=5,
                          dataset='edmunds',
                          poll_count=-1,
                          question_count=1,
                          review_count_lowbound=200,
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
        dataset_profile: SimulationStats object, default=None
            dataset's profile
    Returns:
        product_to_config_stats: dict
            product -> config_to_sim_stats, in which
            config_to_sim_stats: SoliConfig -> list of SimulationStats,
                corresponding to SoliConfig.configs()
    """
    product_to_reviews = {key: value
                          for key, value in product_to_reviews.items()
                          if len(value) >= review_count_lowbound}
    logger.info('# products simulated: {}'.format(len(product_to_reviews)))

    _, review_cls, review_soli_sim_cls = dataset_to_review_and_sim_cls[dataset]
    seed_features = review_cls.dup_scenario_features if kwargs['duplicate'] \
        else review_cls.seed_features
    product_to_config_stats = {}
    for product, reviews in product_to_reviews.items():
        logger.debug("Running over '{}'".format(product))
        # different aspects set for each product
        if dataset == 'bliu':
            seed_features = set([feature for review in reviews
                                 for feature in review.features])

        config_to_sim_stats = OrderedDict()
        for soli_config in SoliConfig.configs(dataset):
            reviews_soli_sim = review_soli_sim_cls(
                    reviews, soli_config,
                    poll_count=poll_count,
                    question_count=question_count,
                    seed_features=seed_features,
                    dataset_profile=dataset_profile,
                    **kwargs)

            sim_stat = reviews_soli_sim.simulate()
            config_to_sim_stats[soli_config] = sim_stat
            logger.debug(sim_stat.stats_str(str(soli_config)))
        product_to_config_stats[product] = config_to_sim_stats

    return product_to_config_stats


def summary_product_to_config_stats(product_to_config_stats,
                                    plotted_poll_end=100,
                                    ignore_rating=False):
    """Summary simulation statistics of multiple products.
    Args:
        product_to_config_stats: dict
            product -> config_to_sim_stats, in which
            config_to_sim_stats: SoliConfig -> list of SimulationStats,
                corresponding to SoliConfig.configs()
        ignore_rating: bool, default=False
            do not average ratings of multiple products
    """
    # goal -> SimulationStats (poll_to_cost)
    goal_to_statses = OrderedDict()
    for config_to_stats in product_to_config_stats.values():
        for config, sim_stats in config_to_stats.items():
            if config not in goal_to_statses:
                goal_to_statses[config] = []
            goal_to_statses[config].append(sim_stats)

    goal_to_stats_average = OrderedDict()
    for goal, statses in goal_to_statses.items():
        goal_to_stats_average[goal] = SimulationStats.average_statses(
                statses, plotted_poll_end=plotted_poll_end,
                ignore_rating=ignore_rating)
    return goal_to_stats_average


def summary_optim_goal_ratings(optim_goal_to_product_result_stats):
    """Summary multiple optimization goals.
    TODO - should summary base on same poll or same cost (uncertainty)?

    Args:
        optim_goal_to_product_result_stats: dict,
            optim_goal(UncertaintyMetric) -> product_to_config_stats (output of
            simulate_reviews_soli function)
    """
    poll_to_optim_goal_ratings = OrderedDict()
    for optim_goal, product_to_config_stats in \
            optim_goal_to_product_result_stats.items():
        for pick_answer_to_sim_stats in product_to_config_stats.values():
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
    _, review_cls, _ = dataset_to_review_and_sim_cls[dataset]
    product_to_reviews = review_cls.import_dataset(file_path,
                                                   star_rank=star_rank)
    dataset_profile = Review.probe_dataset(product_to_reviews)
    return dataset_profile


def start_sim(args):
    """
    Attributes:
        args: Namespace object, return by ArgumentParser.parse_args()
    """
    star_rank, review_cls, _ = dataset_to_review_and_sim_cls[args.dataset]
    product_to_reviews = review_cls.import_dataset(args.input,
                                                   star_rank=star_rank,
                                                   duplicate=args.duplicate)

    dataset_profile = Review.probe_dataset(product_to_reviews)
    logger.info('Number of products: {}'.format(dataset_profile.product_count))

    product_to_config_stats = simulate_reviews_soli(
            product_to_reviews,
            star_rank=star_rank,
            dataset=args.dataset,
            poll_count=args.poll_count,
            question_count=args.question_count,
            review_count_lowbound=args.review_count_lowbound,
            dataset_profile=dataset_profile,
            duplicate=args.duplicate)

    with open(args.output, 'wb') as result_file:
        pickle.dump(product_to_config_stats, result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reviews Solicitation")
    parser.add_argument("--input", help="dataset input path")
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
    if args.duplicate and args.question_count < 2:
        args.question_count = 2

    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug("args: {}".format(args))

    if args.profile:
        profile = cProfile.Profile()
        profile.runcall(start_sim, args)
        stats = pstats.Stats(profile).sort_stats('cumulative')
        stats.print_stats()
    else:
        start_time = default_timer()
        start_sim(args)
        elapsed_time = default_timer() - start_time
        logger.info("Simulation finished in {:.2f} seconds or ({:.2f} "
                    "minutes)".format(elapsed_time, elapsed_time / 60))
