import argparse
from collections import OrderedDict
import cProfile
import itertools
import logging
import pickle
import pstats
import sys
from timeit import default_timer

import numpy as np

from data_model import Review
from reviews_soli import SimulationStats, SoliConfig
from edmunds import EdmundsReview, EdmundsReviewSolicitation
from bliu import BliuReview, BliuReviewSolicitation
from semeval import SemevalReview, SemevalReviewSolicitation
from amz_laptop import AmzLaptopReview, AmzLaptopReviewSolicitation
from synthetic import SyntheticReview, SyntheticReviewSolicitation
from uncertainty import UncertaintyMetric


logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


DATASET_SIMULATORS = {
    "edmunds": (5, EdmundsReview, EdmundsReviewSolicitation),   # 5: star
    "bliu": (6, BliuReview, BliuReviewSolicitation),
    "semeval": (3, SemevalReview, SemevalReviewSolicitation),
    "amz_laptop": (3, AmzLaptopReview, AmzLaptopReviewSolicitation),
    "synthetic": (6, SyntheticReview, SyntheticReviewSolicitation),
}


class Scenario(object):
    """Specify experiment scenario details.

    Attributes:
        + name: name of the experiment scenario.
        + soli_configs: list of SoliConfig, which includes how to pick the next
        feature to solicitate, how to generate answer and the optimization
        goal.
        + metrics: list of UncertaintyMetric telling how to measure.
        + product_to_reviews: normally None, useful for 'synthetic' scenario to
        generate synthetic dataset on-fly.
    """
    _scenarios = {}

    def __init__(self, name, soli_configs, metrics,
                 product_to_reviews=None):
        self.name = name
        self.soli_configs = soli_configs
        self.metrics = metrics
        self.product_to_reviews = product_to_reviews

    @classmethod
    def basic(cls):
        metrics = UncertaintyMetric.metrics_standard()
        soli_configs = SoliConfig.build(
            pick_mths=['pick_highest'],
            answer_mths=['answer_by_gen'],
            optm_goals=[
                        UncertaintyMetric('expected_rating_var'),
                        ]
            )
        # sys._getframe().f_code.co_name: current function name, i.e. 'basic'
        return cls(sys._getframe().f_code.co_name, soli_configs, metrics)

    @classmethod
    def passive_vs_active(cls):
        metrics = UncertaintyMetric.metrics_standard()
        soli_configs = SoliConfig.build(
            pick_mths=["pick_highest", "pick_by_user"],
            answer_mths=['answer_almost_real'],
            optm_goals=[
                        UncertaintyMetric('expected_rating_var'),
                        ]
            )
        return cls(sys._getframe().f_code.co_name, soli_configs, metrics)

    @classmethod
    def basic_rated_prob(cls):
        """A basic scenario with consideration of answering probability."""
        metrics = UncertaintyMetric.metrics_standard()
        soli_configs = SoliConfig.build(
            pick_mths=['pick_highest'],
            answer_mths=['answer_by_gen_with_prob'],
            optm_goals=[
                        UncertaintyMetric('expected_rating_var'),
                        UncertaintyMetric('expected_rating_var',
                                          rated_prob=True),
                        ]
            )
        return cls(sys._getframe().f_code.co_name, soli_configs, metrics)

    @classmethod
    def mix_interface(cls):
        """Compare a free-text vs. mixed reviewing interface.

        This method compare a traditional free-text review interface against
        an augmented interface with active solicitation.
        """
        metrics = UncertaintyMetric.metrics_standard()
        soli_configs = []
        MIX = True
        soli_configs.append(SoliConfig(
            'pick_free_text_only', 'answer_by_gen', baseline=True,
            mixed_interface=MIX))

        pick_mths = ['pick_highest']
        answer_mths = ['answer_by_gen', 'answer_by_gen_with_prob']
        optm_goals = [UncertaintyMetric('expected_rating_var')]
        question_counts = [1, 3]

        for pick, question_count, goal, answer in itertools.product(
                pick_mths, question_counts, optm_goals, answer_mths):
            soli_configs.append(SoliConfig(
                pick, answer, optm_goal=goal, mixed_interface=MIX,
                question_count=question_count)
                )
        return cls(sys._getframe().f_code.co_name, soli_configs, metrics)

    @classmethod
    def synthetic(cls):
        """This scenario use created synthetic dataset."""
        metrics = UncertaintyMetric.metrics_standard()
        soli_configs = SoliConfig.build(
            pick_mths=['pick_highest'],
            answer_mths=['answer_by_gen'],
            optm_goals=[
                        UncertaintyMetric('expected_rating_var'),
                        ]
            )
        return cls(
                sys._getframe().f_code.co_name, soli_configs, metrics,
                product_to_reviews=SyntheticReview.import_dataset(
                    None, star_rank=6, feature_count=3, randomize=False)
                )

    @classmethod
    def correlation(cls):
        uncertainty_correlated = UncertaintyMetric(
                'expected_rating_var', correlated=True, corr_threshold=0.5)

        # confidence region measure doesn't apply for co-rating prior
        metrics = [
                uncertainty_correlated,
                UncertaintyMetric('expected_rating_var', correlated=True,
                                  aggregate=np.average, corr_threshold=0.5),
                UncertaintyMetric('confidence_interval_len', correlated=True,
                                  aggregate=np.max, corr_threshold=0.5),
                UncertaintyMetric('confidence_interval_len', correlated=True,
                                  aggregate=np.average, corr_threshold=0.5),
                ]
        soli_configs = SoliConfig.build(
            pick_mths=['pick_highest'],
            answer_mths=['answer_by_gen'],
            optm_goals=[
                        UncertaintyMetric('expected_rating_var'),
                        uncertainty_correlated,
                        ]
            )
        return cls(sys._getframe().f_code.co_name, soli_configs, metrics)


SCENARIOS = {
        Scenario.basic.__name__: Scenario.basic(),
        Scenario.passive_vs_active.__name__: Scenario.passive_vs_active(),
        Scenario.synthetic.__name__: Scenario.synthetic(),
        Scenario.correlation.__name__: Scenario.correlation(),
        Scenario.basic_rated_prob.__name__: Scenario.basic_rated_prob(),
        Scenario.mix_interface.__name__: Scenario.mix_interface(),
        }


def simulate_reviews_soli(product_to_reviews,
                          scenario,
                          star_rank=5,
                          dataset='edmunds',
                          product_count=None,
                          poll_count=100,
                          question_count=1,
                          run_count=5,
                          review_count_lowbound=200,
                          dataset_profile=None,
                          **kwargs):
    """Simulate the asking process
    Args:
        product_to_reviews: dict, product -> list of data_model.Review
        star_rank: int
            e.g. 5 means 1, 2, 3, 4 and 5 stars system
        dataset: string, default='edmunds'
        product_count: int, default=None
            Number of product to experiment
        poll_count: int, default=100
            Number of polls (customers) to ask
        question_count: int, default=1
            Number of questions to ask a customer
        run_count: int, default=5
            Number of simulation run per product
        review_count_lowbound: int, default=200
            Only consider products with more than this lower bound into
        dataset_profile: data_model.DatasetProfile object, default=None
    Returns:
        product_to_config_stats: dict
            product -> config_to_sim_stats, in which
            config_to_sim_stats: SoliConfig -> list of SimulationStats,
                corresponding to SoliConfig.configs()
    """
    _, review_cls, review_soli_sim_cls = DATASET_SIMULATORS[dataset]
    seed_features = review_cls.dup_scenario_features if kwargs['duplicate'] \
        else review_cls.seed_features

    co_ratings_prior = None
    if scenario.name == Scenario.correlation.__name__:
        product_to_reviews, co_ratings_prior = \
                split_dataset_to_correlation_prior(
                        product_to_reviews, star_rank, seed_features)

    product_to_reviews = {key: value
                          for key, value in product_to_reviews.items()
                          if len(value) >= review_count_lowbound}
    if product_count and product_count > 0:
        product_to_reviews = {
                product: product_to_reviews[product]
                for i, product in enumerate(product_to_reviews.keys())
                if i < product_count
                }
    logger.info('# products simulated: {}'.format(len(product_to_reviews)))

    product_to_config_stats = {}
    for product, reviews in product_to_reviews.items():
        logger.info("Running over '{}'".format(product))
        # different aspects set for each product
        if dataset == 'bliu' or dataset == 'semeval' or \
                dataset == 'amz_laptop':
            seed_features = set([feature for review in reviews
                                 for feature in review.features])

        config_to_sim_stats = OrderedDict()
        for soli_config in scenario.soli_configs:
            sim_statses = []
            for i in range(run_count):
                kwargs['co_ratings_prior'] = co_ratings_prior
                reviews_soli_sim = review_soli_sim_cls(
                        reviews, soli_config, scenario.metrics,
                        poll_count=poll_count,
                        question_count=question_count,
                        seed_features=seed_features,
                        dataset_profile=dataset_profile,
                        **kwargs)
                sim_stats = reviews_soli_sim.simulate()
                co_ratings_prior = sim_stats.co_ratings
                sim_statses.append(sim_stats)

            sim_stats_average = \
                SimulationStats.average_same_product_statses(sim_statses)
            config_to_sim_stats[soli_config] = sim_stats_average

            logger.debug(sim_stats_average.stats_str(str(soli_config)))
        product_to_config_stats[product] = config_to_sim_stats

    return product_to_config_stats


def split_dataset_to_correlation_prior(
        product_to_reviews, star_rank, seed_features, prior_ratio=0.5):
    product_to_simulate_count = int(prior_ratio * len(product_to_reviews))
    product_to_reviews_exp = {}
    product_to_reviews_prior = {}
    count = 0
    for product, reviews in product_to_reviews.items():
        count += 1
        if count <= product_to_simulate_count:
            product_to_reviews_prior[product] = reviews
        else:
            product_to_reviews_exp[product] = reviews
    logger.info(f'Use {len(product_to_reviews_prior)} products for co-rating '
                f'count table prior, prior_ratio={prior_ratio}')

    # build prior co-rating count table
    feature_count = len(seed_features)
    co_ratings = np.zeros((feature_count, feature_count, star_rank, star_rank))
    feature_to_id = {feature: i for i, feature in enumerate(seed_features)}
    for product, reviews in product_to_reviews_prior.items():
        for review in reviews:
            for feature_1, stars_1 in review.feature_to_stars.items():
                for feature_2, stars_2 in review.feature_to_stars.items():
                    co_ratings[feature_to_id[feature_2],
                               feature_to_id[feature_1],
                               stars_2[0] - 1,
                               stars_1[0] - 1] += 1
    return product_to_reviews_exp, co_ratings


def summary_product_to_config_stats(product_to_config_stats,
                                    plotted_poll_end=100,
                                    ignore_rating=False):
    """Summary simulation statistics of multiple products.
    Args:
        product_to_config_stats: dict
            product -> config_to_sim_stats, in which
            config_to_sim_stats: SoliConfig -> SimulationStats,
        ignore_rating: bool, default=False
            do not average ratings of multiple products
    Returns:
        soliconfig_to_stats_average: dict,
            soliconfig (SoliConfig) -> SimulationStats
    """
    soliconfig_to_statses = OrderedDict()
    for config_to_stats in product_to_config_stats.values():
        for config, sim_stats in config_to_stats.items():
            if config not in soliconfig_to_statses:
                soliconfig_to_statses[config] = []
            soliconfig_to_statses[config].append(sim_stats)

    soliconfig_to_stats_average = OrderedDict()
    for soliconfig, statses in soliconfig_to_statses.items():
        soliconfig_to_stats_average[soliconfig] = \
                SimulationStats.average_statses(
                        statses,
                        plotted_poll_end=plotted_poll_end,
                        ignore_rating=ignore_rating)
    return soliconfig_to_stats_average


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


def start_sim(args):
    """
    Attributes:
        args: Namespace object, return by ArgumentParser.parse_args()
    """
    star_rank, review_cls, _ = DATASET_SIMULATORS[args.dataset]

    scenario = SCENARIOS[args.scenario]
    if scenario.product_to_reviews:
        product_to_reviews = scenario.product_to_reviews
        star_rank = scenario.star_rank
    else:
        product_to_reviews = review_cls.import_dataset(
                args.input, star_rank=star_rank, duplicate=args.duplicate)

    dataset_profile = Review.probe_dataset(product_to_reviews)
    logger.info('Number of products: {}'.format(dataset_profile.product_count))

    product_to_config_stats = simulate_reviews_soli(
            product_to_reviews,
            scenario,
            star_rank=star_rank,
            dataset=args.dataset,
            product_count=args.product_count,
            poll_count=args.poll_count,
            question_count=args.question_count,
            run_count=args.run_count,
            review_count_lowbound=args.review_count_lowbound,
            dataset_profile=dataset_profile,
            duplicate=args.duplicate)

    with open(args.output, 'wb') as result_file:
        pickle.dump(product_to_config_stats, result_file)
        logger.info("Pickle to '{}'".format(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reviews Solicitation")
    parser.add_argument("--input", help="dataset input path")
    parser.add_argument(
            "--dataset", default="edmunds",
            help="Dataset name (default='edmunds')")
    parser.add_argument(
            "--scenario", default=Scenario.basic.__name__,
            choices=SCENARIOS.keys(),
            help=f"Experiment scenario (default='{Scenario.basic.__name__}')")
    parser.add_argument(
            "--product-count", type=int, default=-1,
            help="Number of product to experiment. Default=-1, or all avail")
    parser.add_argument(
            "--poll-count", type=int, default=-1,
            help="Number of polls (customers) to ask (default=-1, i.e. number"
            " of reviews of the product)")
    parser.add_argument(
            "--question-count", type=int, default=1,
            help="Number of questions to ask a customer (default=1)")
    parser.add_argument(
            "--run-count", type=int, default=5,
            help="Number of simulation run per product (default=5)")
    parser.add_argument(
            "--review-count-lowbound", type=int, default=200,
            help="Only consider products with more than this lower bound into"
            " experiment (default=200)")
    parser.add_argument(
            "--output", default="output",
            help="output file path (default='output/result.pickle')")
    parser.add_argument(
            "--loglevel", default='INFO',
            help="log level (default='INFO')")
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

    logging.getLogger().setLevel(getattr(logging, args.loglevel.upper()))
    logger.info("args: {}".format(args))

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
