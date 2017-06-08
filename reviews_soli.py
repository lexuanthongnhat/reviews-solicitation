from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import itertools
import warnings

import numpy as np

from data_model import Feature, Review
from uncertainty import UncertaintyBook, UncertaintyReport, UncertaintyMetric


class SoliConfig(object):
    """Solicitation Configuation: optimization goal, picking/answering.

    Attributes:
        pick: str, func name of picking method
        answer: str, func name of answering method
        optm_goal: uncertainty.UncertaintyMetric
    """
    def __init__(self, pick, answer, optm_goal=None):
        self.pick = pick
        self.answer = answer
        self.optm_goal = optm_goal

    @classmethod
    def build(cls,
              pick_mths=['pick_highest', 'pick_prob'],
              answer_mths=['answer_by_gen', 'answer_in_time_order'],
              optm_goals=[UncertaintyMetric('expected_rating_var')]
              ):
        """Build Solicitation Configuration.

        Baseline configs are always in the beginning of the return list.
        Returns:
            configs: list of SoliConfig
        """
        configs = []

        pick_baselines = ['pick_random', 'pick_least_count']
        for pick, answer in itertools.product(pick_baselines, answer_mths):
            configs.append(cls(pick, answer))

        for pick, answer, goal in itertools.product(pick_mths, answer_mths,
                                                    optm_goals):
            configs.append(cls(pick, answer, optm_goal=goal))

        return configs

    def pick_goal_str(self):
        config = self.pick
        if self.optm_goal:
            config += '_' + self.optm_goal.show()
        return config

    def __repr__(self):
        return self.pick_goal_str() + '_' + self.answer

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
                and self.pick == other.pick \
                and self.answer == other.answer \
                and self.optm_goal == other.optm_goal

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.pick, self.answer, str(self.optm_goal)))

    def is_gen_answer(self):
        return self.answer.endswith('_gen')


class ReviewsSolicitation(ABC):
    """
    Attributes:
        reviews: list of data_model.Review
        soli_config: SoliConfig object
        metrics: list of UncertaintyMetric objects
        poll_count: int, default=-1 (i.e. len(reviews))
            how many times can ask customers
        question_count: int, default=1,
            Number of questions to ask a customer
        seed_features: list of features name (str), if any (default: [])
        dataset_profile: data_model.DatasetProfile object, default=None
        poll_to_report: OrderedDict,
            cost's report change after each question
            poll starts from 0
    """
    def __init__(self, reviews, soli_config, metrics,
                 poll_count=100,
                 question_count=1,
                 seed_features=[],
                 dataset_profile=None,
                 **kwargs):
        if len(reviews) < 1:
            raise ValueError('Empty or None reviews')
        self.original_reviews = reviews
        self.reviews = reviews.copy()
        self.star_rank = reviews[0].star_rank

        self.soli_config = soli_config
        self.metrics = metrics
        self.poll_count = poll_count
        self.question_count = question_count

        self.seed_features = seed_features
        self.features = [Feature(i, feature_name)
                         for i, feature_name in enumerate(self.seed_features)]
        self.feature_to_star_dist = Review.sample_star_dist(reviews,
                                                            self.features)
        self.duplicate = True if kwargs['duplicate'] else False
        if self.duplicate:
            # 2 duplicate features' index in Review.dup_scenario_features
            self.duplicate_feature_idx = [-1, -2]

        # Keep track feature's uncertainty
        self.uncertainty_book = UncertaintyBook(
                self.star_rank,
                len(self.features),
                optm_goal=soli_config.optm_goal,
                dataset_profile=dataset_profile)
        self.poll_to_report = OrderedDict()

    def simulate(self):
        """Simulate the asking-aswering process."""
        for i in range(self.poll_count):
            self.uncertainty_book.refresh_uncertainty()
            # Keep track in case of answer_in_time_order, i.e. get all answers
            # from a single real review
            self.num_waiting_answers = self.question_count

            already_picked_idx = []
            rated_features = []
            for q in range(self.question_count):
                self.uncertainty_book.refresh_uncertainty()
                picked_feature = self.__getattribute__(self.soli_config.pick)(
                        already_picked_idx)
                answered_star = self.__getattribute__(self.soli_config.answer)(
                        picked_feature)

                # In duplicate feature scenario: make sure dup features get
                # the same star.
                if self.duplicate and already_picked_idx \
                        and picked_feature.idx in self.duplicate_feature_idx:
                    for pre_rated_feature, pre_star in rated_features:
                        if pre_rated_feature.idx in self.duplicate_feature_idx:
                            answered_star = pre_star
                            break

                # Update ratings, rating's uncertainty
                already_picked_idx.append(picked_feature.idx)
                if answered_star:
                    self.uncertainty_book.rate_feature(picked_feature,
                                                       answered_star)
                    self.uncertainty_book.rate_2features(
                            picked_feature, answered_star,
                            picked_feature, answered_star)

                    # Update co-rating of 2 features
                    if rated_features:
                        for pre_rated_feature, pre_star in rated_features:
                            self.uncertainty_book.rate_2features(
                                    pre_rated_feature, pre_star,
                                    picked_feature, answered_star)
                    rated_features.append((picked_feature, answered_star))
                else:
                    picked_feature.no_answer_count += 1

            self.poll_to_report[i] = \
                self.uncertainty_book.report_uncertainty(self.metrics)

        return SimulationStats(self.poll_count, self.question_count,
                               self.poll_to_report,
                               self.features,
                               criterion_to_prior=self.uncertainty_book.prior)

    @abstractmethod
    def answer_by_gen(self, picked_feature):
        """Answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        Args: picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """

    @abstractmethod
    def answer_in_time_order(self, picked_feature):
        """Answer using real reviews sorted in time order.
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """

    def pick_highest(self, already_picked_idx):
        """Pick a feature with highest cost, break tie arbitrarily.
        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
        """
        if not already_picked_idx:
            max_idx = np.argmax(self.uncertainty_book.uncertainties)
            return self.features[max_idx]
        else:
            excluded_uncertainties = np.copy(
                    self.uncertainty_book.uncertainties)
            excluded_uncertainties[already_picked_idx] = -float('inf')
            max_indices = np.where(
                    excluded_uncertainties == excluded_uncertainties.max())[0]
            return self.features[np.random.choice(max_indices)]

    def pick_prob(self, already_picked_idx):
        """Ask features with probability proportional to its cost,
        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
        """
        warnings.filterwarnings("error")
        try:
            weights = self.uncertainty_book.uncertainties / \
                np.sum(self.uncertainty_book.uncertainties)
        except RuntimeWarning:
            import pdb
            pdb.set_trace()
        while True:
            picked_feature = np.random.choice(self.features, p=weights)
            if picked_feature.idx not in already_picked_idx:
                return picked_feature

    def pick_random(self, already_picked_idx):
        """Pick a feature randomly
        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
        """
        while True:
            picked_feature = np.random.choice(self.features)
            if picked_feature.idx not in already_picked_idx:
                return picked_feature

    def pick_least_count(self, already_picked_idx):
        """Pick a feature with least number of ratings.
        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
        """
        rating_counts = self.uncertainty_book.get_rating_count()
        if already_picked_idx:
            rating_counts[already_picked_idx] = float('inf')
        min_indices = np.where(rating_counts == rating_counts.min())[0]
        return self.features[np.random.choice(min_indices)]


class SimulationStats(object):
    """Resulting statistics of simulation
    Attributes:
        poll_count: int, how many time can ask customers
        question_count: int, number of question per customer
        poll_to_report: dict, poll -> UncertaintyReport
            poll starts from 0
        features (list): list of data_model.Feature
        criterion_to_prior: dict, from UncertaintyBook.prior
    """
    def __init__(self, poll_count, question_count, poll_to_report, features,
                 criterion_to_prior=None):
        self.poll_count = poll_count
        self.question_count = question_count
        self.poll_to_report = poll_to_report
        self.polls = list(self.poll_to_report.keys())
        self.uncertainty_reports = list(self.poll_to_report.values())

        self.features = features
        if features:
            self.no_answer_count = sum([feature.no_answer_count
                                        for feature in self.features])
        self.criterion_to_prior = criterion_to_prior

    def stats_str(self, message='', detail=False):
        stat_str = message + '\n'

        if detail:
            reports = ['{}: {:.3f}'.format(poll, cost)
                       for poll, cost in self.poll_to_report.items()]
            stat_str += ', '.join(reports) + '\n'
        else:
            last_poll = len(self.poll_to_report) - 1
            stat_str += 'Final cost after {} polls:\n{}\n'.format(
                last_poll, self.poll_to_report[last_poll])

        stat_str += '/no_answer_count={}'.format(self.no_answer_count)
        return stat_str

    @classmethod
    def average_statses(cls, sim_statses,
                        plotted_poll_end=100,
                        ignore_rating=False):
        """Averaging multiple product's sim stats.

        Args:
            sim_states: list of SimulationStats
            plotted_poll_end: int, default=100
                last poll to be plotted. Must re-weight weighted metric's
                uncertainty accroding to this poll.
            ignore_rating: bool, default=False,
                ignore averaging rating of all products.
        """
        criterion_to_prior = sim_statses[0].criterion_to_prior
        for sim_stats in sim_statses:
            poll_end_ratings_count = np.sum(
                    sim_stats.poll_to_report[plotted_poll_end].ratings, axis=1)
            for report in sim_stats.poll_to_report.values():
                report.re_weight_uncertainty(criterion_to_prior,
                                             poll_end_ratings_count)

        poll_to_reports = defaultdict(list)
        for poll in range(plotted_poll_end):
            for sim_stats in sim_statses:
                if poll >= len(sim_stats.polls):
                    continue
                poll_to_reports[poll].append(sim_stats.poll_to_report[poll])

        poll_to_report_average = OrderedDict()
        for poll, reports in poll_to_reports.items():
            poll_to_report_average[poll] = UncertaintyReport.average_reports(
                    reports, ignore_rating=ignore_rating)

        features = None if ignore_rating else sim_statses[0].features
        return SimulationStats(plotted_poll_end,
                               sim_statses[0].question_count,
                               poll_to_report_average,
                               features)

    @classmethod
    def average_same_product_statses(cls, sim_statses):
        first_stats = sim_statses[0]

        poll_to_reports = OrderedDict()
        for stats in sim_statses:
            for poll, report in stats.poll_to_report.items():
                if poll not in poll_to_reports:
                    poll_to_reports[poll] = []
                poll_to_reports[poll].append(report)

        poll_to_report_average = OrderedDict()
        for poll, reports in poll_to_reports.items():
            poll_to_report_average[poll] = \
                    UncertaintyReport.average_same_product_reports(reports)

        return SimulationStats(
            first_stats.poll_count,
            first_stats.question_count,
            poll_to_report_average,
            first_stats.features,
            criterion_to_prior=first_stats.criterion_to_prior)
