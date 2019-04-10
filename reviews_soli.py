from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
import itertools
import random
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
        mixed_interface: bool, default=False,
            mix free-text review with active solicitation.
    """
    def __init__(self, pick, answer, optm_goal=None, baseline=False,
                 mixed_interface=False,
                 question_count=None,
                 ):
        self.pick = pick
        self.answer = answer
        self.optm_goal = optm_goal
        self.baseline = baseline
        self.mixed_interface = mixed_interface
        self.question_count = question_count

    @classmethod
    def build(cls, pick_mths=None, answer_mths=None, optm_goals=None,
              mixed_interface=False,
              ):
        """Build Solicitation Configuration.

        Baseline configs are always in the beginning of the return list.
        Args:
            pick_mths: list,
                default: ['pick_highest', 'pick_prob']
            answer_mths: list,
                default: ['answer_by_gen', 'answer_in_time_order']
            optm_goals: list, [UncertaintyMetric('expected_rating_var')]
                optimization goal
            mixed_interface: bool, default=False,
                mix free-text review with active solicitation.
        Returns:
            configs: list of SoliConfig
        """
        pick_mths = ['pick_highest', 'pick_prob'] if pick_mths is None else \
            pick_mths
        answer_mths = ['answer_by_gen', 'answer_in_time_order'] if \
            answer_mths is None else answer_mths
        optm_goals = [UncertaintyMetric('expected_rating_var')] if \
            optm_goals is None else optm_goals
        configs = []

        pick_baselines = ['pick_random', 'pick_least_count']
        for pick, answer in itertools.product(pick_baselines, answer_mths):
            configs.append(cls(
                pick, answer, baseline=True, mixed_interface=mixed_interface))

        for pick, answer, goal in itertools.product(pick_mths, answer_mths,
                                                    optm_goals):
            configs.append(cls(
                pick, answer, optm_goal=goal, mixed_interface=mixed_interface))

        return configs

    def pick_goal_str(self):
        config = self.pick
        if self.optm_goal and self.pick != "pick_by_user":
            config += '_' + self.optm_goal.show()
        if self.question_count:
            config += f'_{self.question_count}_aspect'
        if self.mixed_interface and self.answer == 'answer_by_gen_with_prob':
            config += f'_response_prob'
        return config

    def __repr__(self):
        return self.pick_goal_str() + '_' + self.answer

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
                and self.pick == other.pick \
                and self.answer == other.answer \
                and self.optm_goal == other.optm_goal \
                and self.baseline == other.baseline \
                and self.mixed_interface == other.mixed_interface \
                and self.question_count == other.question_count

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.pick, self.answer, str(self.optm_goal),
                     self.baseline, self.mixed_interface, self.question_count))

    def is_gen_answer(self):
        return self.answer.find('_gen') > -1


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
        self.stars = np.arange(1, self.star_rank + 1, 1)

        self.soli_config = soli_config
        self.metrics = metrics
        self.poll_count = poll_count
        self.question_count = soli_config.question_count or question_count

        self.seed_features = seed_features
        self.features = [Feature(i, feature_name)
                         for i, feature_name in enumerate(self.seed_features)]
        self.aspect_to_rated_prob = None if not dataset_profile \
            else {
                feature: dataset_profile.aspect_to_answer_prob[feature.name]
                for feature in self.features
                }
        self.feature_to_star_dist = Review.sample_star_dist(reviews,
                                                            self.features)
        self.star_dists = [self.feature_to_star_dist[feature.name]
                           for feature in self.features]
        self.feature_to_rating_generator = {
                feature: self.rating_generator(self.stars, star_dist)
                for feature, star_dist in self.feature_to_star_dist.items()
                }
        self.duplicate = True if kwargs['duplicate'] else False
        if self.duplicate:
            # 2 duplicate features' index in Review.dup_scenario_features
            self.duplicate_feature_idx = [-1, -2]

        if len(self.features) < self.question_count:
            raise ValueError("The number of features ({}) is smaller than the "
                             "number of questions to ask per poll ({})".format(
                                 len(self.features), self.question_count))

        co_ratings_prior = kwargs['co_ratings_prior'] \
            if 'co_ratings_prior' in kwargs else None
        # Keep track feature's uncertainty
        self.uncertainty_book = UncertaintyBook(
                self.star_rank,
                len(self.features),
                optm_goal=soli_config.optm_goal,
                rating_truth_dists=self.star_dists,
                aspect_to_rated_prob=self.aspect_to_rated_prob,
                co_ratings_prior=co_ratings_prior,
                )
        self.poll_to_report = OrderedDict()

        self.credible_bar = self.star_rank / 2

    def simulate(self):
        """Simulate the asking-aswering process."""
        for i in range(self.poll_count):
            self.uncertainty_book.refresh_uncertainty()

            # Ask up-to-k questions
            local_question_count = self.question_count
            if self.soli_config.answer == "answer_almost_real":
                next_review = self.reviews[0] if self.reviews else \
                        self.original_reviews[0]
                if self.question_count > len(next_review.features):
                    local_question_count = len(next_review.features)

            # Keep track in case of answer_in_time_order, i.e. get all answers
            # from a single real review
            self.num_waiting_answers = local_question_count
            already_picked_idx = []
            rated_features = []

            # In a mixed interface, get all features in free-text review first
            if self.soli_config.mixed_interface:
                if not self.reviews:
                    self.reviews = self.original_reviews.copy()
                rated_review = self.reviews.pop(0)
                for feature in self.features:
                    if feature.name in rated_review.feature_to_stars:
                        already_picked_idx.append(feature.idx)
                        rated_features.append((
                            feature,
                            rated_review.feature_to_stars[feature.name][0]))
                feature_left = max(0, len(self.features) -
                                   len(rated_review.feature_to_stars))
                local_question_count = min(feature_left, self.question_count)

            for q in range(local_question_count):
                self.uncertainty_book.refresh_uncertainty()
                picked_feature = self.__getattribute__(self.soli_config.pick)(
                        already_picked_idx)
                answered_star = self.__getattribute__(self.soli_config.answer)(
                        picked_feature) if picked_feature else None

                # When running out of feature in pick_by_user, or
                # pick_free_text_only
                if not picked_feature and not answered_star:
                    continue

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
                    rated_features.append((picked_feature, answered_star))
                else:
                    picked_feature.no_answer_count += 1

            self._update_feature_ratings(rated_features)
            self.poll_to_report[i] = \
                self.uncertainty_book.report_uncertainty(self.metrics)

        return SimulationStats(
                self.poll_count,
                self.question_count,
                self.poll_to_report,
                self.features,
                co_ratings=self.uncertainty_book.co_ratings,
                )

    def _update_feature_ratings(self, rated_features):
        if not rated_features:
            return
        for picked_feature, answered_star in rated_features:
            self.uncertainty_book.rate_feature(picked_feature, answered_star)

        # Update co-rating of 2 features
        for (feature_1, star_1), (feature_2, star_2) in \
                itertools.combinations_with_replacement(rated_features, 2):
            self.uncertainty_book.rate_2features(feature_1, star_1,
                                                 feature_2, star_2)

    @staticmethod
    def rating_generator(stars, star_dist):
        while True:
            star = np.random.choice(stars, p=star_dist)
            yield star

    def answer_by_gen(self, picked_feature):
        """Answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """
        return next(self.feature_to_rating_generator[picked_feature.name])

    def answer_by_gen_with_prob(self, picked_aspect):
        """Answer using sampling star's distribution of this product's reviews.

        Note: Answer with a probability
        Args:
            picked_aspect: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """
        star = self.answer_by_gen(picked_aspect)
        roll_dice = np.random.random_sample()
        if roll_dice <= self.aspect_to_rated_prob[picked_aspect]:
            return star
        else:
            return None

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

    def pick_free_text_only(self, already_picked_idx):
        """A dummy method picking nothing for free-text reviewing interface."""
        return

    def pick_like_bandit(self, already_picked_idx, exploit_rate=0.5):
        """Pick a feature like a multi-armed bandit.

        Inspired by multi-armed bandit problem, this method mimics
        the Epsilon-greedy strategy:
            * exploitation phase: in (1 - epsilon) number of trials, choose
            the highest lever.
            * exploration phase: in the other "epsilon" time, choose levers
            randomly.
        ref: https://en.wikipedia.org/wiki/Multi-armed_bandit
        """
        if exploit_rate < 0 or exploit_rate > 1:
            raise ValueError("exploit_rate = {} while it must be in "
                             "[0, 1]".format(exploit_rate))
        if random.random() < exploit_rate:
            return self.pick_highest(already_picked_idx)
        else:
            return self.pick_random(already_picked_idx)

    def pick_by_user(self, already_picked_idx):
        """Pick the first feature in the review (contain sorted features)

        Only use this method with answer method: answer_in_time_order or
        answer_almost_real. This is because these answer method will remove the
        first review in the row, thus guarantee having fresh new review to
        pick each time.

        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
                return None when running out of new features
        """
        # Run out of reviews, re-fetch from original reviews
        if not self.reviews:
            self.reviews = self.original_reviews.copy()

        for next_feature_name in self.reviews[0].ordered_features:
            for feature in self.features:
                if feature.name == next_feature_name and \
                        feature.idx not in already_picked_idx:
                    return feature
        # No new features
        return None

    def pick_highest_after_credible(self, already_picked_idx):
        """Optimize for credible threshold first, highest later.

        2 stages:
            * Choose lowest to pass credible interval test.
            * Then choose highest as usual.

        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
                return None when running out of new features
        """
        # Features that have credible interval smaller than a threshold
        credible_width = self.credible_bar
        z = 1.96       # confidence level 95%
        credible_feature_idx = np.where(
                self.uncertainty_book.uncertainties * z <= credible_width)[0]

        if len(credible_feature_idx) < len(self.seed_features):
            already_picked_idx.extend(credible_feature_idx)
            excluded_uncertainties = np.copy(
                    self.uncertainty_book.uncertainties)
            excluded_uncertainties[already_picked_idx] = float('inf')
            max_indices = np.where(
                    excluded_uncertainties == excluded_uncertainties.min())[0]
            return self.features[np.random.choice(max_indices)]
        else:
            if self.credible_bar > self.star_rank / 20:
                self.credible_bar -= self.star_rank / 20
            return self.pick_highest(already_picked_idx)
            # return self.pick_random(already_picked_idx)


class SimulationStats(object):
    """Resulting statistics of simulation
    Attributes:
        poll_count: int, how many time can ask customers
        question_count: int, number of question per customer
        poll_to_report: dict, poll -> UncertaintyReport
            poll starts from 0
        features (list): list of data_model.Feature
        criterion_to_prior: dict, from UncertaintyBook.prior
        co_ratings: 2d numpy array, from UncertaintyBook.co_ratings
    """
    def __init__(self, poll_count, question_count, poll_to_report, features,
                 co_ratings=None):
        self.poll_count = poll_count
        self.question_count = question_count
        self.poll_to_report = poll_to_report
        self.polls = list(self.poll_to_report.keys())
        self.uncertainty_reports = list(self.poll_to_report.values())

        self.features = features
        if features:
            self.no_answer_count = sum([feature.no_answer_count
                                        for feature in self.features])
        self.co_ratings = co_ratings

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

    def correlation_at(self, poll):
        return self.poll_to_report[poll].correlations

    @classmethod
    def average_statses(cls, sim_statses,
                        plotted_poll_end=100,
                        ignore_rating=False):
        """Averaging multiple product's sim stats.

        Args:
            sim_states: list of SimulationStats
            plotted_poll_end: int, default=100
                last poll to be plotted.
            ignore_rating: bool, default=False,
                ignore averaging rating of all products.
        """
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
            )
