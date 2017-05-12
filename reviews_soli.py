from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod

import numpy as np

from data_model import Feature, Review
from uncertainty import UncertaintyBook, UncertaintyReport, UncertaintyMetric


class ReviewsSolicitation(ABC):
    """
    Attributes:
        reviews: list of data_model.Review
        poll_count: integer, default=-1 (i.e. len(reviews))
            how many times can ask customers
        question_count: int, default=1,
            Number of questions to ask a customer
        seed_features: list of features name (string), if any (default: [])
        criterion: string, default='expected_rating_var'
            uncertainty metric
        weighted: Boolean, default=False
            weighted uncertainty metric using prior/global ratings
        correlated: Boolean, default=False
            consider a feature's uncertainty using correlated features
        dataset_profile: SimulationStats object, default=None
            dataset's profile
        poll_to_cost: dict
            cost change over each time of asking questions """
    pick_methods = ['pick_highest_cost',
                    'pick_with_prob',
                    'pick_random',
                    'pick_least_count']

    answer_methods = ['answer_by_gen',
                      'answer_in_time_order']

    # pick_methods = ['pick_highest_cost', 'pick_least_count']
    # answer_methods = ['answer_by_gen']

    def __init__(self, reviews,
                 poll_count=20,
                 question_count=1,
                 seed_features=[],
                 criterion='expected_rating_var',
                 weighted=False,
                 correlated=False,
                 cor_norm_factor=1.0,
                 dataset_profile=None,
                 **kwargs):
        if len(reviews) < 1:
            raise ValueError('Empty or None reviews')
        self.original_reviews = reviews
        self.reviews = reviews.copy()
        self.feature_to_star_dist = Review.sample_star_dist(reviews)
        self.star_rank = reviews[0].star_rank

        self.poll_count = poll_count if poll_count <= len(reviews)\
            and poll_count > 0 else len(reviews)
        self.question_count = question_count
        self.criterion = criterion
        self.weighted = weighted
        self.correlated = correlated
        self.cor_norm_factor = cor_norm_factor

        self.seed_features = seed_features
        self.features = [Feature(i, feature_name)
                         for i, feature_name in enumerate(self.seed_features)]
        self.duplicate = True if kwargs['duplicate'] else False
        if self.duplicate:
            # 2 duplicate features' index in Review.dup_scenario_features
            self.duplicate_feature_idx = [1, 2]

        # Keep track feature's uncertainty
        self.uncertainty_book = UncertaintyBook(
                self.star_rank,
                len(self.features),
                metric=UncertaintyMetric(criterion,
                                         weighted=weighted,
                                         correlated=correlated,
                                         cor_norm_factor=cor_norm_factor),
                dataset_profile=dataset_profile)
        self.poll_to_cost = OrderedDict()
        self.poll_to_cost[0] = self.uncertainty_book.report_uncertainty()
        self.poll_to_ratings = OrderedDict()
        self.poll_to_ratings[0] = np.copy(self.uncertainty_book.ratings)

    def simulate(self, pick_method, answer_method):
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
                picked_feature = self.__getattribute__(pick_method)(
                        already_picked_idx)
                answered_star = self.__getattribute__(answer_method)(
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
                    # Update co-rating of 2 features
                    if rated_features:
                        for pre_rated_feature, pre_star in rated_features:
                            self.uncertainty_book.rate_2features(
                                    pre_rated_feature, pre_star,
                                    picked_feature, answered_star)
                    rated_features.append((picked_feature, answered_star))
                else:
                    picked_feature.no_answer_count += 1
            self.poll_to_cost[i + 1] = \
                self.uncertainty_book.report_uncertainty()
            self.poll_to_ratings[i + 1] = \
                np.copy(self.uncertainty_book.ratings)

        return SimulationStats(self.poll_count, self.question_count,
                               self.poll_to_cost, self.poll_to_ratings,
                               self.features, self.uncertainty_book.ratings)

    @abstractmethod
    def answer_by_gen(self, picked_feature):
        """Answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
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

    def pick_highest_cost(self, already_picked_idx):
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
            max_idx = np.argmax(excluded_uncertainties)
            return self.features[max_idx]

    def pick_with_prob(self, already_picked_idx):
        """Ask features with probability proportional to its cost,
        Args:
            already_picked_idx: list
                list of already picked feature indexes
        Returns:
            datamodel.Feature
        """
        weights = self.uncertainty_book.uncertainties / \
            np.sum(self.uncertainty_book.uncertainties)
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
            rating_counts[already_picked_idx] = -float('inf')
        max_idx = np.argmax(rating_counts)
        return self.features[max_idx]


class SimulationStats(object):
    """Resulting statistics of simulation
    Attributes:
        poll_count (int): how many time can ask customers
        question_count (int): number of question per customer
        poll_to_cost (dict): poll (int) -> UncertaintyReport
        polls: list of polls
        uncertainty_reports: list of uncertainty.UncertaintyReport
        final_features (list): list of data_model.Feature
        final_ratings: 2d np array, from UncertaintyBook.ratings
    """
    def __init__(self, poll_count, question_count,
                 poll_to_cost, poll_to_ratings,
                 final_features, final_ratings):
        self.poll_count = poll_count
        self.question_count = question_count
        self.poll_to_cost = poll_to_cost
        self.polls = list(self.poll_to_cost.keys())
        self.uncertainty_reports = list(self.poll_to_cost.values())

        self.poll_to_ratings = poll_to_ratings
        self.final_features = final_features
        self.no_answer_count = sum([feature.no_answer_count
                                    for feature in self.final_features])
        self.final_ratings = final_ratings

    def stats_str(self, message='', detail=False):
        stat_str = message + '\n'

        if detail:
            costs = ['{}: {:.3f}'.format(poll, cost)
                     for poll, cost in self.poll_to_cost.items()]
            stat_str += ', '.join(costs) + '\n'
        else:
            last_poll = len(self.poll_to_cost) - 1
            stat_str += 'Final cost after {} polls:\n{}\n'.format(
                last_poll, self.poll_to_cost[last_poll])

        stat_str += 'final_features: '
        for feature in self.final_features:
            stat_str += '{}={}   '.format(feature.name,
                                          self.final_ratings[feature.idx])
        stat_str += '/no_answer_count={}'.format(self.no_answer_count)
        return stat_str

    @classmethod
    def sim_stats_average(cls, sim_statses):
        """Averaging multiple product's sim stats."""
        poll_to_costs = defaultdict(list)
        poll_count = max([len(sim_stats.polls) for sim_stats in sim_statses])
        for poll in range(poll_count):
            for sim_stats in sim_statses:
                if poll >= len(sim_stats.polls):
                    continue
                poll_to_costs[poll].append(sim_stats.poll_to_cost[poll])

        poll_to_cost_average = OrderedDict()
        for poll, costs in poll_to_costs.items():
            poll_to_cost_average[poll] = UncertaintyReport.reports_average(
                    costs)

        return SimulationStats(len(poll_to_costs),
                               sim_statses[0].question_count,
                               poll_to_cost_average,
                               {}, [],
                               None)
