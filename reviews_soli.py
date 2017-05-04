from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np

from data_model import Feature
from uncertainty import UncertaintyBook


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
        weighting: Boolean, default=False
            weighting uncertainty metric using prior/global ratings
        correlating: Boolean, default=False
            consider a feature's uncertainty using correlated features
        dataset_profile: SimulationStats object, default=None
            dataset's profile
        poll_to_cost: dict
            cost change over each time of asking questions
    """
    # pick_methods = ['pick_highest_cost',
                    # 'pick_with_prob',
                    # 'pick_random']

    answer_methods = ['answer_by_gen',
                      'answer_in_time_order']

    pick_methods = ['pick_highest_cost']
    # answer_methods = ['answer_by_gen']

    def __init__(self, reviews,
                 poll_count=20,
                 question_count=1,
                 seed_features=[],
                 criterion='expected_rating_var',
                 weighting=False,
                 correlating=False,
                 dataset_profile=None,
                 **kargs):
        if len(reviews) < 1:
            raise ValueError('Empty or None reviews')
        self.original_reviews = reviews
        self.reviews = reviews.copy()
        self.star_rank = reviews[0].star_rank

        self.poll_count = poll_count if poll_count <= len(reviews)\
            and poll_count > 0 else len(reviews)
        self.question_count = question_count
        self.criterion = criterion
        self.weighting = weighting
        self.correlating = correlating

        self.seed_features = seed_features
        self.features = [Feature(i, feature_name)
                         for i, feature_name in enumerate(self.seed_features)]

        # Keep track feature's uncertainty
        self.uncertainty_book = UncertaintyBook(
                self.star_rank,
                len(self.features),
                criterion=criterion,
                weighting=weighting,
                correlating=correlating,
                dataset_profile=dataset_profile,
                confidence_level=kargs['confidence_level'])
        self.poll_to_cost = OrderedDict()
        self.poll_to_cost[0] = self.uncertainty_book.report_uncertainty()

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

        return SimulationStats(self.poll_count, self.question_count,
                               self.poll_to_cost, self.features,
                               self.uncertainty_book)

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
            excluded_uncertainties[already_picked_idx] = np.NINF
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
            self.uncertainty_book.uncertainty_total()
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


class SimulationStats(object):
    """Resulting statistics of simulation
    Attributes:
        poll_count (int): how many time can ask customers
        question_count (int): number of question per customer
        poll_to_cost (dict): poll (int) -> cost
        final_features (list): list of data_model.Feature
        uncertainty_book: uncertainty.UncertaintyBook
    """
    def __init__(self, poll_count, question_count,
                 poll_to_cost, final_features,
                 uncertainty_book):
        self.poll_count = poll_count
        self.poll_to_cost = poll_to_cost
        self.final_features = final_features
        self.no_answer_count = sum([feature.no_answer_count
                                    for feature in self.final_features])
        self.uncertainty_book = uncertainty_book

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
            stat_str += '{}={}   '.format(
                    feature.name, self.uncertainty_book.ratings[feature.idx])
        stat_str += '/no_answer_count={}'.format(self.no_answer_count)
        return stat_str
