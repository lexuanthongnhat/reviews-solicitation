from collections import OrderedDict
import random
from abc import ABC, abstractmethod

import numpy as np

from data_model import Feature


class ReviewsSolicitation(ABC):
    """
    Attributes:
        reviews: list of data_model.Review
        num_polls: integer, default=-1 (i.e. len(reviews))
            how many times can ask customers
        num_questions: int, default=1,
            Number of questions to ask a customer
        seed_features: list of features name (string), if any (default: [])
        criterion: string
            the definition of cost. Possible values are
            'weighted_sum_dirichlet_variances', 'sum_dirichlet_variances'
        prior_count: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
        prior_cost: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
        step_to_cost: dict
            cost change over each time of asking questions
        name_to_feature: dict
            feature's name -> data_model.Feature
    """
    ask_methods = ['ask_greedily_answer_by_gen',
                   'ask_greedily_prob_answer_by_gen',
                   'ask_randomly_answer_by_gen',
                   'ask_greedily_answer_mostly',
                   'ask_greedily_answer_in_time_order',
                   'ask_greedily_prob_answer_in_time_order',
                   'ask_randomly_answer_in_time_order']

    def __init__(self, reviews,
                 num_polls=20,
                 num_questions=1,
                 seed_features=[],
                 criterion='weighted_sum_dirichlet_variances',
                 prior_count=None,
                 prior_cost=None):
        self.original_reviews = reviews
        self.reviews = reviews.copy()
        self.num_polls = num_polls if num_polls <= len(reviews)\
            and num_polls > 0 else len(reviews)
        self.num_questions = num_questions
        self.seed_features = seed_features
        self.__init_simulation_stats(criterion=criterion,
                                     prior_count=prior_count,
                                     prior_cost=prior_cost)

    def __init_simulation_stats(self,
                                criterion='weighted_sum_dirichlet_variances',
                                prior_count=None,
                                prior_cost=None):
        self.step_to_cost = OrderedDict()
        self.name_to_feature = {}    # feature_name -> feature (Feature)

        # Initiate all features
        for feature_name in self.seed_features:
            stars = [0] * self.reviews[0].star_rank
            self.name_to_feature[feature_name] = Feature(
                    feature_name, stars, criterion=criterion,
                    prior_count=prior_count, prior_cost=prior_cost)
        self.step_to_cost[0] = Feature.product_cost(
            self.name_to_feature.values())

    def simulate(self, ask_method):
        """Simulate the asking-aswering process."""
        for i in range(self.num_polls):
            # Keep track in case of answer_in_time_order, i.e. get all answers
            # from a single real review
            self.num_waiting_answers = self.num_questions
            for q in range(self.num_questions):
                picked_feature, answered_star = self.__getattribute__(
                        ask_method)()
                # Update ratings, rating's uncertainty
                if answered_star:
                    picked_feature.increase_star(answered_star, count=1)
                else:
                    picked_feature.no_answer_count += 1
            self.step_to_cost[i + 1] = Feature.product_cost(
                self.name_to_feature.values())

        return SimulationStats(self.num_polls,
                               self.num_questions,
                               self.step_to_cost,
                               list(self.name_to_feature.values()))

    @abstractmethod
    def ask_greedily_answer_by_gen(self):
        """Greedily ask question, answer using sampling star's distribution
        of this product's reviews
        Note: Always have answer
        """

    @abstractmethod
    def ask_greedily_prob_answer_by_gen(self):
        """Ask question with probability proportional to feature's cost,
        answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        """

    @abstractmethod
    def ask_randomly_answer_by_gen(self):
        """Ask question randomly, answer using sampling star's distribution
        of this product's reviews.
        Note: Always have answer
        """

    @abstractmethod
    def ask_greedily_answer_mostly(self):
        """Ask 'the most costly' features, then answer by reviews that
        contains the features.

        If no review has the features, consider as no answer. Then extend
        the current reviews set by original reviews set."""

    @abstractmethod
    def ask_greedily_answer_in_time_order(self):
        """Ask 'the most costly' features, then answer by reviews ordered
        chronologically.

        If the review doesn't have that feature, consider as no answer and
        remove review from the set."""

    @abstractmethod
    def ask_greedily_prob_answer_in_time_order(self):
        """Ask features with probability proportional to its cost,
        then answer by reviews ordered chronologically.

        If the review doesn't have that feature, consider as no answer and
        remove review from the set."""

    @abstractmethod
    def ask_randomly_answer_in_time_order(self):
        """Ask features randomly, then answer by reviews ordered
        chronologically.

        If the review doesn't have that feature, consider as no answer and
        remove review from the set."""

    def pick_highest_cost_feature(self):
        """Pick a feature with highest cost, break tie arbitrarily.
        Returns:
            datamodel.Feature
        """
        sorted_features = sorted(self.name_to_feature.values(), reverse=True)
        highest_cost = sorted_features[0].criterion()
        picked_features = [feature for feature in sorted_features
                           if feature.criterion() == highest_cost]
        return random.choice(picked_features)

    def pick_feature_with_prob(self):
        """Ask features with probability proportional to its cost,
        Returns:
            datamodel.Feature
        """
        features = list(self.name_to_feature.values())
        costs = np.array([feature.criterion() for feature in features])
        weights = costs / np.sum(costs)
        return np.random.choice(features, p=weights)

    def pick_random_feature(self):
        """Pick a feature randomly
        Returns:
            datamodel.Feature
        """
        return random.choice(list(self.name_to_feature.values()))


class SimulationStats(object):
    """Resulting statistics of simulation
    Attributes:
        num_polls (int): how many time can ask customers
        num_questions (int): number of question per customer
        step_to_cost (dict): step (int) -> cost
        final_features (list): list of data_model.Feature
    """

    def __init__(self, num_polls, num_questions, step_to_cost, final_features):
        self.num_polls = num_polls
        self.step_to_cost = step_to_cost
        self.final_features = list(final_features)
        self.no_answer_count = sum([feature.no_answer_count
                                    for feature in self.final_features])

    def stats_str(self, message='', detail=False):
        stat_str = message + '\n'

        if detail:
            costs = ['{}: {:.3f}'.format(step, cost)
                     for step, cost in self.step_to_cost.items()]
            stat_str += ', '.join(costs) + '\n'
        else:
            last_poll = len(self.step_to_cost) - 1
            stat_str += 'Final cost after {} polls: {:.3f}\n'.format(
                last_poll, self.step_to_cost[last_poll])

        stat_str += 'final_features: {}'.format(self.final_features)
        stat_str += '/no_answer_count={}'.format(self.no_answer_count)
        return stat_str
