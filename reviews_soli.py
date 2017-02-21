from collections import OrderedDict
from abc import ABC, abstractmethod

from data_model import Feature


class ReviewsSolicitation(ABC):
    """
    Attributes:
        reviews: list of data_model.Review
        num_polls: integer of how many times can ask customers (default: -1,
            i.e. len(reviews))
        seed_features: list of features name (string), if any (default: [])
    """
    ask_methods = ['ask_greedily_answer_mostly',
                   'ask_greedily_answer_in_time_order',
                   'ask_greedily_prob_answer_in_time_order',
                   'ask_randomly_answer_in_time_order']

    def __init__(self, reviews, num_polls=20, seed_features=[],
                 criterion='weighted_sum_dirichlet_variances'):
        self.original_reviews = reviews
        self.reviews = reviews.copy()
        self.num_polls = num_polls if num_polls <= len(reviews)\
            and num_polls > 0 else len(reviews)
        self.seed_features = seed_features
        self.__init_simulation_stats(criterion=criterion)

    def __init_simulation_stats(self,
                                criterion='weighted_sum_dirichlet_variances'):
        self.step_to_cost = OrderedDict()
        self.name_to_feature = {}    # feature_name -> feature (Feature)

        # Initiate all features
        for feature_name in self.seed_features:
            stars = [0] * self.reviews[0].star_rank
            self.name_to_feature[feature_name] = Feature(feature_name, stars,
                                                         criterion=criterion)
        self.step_to_cost[0] = Feature.product_cost(
                self.name_to_feature.values())

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


class SimulationStats(object):
    """Resulting statistics of simulation
    Attributes:
        num_polls (int): how many time can ask customers
        step_to_cost (dict): step (int) -> cost
        final_features (list): list of data_model.Feature
    """
    def __init__(self, num_polls, step_to_cost, final_features):
        self.num_polls = num_polls
        self.step_to_cost = step_to_cost
        self.final_features = list(final_features)
        self.no_answer_count = sum([feature.no_answer_count
                                    for feature in self.final_features])

    def stats_str(self, message=''):
        stat_str = message + '\n'
        costs = ['{}: {:.3f}'.format(step, cost)
                 for step, cost in self.step_to_cost.items()]
        stat_str += ', '.join(costs) + '\n'
        stat_str += 'final_features: {}'.format(self.final_features)
        stat_str += '/no_answer_count={}'.format(self.no_answer_count)
        return stat_str
