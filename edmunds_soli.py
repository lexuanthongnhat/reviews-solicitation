import random

import numpy as np

from reviews_soli import ReviewsSolicitation, SimulationStats
from data_model import Feature


class EdmundsReviewSolicitation(ReviewsSolicitation):

    def ask_greedily_answer_mostly(self):
        """Greedily ask questions to reduce the cost
        Args:
            answer_possibility: float [0, 1] representing the possibility
                that customers answer a question
        """
        # Iteratively picking feature to ask customers
        for i in range(self.num_polls):
            picked_feature = self.pick_highest_cost_feature()

            answered_review = None
            for review in self.reviews:
                if picked_feature.name in review.feature_to_star.keys():
                    answered_review = review
                    answered_star = review.feature_to_star[picked_feature.name]
                    picked_feature.increase_star(answered_star, count=1)
                    self.reviews.remove(answered_review)
                    break

            if not answered_review:
                picked_feature.no_answer_count += 1
                self.reviews.extend(self.original_reviews.copy())
            self.step_to_cost[i + 1] = Feature.product_cost(
                    self.name_to_feature.values())

        return SimulationStats(self.num_polls, self.step_to_cost,
                               list(self.name_to_feature.values()))

    def ask_greedily_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
                pick_func='pick_highest_cost_feature')

    def ask_greedily_prob_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
                pick_func='pick_feature_with_prob')

    def ask_randomly_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
                pick_func='pick_random_feature')

    def ask_then_answer_in_time_order(self, pick_func='pick_random_feature'):
        """Ask questions using pick_func, answer in time order."""
        for i in range(self.num_polls):
            picked_feature = self.__getattribute__(pick_func)()
            answered_review = self.reviews.pop(0)   # earliest review

            if picked_feature.name in answered_review.feature_to_star.keys():
                answered_star = answered_review.feature_to_star[
                        picked_feature.name]
                picked_feature.increase_star(answered_star, count=1)
            else:
                picked_feature.no_answer_count += 1

            self.step_to_cost[i + 1] = Feature.product_cost(
                    self.name_to_feature.values())

        return SimulationStats(self.num_polls, self.step_to_cost,
                               list(self.name_to_feature.values()))

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
        """Pick a feature with highest cost, break tie arbitrarily.
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
