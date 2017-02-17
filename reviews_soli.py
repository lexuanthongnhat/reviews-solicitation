import sys
import random
from collections import OrderedDict

import numpy as np

import data_model
from edmunds import EdmundReview
from data_model import Feature


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

    def print_stats(self, message=''):
        print('\n' + message) 
        costs = ['{}: {:.3f}'.format(step, cost)
                 for step, cost in self.step_to_cost.items()]
        print(', '.join(costs))
        print(self.final_features)


class EdmundReviewSolicitation(object):
    """
    Attributes:
        reviews: list of data_model.Review
        num_polls: integer of how many times can ask customers (default: -1,
            i.e. len(reviews))
        features_list: list of features name (string), if any (default: []) 
    """
    def __init__(self, reviews, num_polls=20, features_list=[],
                 criterion='weighted_sum_dirichlet_variances'):
        self.original_reviews = reviews
        self.reviews = reviews.copy()
        self.num_polls = num_polls if num_polls <= len(reviews)\
            and num_polls > 0 else len(reviews)
        self.features_list = features_list
        self.__init_simulation_stats(criterion=criterion)

    def __init_simulation_stats(self,
                                criterion='weighted_sum_dirichlet_variances'):
        self.step_to_cost = OrderedDict()
        self.name_to_feature = {}    # feature_name -> feature (Feature)
    
        # Initiate all features
        for feature_name in self.features_list:
            stars = [0] * self.reviews[0].star_rank
            self.name_to_feature[feature_name] = Feature(feature_name, stars,
                    criterion=criterion)
        self.step_to_cost[0] = Feature.product_cost(
                self.name_to_feature.values())
    
    def ask_greedily(self, answer_possibility=1):
        """Greedily ask questions to reduce the cost 
        Args: 
            answer_possibility: float number [0, 1] representing the possibility
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
                self.reviews.extend(self.original_reviews.copy()) 
            self.step_to_cost[i + 1] = Feature.product_cost(
                    self.name_to_feature.values())

        return SimulationStats(self.num_polls, self.step_to_cost,
                list(self.name_to_feature.values()))

    def ask_greedily_then_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
                pick_func='pick_highest_cost_feature')

    def ask_greedily_prob_then_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
                pick_func='pick_feature_with_prob')

    def ask_randomly_then_answer_in_time_order(self):
        return self.ask_then_answer_in_time_order(
                pick_func='pick_random_feature') 

    def ask_then_answer_in_time_order(self, pick_func='pick_random_feature'):
        """Ask questions using pick_func, answer in time order."""
        for i in range(self.num_polls):
            picked_feature = self.__getattribute__(pick_func)()
            answered_review = self.reviews.pop(0) # earliest review

            if picked_feature.name in answered_review.feature_to_star.keys(): 
                answered_star = answered_review.feature_to_star[
                        picked_feature.name]
                picked_feature.increase_star(answered_star, count=1)

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


def simulate_reviews_soli(file_path, star_rank=5,
                          criterion='weighted_sum_dirichlet_variances'):
    """Simulate the asking process
    Args:
        file_path (string)
        star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
    """
    car_to_reviews = EdmundReview.import_csv(file_path, star_rank=star_rank)
    car_to_reviews = {key: value for key, value in car_to_reviews.items()
                            if len(value) >= 970}
    car_to_result_stats = {}
    for car, car_reviews in car_to_reviews.items():
        # print("{} - {} reviews".format(car, len(car_reviews)))
        car_to_result_stats[car] = simulate_reviews_soli_per_product(
                car_reviews, num_polls=100,
                features_list=EdmundReview.main_features,
                criterion=criterion)

    return (car_to_reviews, car_to_result_stats)


def simulate_reviews_soli_per_product(reviews, num_polls=-1, features_list=[],
        criterion='weighted_sum_dirichlet_variances'):
    """
    Args:
        reviews: list of Review
        num_polls: integer of the number of times (reviews) to ask customers
            (default: -1, means the number of reviews available for simulation)
        features_list: list of product's features, if know upfront
    Returns: 
        (greedy_stats, random_stats): tuple of SimulationStats
    """
    greedy_stats = EdmundReviewSolicitation(reviews.copy(),
            num_polls=num_polls,
            features_list=features_list,
            criterion=criterion)\
                    .ask_greedily(answer_possibility=1)
    greedy_stats.print_stats('Greedily picking:')

    time_greedy_stats = EdmundReviewSolicitation(reviews.copy(),
            num_polls=num_polls,
            features_list=features_list,
            criterion=criterion)\
                    .ask_greedily_then_answer_in_time_order() 
    time_greedy_stats.print_stats('Greedily picking with time order:')

    time_prob_greedy_stats = EdmundReviewSolicitation(reviews.copy(),
            num_polls=num_polls,
            features_list=features_list,
            criterion=criterion)\
                    .ask_greedily_prob_then_answer_in_time_order()
    time_prob_greedy_stats.print_stats(
            'Greedily prob picking with time order:')

    random_stats = EdmundReviewSolicitation(reviews.copy(),
            num_polls=num_polls,
            features_list=features_list,
            criterion=criterion)\
                    .ask_randomly_then_answer_in_time_order()
    random_stats.print_stats('Randomly picking:') 

    return (greedy_stats, time_greedy_stats, time_prob_greedy_stats,
            random_stats)

   
def main(file_path):
    reviews = data_model.import_csv(file_path) 
    print("# reviews: {}".format(len(reviews)))

    print("# reviews that have minor features rating: {}".format(
            data_model.count_reviews_with_minor(reviews)))

    print("main feature rating count: {}".format(
            data_model.count_feature_ratings(reviews))) 
    print("minor feature rating count: {}".format(
            data_model.count_feature_ratings(reviews, 'minor_features')))

    car_to_reviews = data_model.group_reviews_into_cars(reviews)
    count = 0
    for car, car_reviews in car_to_reviews.items():
        if len(car_reviews) < 50:
            continue
        print(car)
        print("main feature rating count: {}".format(
                data_model.count_feature_ratings(car_reviews))) 
        print("minor feature rating count: {}".format(
                data_model.count_feature_ratings(car_reviews, 'minor_features')))
        if count > 2:
            break
        count += 1
    print('# cars: {}'.format(len(car_to_reviews.keys())))


if __name__ == '__main__':
    file_path = sys.argv[1]
    if file_path.endswith('.csv'):
        simulate_reviews_soli(file_path)
