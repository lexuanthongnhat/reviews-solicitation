import sys
import random
from collections import OrderedDict

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


def simulate_reviews_soli(file_path, star_rank=5):
    """Simulate the asking process
    Args:
        file_path (string)
        star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
    """
    car_to_reviews = EdmundReview.import_csv(file_path, star_rank=star_rank)
    target = 1
    count = 0
    for car, car_reviews in car_to_reviews.items():
        if len(car_reviews) > 50:
            print("{} - {} reviews".format(car, len(car_reviews)))
            simulate_reviews_soli_per_product(car_reviews,
                    features_list=EdmundReview.main_features)

            count += 1
            if count == target:
                break

class EdmundReviewSolicitation(object):
    """
    Attributes:
        reviews: list of data_model.Review
        num_polls: integer of how many times can ask customers (default: -1,
            i.e. len(reviews))
        features_list: list of features name (string), if any (default: []) 
    """
    def __init__(self, reviews, num_polls=20, features_list=[]):
        self.reviews = reviews
        self.num_polls = num_polls if num_polls <= len(reviews)\
            and num_polls > 0 else len(reviews)
        self.features_list = features_list
        self.__init_simulation_stats()

    def __init_simulation_stats(self):
        self.step_to_cost = OrderedDict()
        self.name_to_feature = {}    # feature_name -> feature (Feature)
    
        # Initiate all features
        for feature_name in self.features_list:
            stars = [0] * self.reviews[0].star_rank
            self.name_to_feature[feature_name] = Feature(feature_name, stars)
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
            # Pick a Feature with highest cost to reduce
            picked_feature = sorted(self.name_to_feature.values(),
                                    reverse=True)[0]
            # print("Cost of {}: {:.3f}".format(picked_feature.name,
                # picked_feature.criterion()))
    
            answered_review = None
            for review in self.reviews:
                if picked_feature.name in review.feature_to_star.keys():
                    answered_review = review
                    answered_star = review.feature_to_star[picked_feature.name]
                    picked_feature.increase_star(answered_star, count=1) 
                    self.reviews.remove(answered_review)
                    break
    
            if not answered_review:
                picked_feature, answered_star = self.pick_random_feature()
                self.name_to_feature[picked_feature].increase_star(
                        answered_star, count=1)
    
            self.step_to_cost[i + 1] = Feature.product_cost(
                    self.name_to_feature.values())
    
        return SimulationStats(self.num_polls, self.step_to_cost,
                list(self.name_to_feature.values()))
    
    
    def ask_randomly(self):
        """Randomly ask questions to reduce the cost""" 
        # Iteratively picking feature to ask customers
        for i in range(self.num_polls): 
            picked_feature, answered_star = self.pick_random_feature()
            self.name_to_feature[picked_feature].increase_star(
                    answered_star, count=1)
            self.step_to_cost[i + 1] = Feature.product_cost(
                    self.name_to_feature.values())
    
        return SimulationStats(self.num_polls, self.step_to_cost,
                list(self.name_to_feature.values()))
    
    
    def pick_random_feature(self):
        """Returns tuple (picked_feature, answered_star)""" 
        answered_review = None
        while not answered_review or not answered_review.features \
                or not self.reviews: 
            answered_review = self.reviews.pop(
                    random.randint(0, len(self.reviews) - 1))
        if not answered_review:
            return None
    
        random_feature = random.choice(list(answered_review.features))
        answered_star = answered_review.feature_to_star[random_feature]
        return (random_feature, answered_star) 
    
    
def simulate_reviews_soli_per_product(reviews, features_list=[]):
    """
    Args:
        reviews (list)
    """
    num_polls = 20
    greedy_stats = EdmundReviewSolicitation(reviews.copy(), num_polls=num_polls,
            features_list=features_list).ask_greedily(answer_possibility=1)
    greedy_stats.print_stats('Greedily picking:')

    random_stats = EdmundReviewSolicitation(reviews.copy(), num_polls=num_polls,
            features_list=features_list).ask_randomly()
    random_stats.print_stats('Randomly picking:')


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
