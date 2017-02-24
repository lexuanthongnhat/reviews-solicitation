import scipy.stats as stats
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class Review(ABC):
    """Abstract class for review
    """
    @property
    @abstractmethod
    def seed_features(self):
        return "Please implement this property!"

    def __init__(self, feature_to_star, star_rank=5):
        self.feature_to_star = feature_to_star
        self.star_rank = star_rank

        self.features = self.feature_to_star.keys()

    def star_of_feature(self, feature):
        return self.feature_to_star[feature]

    def __repr__(self):
        return repr(self.feature_to_star)

    @classmethod
    @abstractmethod
    def import_csv(cls, file_path, star_rank=5):
        """
        Args:
            file_path (string)
            star_rank (int): e.g. 5 means 1, 2, 3, 4 and 5 stars system
        Returns:
            product_to_reviews (dict): product -> list of Reviews
        """

    @classmethod
    def sample_star_dist(cls, reviews):
        """Sampling a set of reviews for the distribution of stars.
        Args:
            reviews: list of Review
        Returns:
            star_dist: np.array of star's distribution
        """
        if not reviews:
            return None

        star_to_count = {i: 0 for i in range(1, reviews[0].star_rank + 1)}
        for review in reviews:
            for feature, star in review.feature_to_star.items():
                star_to_count[star] += 1
        ordered_counts = [star_to_count[star]
                          for star in range(1, reviews[0].star_rank + 1)]
        star_dist = np.array(ordered_counts) / sum(ordered_counts)
        return star_dist

    @classmethod
    def profile_dataset(cls, product_to_reviews):
        """Profiling dataset's properties.
        Args:
            product_to_reviews (dict): product -> list of Reviews
        """
        star_rank = list(product_to_reviews.values())[0][0].star_rank
        num_products = len(product_to_reviews)
        num_reviews_per_product = [len(reviews)
                                   for reviews in product_to_reviews.values()]
        num_reviews = sum(num_reviews_per_product)

        # histogram of the number of reviews per product
        num_reviews_to_num_products = defaultdict(int)
        for product, reviews in product_to_reviews.items():
            num_reviews_to_num_products[len(reviews)] += 1

        # histogram of the number of review per feature
        feature_to_num_reviews = defaultdict(int)
        for product, reviews in product_to_reviews.items():
            for review in reviews:
                for feature in review.feature_to_star.keys():
                    feature_to_num_reviews[feature] += 1

        feature_to_ave_num_reviews_per_product = {
            feature: num_reviews / num_products
            for feature, num_reviews in feature_to_num_reviews.items()}

        # estimate m, V_0
        ave_num_feature_ratings_per_product = np.average(np.array(
            list(feature_to_ave_num_reviews_per_product.values())))

        sum_variances = []
        for product, reviews in product_to_reviews.items():
            name_to_feature = {}
            for review in reviews:
                for feature_name, star in review.feature_to_star.items():
                    if feature_name not in name_to_feature:
                        name_to_feature[feature_name] = Feature(
                            feature_name,
                            [0] * star_rank,
                            criterion='sum_dirichlet_variances')

                    name_to_feature[feature_name].increase_star(star)
            sum_variances.extend([feature.criterion()
                                  for feature in name_to_feature.values()])
        global_ave_sum_variances = np.average(np.array(sum_variances))

        return DatasetProfile(star_rank, num_products, num_reviews,
                              num_reviews_per_product,
                              num_reviews_to_num_products,
                              feature_to_num_reviews,
                              feature_to_ave_num_reviews_per_product,
                              ave_num_feature_ratings_per_product,
                              global_ave_sum_variances)


class DatasetProfile(object):

    def __init__(self,
                 star_rank, num_products, num_reviews,
                 num_reviews_per_product,
                 num_reviews_to_num_products,
                 feature_to_num_reviews,
                 feature_to_ave_num_reviews_per_product,
                 ave_num_feature_ratings_per_product,
                 global_ave_sum_variances):
        self.star_rank = star_rank
        self.num_products = num_products
        self.num_reviews = num_reviews
        self.num_reviews_per_product = num_reviews_per_product
        self.num_reviews_to_num_products = num_reviews_to_num_products
        self.feature_to_num_reviews = feature_to_num_reviews
        self.feature_to_ave_num_reviews_per_product = \
            feature_to_ave_num_reviews_per_product
        self.ave_num_feature_ratings_per_product = \
            ave_num_feature_ratings_per_product
        self.global_ave_sum_variances = global_ave_sum_variances

    def __str__(self):
        profile = 'Dataset of {} stars, with {} products and {} reviews\n'\
            .format(self.star_rank, self.num_products, self.num_reviews)
        profile += '# average feature ratings per product = {:.3f}\n'.format(
            self.ave_num_feature_ratings_per_product)
        profile += 'Global average sum of variance of feature = {:.3f}'.format(
            self.global_ave_sum_variances)
        return profile

    def full_str(self):
        profile = self.__str__()
        profile += '# reviews -> # products: {}\n'.format(
            self.num_reviews_to_num_products)
        profile += 'feature -> # reviews: {}\n'.format(
            self.feature_to_num_reviews)
        profile += 'feature -> # average reviews per product: {}\n'.format(
            self.feature_to_ave_num_reviews_per_product)
        return profile


class Feature(object):
    """
    Represent a product feature/attribute/aspect
    The Feature's comparision is based on feature's cost. E.g.,
        'feature1 > feature2' means cost of feature 1 > cost of feature 2

    Attributes:
        name: string
        ratings: list
            e.g., [3, 0, 6] corresponds to 3, 0, 6 ratings for 1, 2, 3 stars
            respectively. Require 0 with no ratings for that star.
        criterion: string, default='weighted_sum_dirichlet_variances'
            cost need to be optimized
        prior_count: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
        prior_cost: string, default=None
            only when criterion='weighted_sum_dirichlet_variances'
    """
    criteria = ['weighted_sum_dirichlet_variances',
                'sum_dirichlet_variances']

    def __init__(self, name, ratings,
                 criterion='weighted_sum_dirichlet_variances',
                 prior_count=None,
                 prior_cost=None):

        self.name = name
        self.ratings = ratings
        self.star_rank = len(self.ratings)
        self.no_answer_count = 0
        self.criterion = self.__getattribute__(criterion)
        self.prior_count = prior_count
        self.prior_cost = prior_cost

    def increase_star(self, star, count=1):
        if star < 1 or star > len(self.ratings):
            raise IndexError
        self.ratings[star - 1] += count

    def get_num_ratings(self, star):
        if star > 0 and star <= len(self.ratings):
            return self.ratings[star - 1]
        else:
            raise IndexError

    def __repr__(self):
        return "{}: {}/no_answer={}".format(self.name, self.ratings,
                                            self.no_answer_count)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.criterion() < other.criterion()

    def __le__(self, other):
        return self.criterion() <= other.criterion()

    def __gt__(self, other):
        return self.criterion() > other.criterion()

    def __ge__(self, other):
        return self.criterion() >= other.criterion()

    def sum_dirichlet_variances(self):
        alphas = [star + 1 for star in self.ratings]
        return sum(stats.dirichlet.var(alphas))

    def weighted_sum_dirichlet_variances(self):
        if not self.prior_cost or not self.prior_count:
            self.prior_count = self.star_rank
            self.prior_cost = sum(stats.dirichlet.var([1] * self.prior_count))

        weighted_sum = (sum(self.ratings) * self.sum_dirichlet_variances()
                        + self.prior_count * self.prior_cost) / \
            (sum(self.ratings) + self.prior_count)
        return weighted_sum

    @classmethod
    def product_cost(cls, features):
        return sum([feature.criterion() for feature in features])
