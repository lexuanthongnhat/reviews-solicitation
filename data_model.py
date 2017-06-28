from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class Review(ABC):
    """Abstract class for review
    Attributes:
        feature_to_stars: dict, feature name -> star
        star_rank: int (default=5), number of star levels
    """
    @property
    @classmethod
    @abstractmethod
    def seed_features(self):
        return "Please implement this property!"

    @property
    @classmethod
    @abstractmethod
    def dup_scenario_features(self):
        return "Please implement this property!"

    def __init__(self, feature_to_stars, star_rank=5, ordered_features=None):
        self.feature_to_stars = feature_to_stars
        self.star_rank = star_rank

        self.features = self.feature_to_stars.keys()
        self.ordered_features = ordered_features

    def __repr__(self):
        return repr(self.feature_to_stars)

    @classmethod
    @abstractmethod
    def import_dataset(cls, dataset_path, star_rank=5, duplicate=False):
        """
        Args:
            dataset_path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            product_to_reviews (dict): product -> list of Reviews
        """

    @classmethod
    def sample_star_dist(cls, reviews, features):
        """Sampling a set of reviews for the distribution of stars.
        Args:
            reviews: list of Review
            features: list of Feature
        Returns:
            feature_to_star_dist: dict: feature's name -> np.array of
                star's distribution
        """
        if not reviews:
            return None

        star_rank = reviews[0].star_rank
        feature_to_star_counts = defaultdict(lambda: np.ones(star_rank))
        for review in reviews:
            for feature, stars in review.feature_to_stars.items():
                for star in stars:
                    feature_to_star_counts[feature][star - 1] += 1

        feature_to_star_dist = {
                feature: star_counts / np.sum(star_counts)
                for feature, star_counts in feature_to_star_counts.items()}

        # Uniform dist for unknown features
        for feature in features:
            if feature.name not in feature_to_star_dist:
                feature_to_star_dist[feature.name] = \
                        np.ones(star_rank) / star_rank
        return feature_to_star_dist

    @classmethod
    def probe_dataset(cls, product_to_reviews):
        """Profiling dataset's properties.
        Args:
            product_to_reviews (dict): product -> list of Reviews
        """
        star_rank = list(product_to_reviews.values())[0][0].star_rank
        product_count = len(product_to_reviews)
        per_product_review_counts = [len(reviews) for reviews
                                     in product_to_reviews.values()]
        review_count = sum(per_product_review_counts)

        # histogram of the number of reviews per product
        review_count_to_product_count = defaultdict(int)
        for product, reviews in product_to_reviews.items():
            review_count_to_product_count[len(reviews)] += 1

        # histogram of the number of review per feature
        feature_to_review_count = defaultdict(int)
        for product, reviews in product_to_reviews.items():
            for review in reviews:
                for feature in review.feature_to_stars.keys():
                    feature_to_review_count[feature] += 1

        feature_to_review_count_average = {
                feature: review_count / product_count
                for feature, review_count in feature_to_review_count.items()}

        # Average of feature rating's count per product
        feature_rating_count_average = np.average(np.array(
            list(feature_to_review_count_average.values())))

        # Prior rating distribution
        feature_ratings = []
        product_to_feature_ratings = {}
        for product, reviews in product_to_reviews.items():
            local_feature_to_ratings = defaultdict(lambda: np.zeros(star_rank))
            for review in reviews:
                for feature, stars in review.feature_to_stars.items():
                    for star in stars:
                        local_feature_to_ratings[feature][star - 1] += 1

            feature_ratings.extend(local_feature_to_ratings.values())
            product_to_feature_ratings[product] = local_feature_to_ratings

        return DatasetProfile(star_rank, product_count, review_count,
                              per_product_review_counts,
                              review_count_to_product_count,
                              feature_to_review_count,
                              feature_to_review_count_average,
                              feature_rating_count_average,
                              feature_ratings,
                              product_to_feature_ratings)


class DatasetProfile(object):

    def __init__(self,
                 star_rank, product_count, review_count,
                 per_product_review_counts,
                 review_count_to_product_count,
                 feature_to_review_count,
                 feature_to_review_count_average,
                 feature_rating_count_average,
                 feature_ratings,
                 product_to_feature_ratings):
        self.star_rank = star_rank
        self.product_count = product_count
        self.review_count = review_count
        self.per_product_review_counts = per_product_review_counts
        self.review_count_to_product_count = review_count_to_product_count
        self.feature_to_review_count = feature_to_review_count
        self.feature_to_review_count_average = feature_to_review_count_average
        self.feature_rating_count_average = feature_rating_count_average
        self.feature_ratings = feature_ratings
        self.product_to_feature_ratings = product_to_feature_ratings

    def __str__(self):
        profile = 'Dataset of {} stars, with {} products and {} reviews\n'\
            .format(self.star_rank, self.product_count, self.review_count)
        profile += 'average of feature_rating_count/product = {:.3f}\n'.format(
            self.feature_rating_count_average)
        return profile

    def full_str(self):
        profile = self.__str__()
        profile += '# reviews -> # products: {}\n'.format(
            self.review_count_to_product_count)
        profile += 'feature -> review count: {}\n'.format(
            self.feature_to_review_count)
        profile += 'feature -> average of review_count/product: {}\n'.format(
            self.feature_to_review_count_average)
        return profile


class Feature(object):
    """
    Represent a product feature/attribute/aspect

    Attributes:
        idx: int, starting from 0
            can be used to look up in uncertainty.UncertaintyBook
        name: string
            must be unique, used in Review object
    """
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
        self.no_answer_count = 0

    def __repr__(self):
        return "{}: no_answer={}".format(self.name, self.no_answer_count)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)
