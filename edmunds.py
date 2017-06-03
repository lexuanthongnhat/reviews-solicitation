import csv
from collections import defaultdict
from dateutil.parser import parse

import numpy as np

from data_model import Review
from reviews_soli import ReviewsSolicitation


class EdmundsReview(Review):

    features_dict = {
        'performanceRating': ['acceleration', 'braking', 'roadHolding',
                              'shifting', 'steering'],
        'comfortRating': ['frontSeats', 'rearSeats', 'gettingInOut',
                          'noiseAndVibration', 'rideComfort'],
        'interiorRating': ['cargoStorage', 'instrumentation',
                           'interiorDesign', 'logicOfControls',
                           'qualityOfMaterials'],
        'safetyRating': ['outwardVisibility', 'parkingAids',
                         'rainSnowTraction', 'activeSafety'],
        'technologyRating': ['entertainment', 'navigation', 'bluetooth',
                             'usbPorts', 'climateControl'],
        'reliabilityRating': ['repairFrequency', 'dealershipSupport',
                              'engine', 'transmission', 'electronics'],
        'valueRating': ['fuelEconomy', 'maintenanceCost', 'purchaseCost',
                        'resaleValue', 'warranty']
    }
    main_features = features_dict.keys()
    seed_features = main_features
    non_rare_features = ['performanceRating', 'comfortRating',
                         'interiorRating', 'reliabilityRating', 'valueRating']
    dup_scenario_features = ['performanceRating', 'interiorRating',
                             'safetyRating', 'technologyRating',
                             'reliabilityRating', 'valueRating',
                             'comfortRating', 'comfortStar']
    minor_features = [fture for ftures in features_dict.values()
                      for fture in ftures]
    overall_rating = 'userRating'

    @classmethod
    def import_dataset(cls, dataset_path, star_rank=5, duplicate=False):
        """Import Edmund dataset from csv file

        Args:
            dataset_path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            car_to_reviews: dict of car -> list of time-sorted EdmundsReview
        """
        car_to_reviews = defaultdict(list)
        car_to_rows = defaultdict(list)
        with open(dataset_path) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Filter out rows with erroneous rating
                stars = [int(row[feature]) for feature in cls.seed_features
                         if row[feature]]
                is_erroneous = any([star for star in stars
                                    if star > star_rank or star <= 0])
                if not is_erroneous:
                    car = Car(row["make"], row["model"],
                              row["year"], row["styleId"])
                    car_to_rows[car].append(row)

        # Sort rows of a car using created dates
        for car, rows in car_to_rows.items():
            time_sorted_rows = sorted(rows,
                                      key=lambda row: parse(row['created']))

            for row in time_sorted_rows:
                feature_to_stars = {
                        feature: [int(row[feature])]
                        for feature in cls.seed_features if row[feature]}
                car = Car(row["make"], row["model"],
                          row["year"], row["styleId"])
                car_to_reviews[car].append(
                    cls(feature_to_stars, star_rank=star_rank))

        # Duplicate feature scenario: just for experimentation
        if duplicate:
            car_to_reviews = cls.create_duplicate_scenario(car_to_reviews)
        return car_to_reviews

    @classmethod
    def create_duplicate_scenario(cls, car_to_reviews):
        car_to_reviews_trim = defaultdict(list)
        for car, reviews in car_to_reviews.items():
            for review in reviews:
                dup_feature_to_stars = {}
                for feature in cls.dup_scenario_features:
                    if feature in review.features:
                        dup_feature_to_stars[feature] = \
                                review.feature_to_stars[feature]
                if cls.dup_scenario_features[-2] in review.features:
                    dup_feature_to_stars[cls.dup_scenario_features[-1]] = \
                        review.feature_to_stars[cls.dup_scenario_features[-2]]
                car_to_reviews_trim[car].append(cls(dup_feature_to_stars,
                                                    review.star_rank))

        return car_to_reviews_trim


class EdmundsReviewSolicitation(ReviewsSolicitation):
    """Edmunds reviews have a fixed set of features that make the
    simulation much simpler.
    """

    def answer_by_gen(self, picked_feature):
        """Answer using sampling star's distribution of this product's reviews.
        Note: Always have answer
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """
        star_dist = self.feature_to_star_dist[picked_feature.name]
        stars = np.arange(1, self.star_rank + 1, 1)
        answered_star = np.random.choice(stars, p=star_dist)
        return answered_star

    def answer_in_time_order(self, picked_feature):
        """Answer using real reviews sorted in time order.
        Args:
            picked_feature: datamodel.Feature, returned by pick_method
        Returns:
            answered_star: int
        """
        # Run out of reviews, re-fetch from original reviews
        if not self.reviews:
            self.reviews = self.original_reviews.copy()

        answered_review = self.reviews[0]   # earliest review
        answered_star = None
        if picked_feature.name in answered_review.features:
            answered_star = np.random.choice(
                    answered_review.feature_to_stars[picked_feature.name])
        self.num_waiting_answers -= 1

        if self.num_waiting_answers <= 0:
            self.reviews.pop(0)
        return answered_star


class Car(object):
    """Car object
    At this moment, a car is identified by make, model and year (not style_id)
    Attrs:
        make
        model
        year
        style_id
    """

    def __init__(self, make, model, year, style_id):
        self.make = make
        self.model = model
        self.year = year
        self.style_id = style_id

    def __repr__(self):
        return "{}-{}-{}".format(self.make, self.model, self.year)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.make == other.make \
            and self.model == other.model and self.year == other.year

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.make, self.model, self.year))
