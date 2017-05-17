import csv
from collections import defaultdict
from dateutil.parser import parse

from data_model import Review


"""
Edmunds dataset comes with following fields
    id, make, model, year, styleId, author, title,
    reviewText, favoriteFeatures, suggestedImprovements, userRating,
    comfortRating:
        frontSeats, rearSeats, gettingInOut, noiseAndVibration, rideComfort
    interiorRating:
        cargoStorage, instrumentation, interiorDesign, logicOfControls,
        qualityOfMaterials
    performanceRating:
        acceleration, braking, roadHolding, shifting, steering
    reliabilityRating:
        repairFrequency, dealershipSupport, engine, transmission, electronics
    safetyRating:
        headlights, outwardVisibility, parkingAids, rainSnowTraction,
        activeSafety
    technologyRating:
        entertainment, navigation, bluetooth, usbPorts, climateControl
    valueRating:
        fuelEconomy, maintenanceCost, purchaseCost, resaleValue, warranty
    created, updated
"""


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
    def import_csv(cls, file_path, star_rank=5, duplicate=False):
        """Import Edmund dataset from csv file

        Args:
            file_path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            car_to_reviews: dict of car -> list of time-sorted EdmundsReview
        """
        car_to_reviews = defaultdict(list)
        car_to_rows = defaultdict(list)
        with open(file_path) as csvfile:
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
                feature_to_star = {feature: int(row[feature])
                                   for feature in cls.seed_features
                                   if row[feature]}
                car = Car(row["make"], row["model"],
                          row["year"], row["styleId"])
                car_to_reviews[car].append(
                    cls(feature_to_star, star_rank=star_rank))

        # Duplicate feature scenario: just for experimentation
        if duplicate:
            car_to_reviews = cls.create_duplicate_scenario(car_to_reviews)
        return car_to_reviews

    @classmethod
    def create_duplicate_scenario(cls, car_to_reviews):
        car_to_reviews_trim = defaultdict(list)
        for car, reviews in car_to_reviews.items():
            for review in reviews:
                dup_feature_to_star = {}
                for feature in cls.dup_scenario_features:
                    if feature in review.features:
                        dup_feature_to_star[feature] = review.feature_to_star[
                                feature]
                if cls.dup_scenario_features[-2] in review.features:
                    dup_feature_to_star[cls.dup_scenario_features[-1]] = \
                        review.feature_to_star[cls.dup_scenario_features[-2]]
                car_to_reviews_trim[car].append(cls(dup_feature_to_star,
                                                    review.star_rank))

        return car_to_reviews_trim


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
