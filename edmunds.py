import sys
import inspect
import csv
from collections import defaultdict
from collections import OrderedDict
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
    minor_features = [fture for ftures in features_dict.values()
                      for fture in ftures]
    overall_rating = 'userRating'

    @classmethod
    def import_csv(cls, file_path, star_rank=5):
        """Import Edmund dataset from csv file

        Returns:
            car_to_reviews: dict of car -> list of time-sorted reviews
        """
        car_to_reviews = defaultdict(list)
        car_to_rows = defaultdict(list)
        with open(file_path) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Filter out rows with erroneous rating
                stars = [int(row[feature]) for feature in cls.main_features
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
                                   for feature in cls.main_features
                                   if row[feature]}
                car = Car(row["make"], row["model"],
                          row["year"], row["styleId"])
                car_to_reviews[car].append(
                    cls(feature_to_star, star_rank=star_rank))
        return car_to_reviews


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


def group_reviews_into_cars(reviews):
    """
    Args:
        reviews(list)
    Returns:
        car_to_reviews(dict): Car -> list of reviews
    """

    car_to_reviews = defaultdict(list)
    for rev in reviews:
        car = Car(rev["make"], rev["model"], rev["year"], rev["styleId"])
        car_to_reviews[car].append(rev)

    return car_to_reviews


def count_feature_ratings(reviews, feature_category='main_features'):
    """Count reviews that have a specific number of rated minor features
    Args:
        reviews(list)
        feature_category: 'main_features', or 'minor_features' - attributes
        of EdmundsReview
    Returns:
        num_ratings_to_count(dict)
    """

    num_ratings_to_count = defaultdict(int)
    attrs = inspect.getmembers(EdmundsReview)
    features = list(filter(lambda attr: attr[0] == feature_category,
                           attrs))[0][1]
    for review in reviews:
        num_ratings = sum(
            map(lambda f: 1 if review[f] else 0, features))
        num_ratings_to_count[num_ratings] += 1

    return OrderedDict(num_ratings_to_count)


def count_reviews_with_minor(reviews):
    count_minor = 0
    for row in reviews:
        for minor_feature in EdmundsReview.minor_features:
            if row[minor_feature]:
                count_minor += 1
                break
    return count_minor


def import_csv(file_path):
    reviews = []
    with open(file_path) as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            reviews.append(row)
    return reviews


def main(file_path):

    reviews = import_csv(file_path)
    print("# reviews: {}".format(len(reviews)))

    print("# reviews that have minor features rating: {}".format(
        count_reviews_with_minor(reviews)))

    print("main feature rating count: {}".format(
        count_feature_ratings(reviews)))
    print("minor feature rating count: {}".format(
        count_feature_ratings(reviews, 'minor_features')))

    car_to_reviews = group_reviews_into_cars(reviews)
    count = 0
    for car, car_reviews in car_to_reviews.items():
        if len(car_reviews) < 50:
            continue
        print(car)
        print("main feature rating count: {}".format(
            count_feature_ratings(car_reviews)))
        print("minor feature rating count: {}".format(
            count_feature_ratings(car_reviews, 'minor_features')))
        if count > 2:
            break
        count += 1
    print('# cars: {}'.format(len(car_to_reviews.keys())))


if __name__ == "__main__":
    file_path = sys.argv[1]
    if file_path.endswith('.csv'):
        main(file_path)
