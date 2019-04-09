import sys
import logging
from collections import defaultdict
import json

from data_model import Review
from bliu import BliuReviewSolicitation


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

RATING_COUNT_MIN = 10
ASPECT_COUNT_MIN = 4

BLIU_POLARITY_TO_STAR = {-3: 1, -2: 2, -1: 3, 1: 4, 2: 5, 3: 6}


class AmzLaptopReview(Review):

    seed_features = []
    dup_scenario_features = []

    @classmethod
    def import_dataset(cls, dataset_path, star_rank=3, duplicate=False):
        """Preprocess Amazon Laptop dataset which is parsed already.

        ref: https://s3.amazonaws.com/amazon-reviews-pds/readme.html
        Dataset is in json format, e.g.
        {"B00T7XRGGC": {"R1HG5E8UO15FCC": {"time": 3}, "R185YF7PJBPYXX": {},...
        Args:
            dataset_path: string
            star_rank: int, e.g. 3 means 1, 2, and 3 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            product_to_reviews: dict,
                product -> list of time-sorted AmzLaptopReview
        """
        product_reviews_labeled = {}
        product_to_reviews = defaultdict(list)
        with open(dataset_path, 'r') as f:
            product_reviews_labeled = json.load(f)

        for product, reviews_labeled in product_reviews_labeled.items():
            aspect_to_count = defaultdict(int)
            for aspect_labeled in reviews_labeled.values():
                for aspect in aspect_labeled:
                    aspect_to_count[aspect] += 1

            for aspect_labeled in reviews_labeled.values():
                aspect_to_stars = defaultdict(list)
                for aspect, star in aspect_labeled.items():
                    aspect_to_stars[aspect] = [star]
                aspect_to_stars = {
                        aspect: stars
                        for aspect, stars in aspect_to_stars.items()
                        if aspect_to_count[aspect] >= RATING_COUNT_MIN
                        }

                product_to_reviews[product].append(
                        AmzLaptopReview(aspect_to_stars, star_rank=3))

        return product_to_reviews


class AmzLaptopReviewSolicitation(BliuReviewSolicitation):
    pass


if __name__ == '__main__':
    AmzLaptopReview.import_dataset(sys.argv[1])
