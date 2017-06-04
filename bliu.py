import sys
import logging
from collections import defaultdict, OrderedDict

from data_model import Review
from edmunds import EdmundsReviewSolicitation
from anno_utils import import_bliu_dataset, match_datafiles


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

RATING_COUNT_MIN = 10
ASPECT_COUNT_MIN = 4

BLIU_POLARITY_TO_STAR = {-3: 1, -2: 2, -1: 3, 1: 4, 2: 5, 3: 6}


class BliuReview(Review):

    seed_features = []
    dup_scenario_features = []

    @classmethod
    def import_dataset(cls, dataset_path, star_rank=6, duplicate=False):
        """Preprocess Bliu dataset from a directory

        Args:
            dataset_path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            product_to_reviews: dict,
                product -> list of time-sorted BliuReview
        """
        product_to_reviews, _ = cls.preprocess_dataset(dataset_path,
                                                       duplicate=duplicate)
        return product_to_reviews

    @classmethod
    def preprocess_dataset(cls, dataset_path, star_rank=6, duplicate=False):
        """Preprocess Bliu dataset from a directory

        Args:
            dataset_path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            product_to_reviews: dict,
                product -> list of time-sorted BliuReview
            product_to_aspects: dict,
                product -> aspect_to_star_counts (dict)
                    aspect_to_star_counts: dict, aspect -> star_counts
                        star_counts: dict, star -> count
        """
        product_to_reviews = OrderedDict()
        product_to_aspects = OrderedDict()
        filepaths = match_datafiles(dataset_path, ".txt")
        for filepath in filepaths:
            product, aspect_to_polarity_counts, anno_reviews = \
                import_bliu_dataset(filepath)
            aspect_to_star_counts = clean_dataset(
                aspect_to_polarity_counts,
                RATING_COUNT_MIN,
                ASPECT_COUNT_MIN,
                BLIU_POLARITY_TO_STAR)

            if aspect_to_star_counts:
                product_to_reviews[product] = cls.convert_anno_review_to_bliu(
                    anno_reviews, aspect_to_star_counts.keys())
                product_to_aspects[product] = aspect_to_star_counts

        logger.debug("{} products with at least {} aspects, each aspect has "
                     "at least {} ratings".format(len(product_to_reviews),
                                                  ASPECT_COUNT_MIN,
                                                  RATING_COUNT_MIN))
        return (product_to_reviews, product_to_aspects)

    @classmethod
    def convert_anno_review_to_bliu(cls, anno_reviews, eligible_aspects):
        bliu_reviews = []
        for anno_review in anno_reviews:
            aspect_to_stars = defaultdict(list)
            for aspect_polarities in anno_review.polarized_sentences.values():
                for aspect, polarity in aspect_polarities:
                    if aspect in eligible_aspects:
                        aspect_to_stars[aspect].append(
                            BLIU_POLARITY_TO_STAR[polarity])
            if aspect_to_stars:
                bliu_reviews.append(cls(aspect_to_stars, star_rank=6))

        return bliu_reviews


class BliuReviewSolicitation(EdmundsReviewSolicitation):
    """Bliu reviews have a fixed set of features that make the
    simulation much simpler.
    """


def clean_dataset(aspect_to_polarity_counts,
                  star_count_min,
                  aspect_count_min,
                  polarity_to_star):
    """Clean dataset.

    Cleaning:
        1. filter aspects that has less than star_count_min
        2. mapping polarity to star
        3. bootstrap by adding 1 for every star
        4. return None if having less than 'aspect_count_min' aspects
    """
    aspect_to_polarity_counts_trimmed = {
        aspect: polarity_counts
        for aspect, polarity_counts in aspect_to_polarity_counts.items()
        if sum(polarity_counts.values()) >= star_count_min}

    if len(aspect_to_polarity_counts_trimmed) < aspect_count_min:
        return None

    aspect_to_star_counts = defaultdict(lambda: defaultdict(int))
    for aspect, polarity_counts in aspect_to_polarity_counts_trimmed.items():
        aspect_to_star_counts[aspect] = {
            polarity_to_star[polarity]: polarity_counts[polarity]
            for polarity in polarity_to_star.keys()}

    # bootstrap by adding 1 for every star
    for star_counts in aspect_to_star_counts.values():
        for star in star_counts.keys():
            star_counts[star] += 1

    return aspect_to_star_counts


if __name__ == '__main__':
    BliuReview.import_dataset(sys.argv[1])
