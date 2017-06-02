import sys
import os
import logging
from collections import defaultdict, OrderedDict

from data_model import Review
from edmunds import EdmundsReviewSolicitation
from anno_utils import import_bliu_dataset, AnnoReview


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

RATING_COUNT_MIN = 10
ASPECT_COUNT_MIN = 4

SENTIMENT_TO_STAR = {-3: 1, -2: 2, -1: 3, 1: 4, 2: 5, 3: 6}


class BliuReview(Review):

    seed_features = []
    dup_scenario_features = []

    @classmethod
    def import_dataset(cls, path, star_rank=6, duplicate=False):
        """Import Bliu dataset from a directory

        Args:
            path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            product_to_reviews: dict,
                product -> list of time-sorted BliuReview
        """
        product_to_reviews = OrderedDict()
        filepaths = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.txt') \
                        and not filename.startswith('Readme'):
                    filepaths.append(os.path.join(dirpath, filename))

        for filepath in filepaths:
            product, anno_reviews = import_bliu_dataset(filepath)
            aspect_to_sentiment_count = AnnoReview.aggregate_aspects(
                    anno_reviews)
            aspect_to_sentiment_count = _clean_dataset(
                    aspect_to_sentiment_count)

            if len(aspect_to_sentiment_count) >= ASPECT_COUNT_MIN:
                logger.debug(product)
                product_to_reviews[product] = \
                    cls.convert_anno_review_to_bliu(
                            anno_reviews, aspect_to_sentiment_count.keys())
            else:
                logger.debug("Product with less than {} aspects: {}".format(
                    ASPECT_COUNT_MIN, product))

        logger.debug("{} products with aspects of at least {} ratings".format(
            len(product_to_reviews), RATING_COUNT_MIN))
        return product_to_reviews

    @classmethod
    def import_dataset_with_aspects(cls, path, star_rank=6, duplicate=False):
        """Import Bliu dataset from a directory
        Args:
            path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
        """
        product_to_aspect_sentiments = defaultdict(list)
        filepaths = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.txt') \
                        and not filename.startswith('Readme'):
                    filepaths.append(os.path.join(dirpath, filename))

        for filepath in filepaths:
            product, anno_reviews = import_bliu_dataset(filepath)
            aspect_to_sentiment_count = AnnoReview.aggregate_aspects(
                    anno_reviews)
            aspect_to_sentiment_count = _clean_dataset(
                    aspect_to_sentiment_count)

            if len(aspect_to_sentiment_count) >= ASPECT_COUNT_MIN:
                logger.debug(product)
                product_to_aspect_sentiments[product] = \
                    aspect_to_sentiment_count
            else:
                logger.debug("Product with less than {} aspects: {}".format(
                    ASPECT_COUNT_MIN, product))

        logger.debug("{} products with aspects of at least {} ratings".format(
            len(product_to_aspect_sentiments), RATING_COUNT_MIN))
        return product_to_aspect_sentiments

    @classmethod
    def convert_anno_review_to_bliu(cls, anno_reviews, eligible_aspects):
        bliu_reviews = []
        for anno_review in anno_reviews:
            feature_to_stars = defaultdict(list)
            for aspect_sentiments in anno_review.sentence_to_aspects.values():
                for aspect, sentiment in aspect_sentiments:
                    if aspect in eligible_aspects:
                        feature_to_stars[aspect].append(
                                SENTIMENT_TO_STAR[sentiment])
            if feature_to_stars:
                bliu_reviews.append(cls(feature_to_stars, star_rank=6))

        return bliu_reviews

    @classmethod
    def create_mock_reviews(cls, aspect_to_sentiment_count, star_rank=6):
        reviews = []
        for aspect, sentiment_count in aspect_to_sentiment_count.items():
            for sentiment, count in sentiment_count.items():
                for i in range(count):
                    reviews.append(cls({aspect: sentiment},
                                       star_rank=star_rank))
        return reviews


class BliuReviewSolicitation(EdmundsReviewSolicitation):
    """Bliu reviews have a fixed set of features that make the
    simulation much simpler.
    """


def _clean_dataset(aspect_to_sentiment_count):
    """Clean dataset.

    Cleaning:
        1. filter aspects that has less than RATING_COUNT_MIN
        2. map sentiment (-3, -2, -1, 1, 2, 3) to star (1, 2, 3, 4, 5, 6)
        3. bootstrap by adding 1 for every star
    """
    aspect_to_sentiment_count_cleaned = _filter_rare_aspects(
            aspect_to_sentiment_count, RATING_COUNT_MIN)
    aspect_to_sentiment_count_cleaned = _map_sentiment(
            aspect_to_sentiment_count_cleaned)
    # bootstrap by adding 1 for every sentiment
    for sentiment_count in aspect_to_sentiment_count_cleaned.values():
        for sentiment in sentiment_count.keys():
            sentiment_count[sentiment] += 1

    return aspect_to_sentiment_count_cleaned


def _filter_rare_aspects(aspect_to_sentiment_count, rating_count_min):
    aspect_to_sentiment_count_filtered = {}
    for aspect, sentiment_count in aspect_to_sentiment_count.items():
        if sum(sentiment_count.values()) >= rating_count_min:
            aspect_to_sentiment_count_filtered[aspect] = sentiment_count
    return aspect_to_sentiment_count_filtered


def _map_sentiment(aspect_to_sentiment_count):
    aspect_to_sentiment_count_mapped = {}
    for aspect, sentiment_count in aspect_to_sentiment_count.items():
        sentiment_count_mapped = defaultdict(int)
        sentiment_count_mapped[1] = sentiment_count[-3]
        sentiment_count_mapped[2] = sentiment_count[-2]
        sentiment_count_mapped[3] = sentiment_count[-1]
        sentiment_count_mapped[4] = sentiment_count[1]
        sentiment_count_mapped[5] = sentiment_count[2]
        sentiment_count_mapped[6] = sentiment_count[3]
        aspect_to_sentiment_count_mapped[aspect] = sentiment_count_mapped
    return aspect_to_sentiment_count_mapped


if __name__ == '__main__':
    BliuReview.import_dataset(sys.argv[1])
