import sys
import os
import logging
from collections import defaultdict

from data_model import Review
from edmunds import EdmundsReviewSolicitation
from anno_utils import import_bliu_dataset, AnnoReview


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


class BliuReview(Review):

    seed_features = []
    dup_scenario_features = []

    @classmethod
    def import_dataset(cls, path, star_rank=6, duplicate=False):
        """Import Edmund dataset from csv file

        Args:
            path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            car_to_reviews: dict of car -> list of time-sorted EdmundsReview
        """
        RATING_COUNT_MIN = 10
        ASPECT_COUNT_MIN = 3

        product_to_reviews = defaultdict(list)
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

            aspect_to_sentiment_count = _filter_rare_aspects(
                    aspect_to_sentiment_count, RATING_COUNT_MIN)
            aspect_to_sentiment_count = _map_sentiment(
                    aspect_to_sentiment_count)

            # bootstrap
            for sentiment_count in aspect_to_sentiment_count.values():
                for sentiment in sentiment_count.keys():
                    sentiment_count[sentiment] += 1

            if len(aspect_to_sentiment_count) >= ASPECT_COUNT_MIN:
                logger.debug(product)
                product_to_reviews[product] = cls.create_mock_reviews(
                        aspect_to_sentiment_count, star_rank=star_rank)
            else:
                logger.debug("Product with less than {} aspects: {}".format(
                    ASPECT_COUNT_MIN, product))

        logger.debug("{} products with aspects of at least {} ratings".format(
            len(product_to_reviews), RATING_COUNT_MIN))
        return product_to_reviews

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
    def answer_in_time_order(self, picked_feature):
        raise NotImplementedError("Bliu reviews don't support this method!")


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
