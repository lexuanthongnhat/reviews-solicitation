import logging
from collections import OrderedDict

from data_model import Review
from edmunds import EdmundsReviewSolicitation
from anno_utils import import_semeval_dataset, match_datafiles
from bliu import clean_dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

POLARITY_TO_STAR = {'positive': 3, 'neutral': 2, 'negative': 1}
RATING_COUNT_MIN = 10
ASPECT_COUNT_MIN = 5


class SemevalReview(Review):

    seed_features = []
    dup_scenario_features = []

    @classmethod
    def import_dataset(cls, dataset_path, star_rank=3, duplicate=False):
        """Import Semeval dataset from a directory

        Args:
            dataset_path: string
            star_rank: int, e.g. 5 means 1, 2, 3, 4 and 5 stars system
            duplicate: bool, default=False, duplicate experiment scenario
        Returns:
            product_to_reviews: dict,
                product -> list of SemevalReview
        """
        product_to_reviews = OrderedDict()
        filepaths = match_datafiles(dataset_path, "_v2.xml")

        for filepath in filepaths:
            product, aspect_to_polarity_counts = import_semeval_dataset(
                filepath)
            aspect_to_star_counts = clean_dataset(
                aspect_to_polarity_counts,
                RATING_COUNT_MIN, ASPECT_COUNT_MIN, POLARITY_TO_STAR)

            if aspect_to_star_counts:
                product_to_reviews[product] = \
                    cls.create_mock_reviews(aspect_to_star_counts)

        logger.debug("{} products with at least {} aspects, each aspect has "
                     " at least {} ratings".format(len(product_to_reviews),
                                                   ASPECT_COUNT_MIN,
                                                   RATING_COUNT_MIN))
        return product_to_reviews

    @classmethod
    def create_mock_reviews(cls, aspect_to_star_counts, star_rank=3):
        reviews = []
        for aspect, star_to_count in aspect_to_star_counts.items():
            for star, count in star_to_count.items():
                for i in range(count):
                    reviews.append(cls({aspect: [star]},
                                       star_rank=star_rank))
        return reviews


class SemevalReviewSolicitation(EdmundsReviewSolicitation):
    """Bliu reviews have a fixed set of features that make the
    simulation much simpler.
    """

    def answer_in_time_order(self, picked_feature):
        raise NotImplementedError("Bliu reviews don't support this method!")
