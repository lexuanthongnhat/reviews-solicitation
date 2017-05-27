from os import path
import logging
import argparse
import re
from collections import OrderedDict, defaultdict

from lxml import etree


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


class AnnoReview(object):
    """Annotated Review

    Attributes:
        idx: int
        title: str
        sentence_to_aspects: dict,
            sentence (str) -> tuple of (aspect, sentiment)
        aspects: set
            aspects are from sentence_to_aspects
    """
    def __init__(self, idx, title):
        self.idx = idx
        self.title = title
        self.sentence_to_aspects = OrderedDict()
        self.aspects = set([])

    def add_sentence(self, sentence, aspect_sentiments):
        """Add sentence and its aspect, sentiments.

        Args:
            sentence: str
            aspect_sentiments: list of tuple (aspect, sentiment)
        """
        self.sentence_to_aspects[sentence] = aspect_sentiments
        for aspect, _ in aspect_sentiments:
            self.aspects.add(aspect)

    @classmethod
    def aggregate_aspects(cls, anno_reviews):
        """Aggregate set of annotated reviews based on aspects.

        Args:
            anno_reviews: list of AnnoReview
        Returns:
            aspect_to_sentiment_count: dict,
                aspect -> {sentiment -> count}
        """
        aspect_to_sentiment_count = defaultdict(lambda: defaultdict(int))
        for review in anno_reviews:
            for aspect_sentiments in review.sentence_to_aspects.values():
                for aspect, sentiment in aspect_sentiments:
                    aspect_to_sentiment_count[aspect][sentiment] += 1

        return aspect_to_sentiment_count


def import_bliu_dataset(file_path):
    """
    Returns:
        file_name, reviews, aspect_to_sentiment_count: tuple
            file_name: str, also means product's name
            reviews: list of AnnoReview
    """
    REVIEW_TITLE_TAG = '[t]'
    SENTENCE_TAG = '##'

    reviews = []
    with open(file_path, errors='ignore') as f:
        review = None
        for line in f:
            if line.startswith(REVIEW_TITLE_TAG):
                if review:      # flush previous review
                    reviews.append(review)
                review = AnnoReview(len(reviews),
                                    line[len(REVIEW_TITLE_TAG):].strip())
                continue

            if line.find(SENTENCE_TAG) < 0:
                continue

            # None-title line: supposed to be sentences
            aspect_sentiment_str, sentence = line.split(SENTENCE_TAG)
            aspect_sentiments = []
            if aspect_sentiment_str:
                for annotated_aspect in aspect_sentiment_str.split(','):
                    if not annotated_aspect.strip():
                        continue
                    aspect = annotated_aspect[:annotated_aspect.find('[')]\
                        .strip()
                    sentiment_match = re.search(r'[\+\-]\d+', annotated_aspect)
                    if sentiment_match:
                        sentiment = int(sentiment_match.group(0))
                        aspect_sentiments.append((aspect, sentiment))
                    else:
                        continue

            if not review:      # missing review title tag
                review = AnnoReview(len(reviews), 'auto-created review title')

            review.add_sentence(sentence, aspect_sentiments)

        # flush the last review
        if review and (not reviews or review.idx != reviews[-1].idx):
            reviews.append(review)

    file_name = path.splitext(path.basename(file_path))[0]
    return (file_name, reviews)


def import_semeval_dataset(file_path):
    """Import annotated dataset of SemEval 2014 Task 4.

    File is in xml format
    """
    POLARITY_TO_SENTIMENT = {'positive': 3, 'neutral': 2, 'negative': 1}
    aspect_to_sentiment_count = defaultdict(lambda: defaultdict(int))

    tree = etree.parse(file_path)
    for aspect_term_element in tree.getiterator(tag='aspectTerm'):
        aspect = aspect_term_element.attrib['term'].strip()
        polarity = aspect_term_element.attrib['polarity'].strip()

        if polarity not in POLARITY_TO_SENTIMENT:
            continue
        sentiment = POLARITY_TO_SENTIMENT[polarity]
        aspect_to_sentiment_count[aspect][sentiment] += 1

    for aspect, sentiment_count in aspect_to_sentiment_count.items():
        count = sum([count for count in sentiment_count.values()])
        if count >= 20 and len(sentiment_count) >= 5:
            logger.debug('{:16s}: {}'.format(aspect, sentiment_count))
    logger.debug('There are {} aspects'.format(len(aspect_to_sentiment_count)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Product Aspect Extractor")
    parser.add_argument("--input", help="dataset input path")
    parser.add_argument("--dataset", help="dataset name (different reader)")
    parser.add_argument(
        "--loglevel", default="WARN",
        help="log level (default='WARN')")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug("args: {}".format(args))

    if args.dataset == "bliu":
        import_bliu_dataset(args.input)
    elif args.dataset == "semeval":
        import_semeval_dataset(args.input)
