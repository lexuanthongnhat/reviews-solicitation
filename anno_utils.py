from os import path, walk
import logging
import argparse
import re
from collections import OrderedDict, defaultdict

from lxml import etree


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


class AnnoReview(object):
    """Annotated Review

    Attributes:
        idx: int
        title: str
        polarized_sentences: dict,
            sentence (str) -> tuple of (aspect, polarity)
        aspects: set
            aspects are from polarized_sentences
    """
    def __init__(self, idx, title):
        self.idx = idx
        self.title = title
        self.polarized_sentences = OrderedDict()
        self.aspects = set([])

    def add_sentence(self, sentence, polarized_aspects):
        """Add sentence and its aspect, sentiments.

        Args:
            sentence: str
            polarized_aspects: list of tuple (aspect, polarity)
        """
        self.polarized_sentences[sentence] = polarized_aspects
        for aspect, _ in polarized_aspects:
            self.aspects.add(aspect)

    @classmethod
    def aggregate_aspects(cls, anno_reviews):
        """Aggregate set of annotated reviews based on aspects.

        Args:
            anno_reviews: list of AnnoReview
        Returns:
            aspect_to_polarity_counts: dict,
                aspect -> {polarity -> count}
        """
        aspect_to_polarity_counts = defaultdict(lambda: defaultdict(int))
        for review in anno_reviews:
            for polarized_aspects in review.polarized_sentences.values():
                for aspect, polarity in polarized_aspects:
                    aspect_to_polarity_counts[aspect][polarity] += 1

        return aspect_to_polarity_counts


def import_bliu_dataset(filepath):
    """
    Returns:
        product, aspect_to_polarity_counts, reviews: tuple
            product: str, also means product's name
            aspect_to_polarity_counts: dict,
                aspect -> {polarity -> count}
            reviews: list of AnnoReview
    """
    REVIEW_TITLE_TAG = '[t]'
    SENTENCE_TAG = '##'

    reviews = []
    with open(filepath, errors='ignore') as f:
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
            polarized_aspect_str, sentence = line.split(SENTENCE_TAG)
            polarized_aspects = []
            if polarized_aspect_str:
                for polarized_aspect in polarized_aspect_str.split(','):
                    if not polarized_aspect.strip():
                        continue
                    aspect = polarized_aspect[:polarized_aspect.find('[')]\
                        .strip()
                    polarity_match = re.search(r'([\+\-]\d)\]',
                                               polarized_aspect)
                    if polarity_match:
                        polarity = int(polarity_match.group(1))
                        polarized_aspects.append((aspect, polarity))
                    else:
                        continue

            if not review:      # missing review title tag
                review = AnnoReview(len(reviews), 'auto-created review title')

            review.add_sentence(sentence, polarized_aspects)

        # flush the last review
        if review and (not reviews or review.idx != reviews[-1].idx):
            reviews.append(review)

    aspect_to_polarity_counts = AnnoReview.aggregate_aspects(reviews)
    product = path.splitext(path.basename(filepath))[0]
    return (product, aspect_to_polarity_counts, reviews)


def import_semeval_dataset(filepath):
    """Import annotated dataset of SemEval 2014 Task 4.

    File is in xml format
    Returns:
        product: str,
        aspect_to_polarity_counts: dict
    """
    POLARIZED_STR_TO_POLARITY = {'positive': 1, 'neutral': 0, 'negative': -1}
    aspect_to_polarity_counts = defaultdict(lambda: defaultdict(int))

    tree = etree.parse(filepath)
    for aspect_term_element in tree.getiterator(tag='aspectTerm'):
        aspect = aspect_term_element.attrib['term'].strip()
        polarity_str = aspect_term_element.attrib['polarity_str'].strip()

        if polarity_str not in POLARIZED_STR_TO_POLARITY:
            continue
        polarity = POLARIZED_STR_TO_POLARITY[polarity_str]
        aspect_to_polarity_counts[aspect][polarity] += 1

    for aspect, polarity_to_count in aspect_to_polarity_counts.items():
        count = sum([count for count in polarity_to_count.values()])
        if count >= 20 and len(polarity_to_count) >= 5:
            logger.debug('{:16s}: {}'.format(aspect, polarity_to_count))
    logger.debug('There are {} aspects'.format(len(aspect_to_polarity_counts)))

    product = path.splitext(path.basename(filepath))[0]
    return (product, aspect_to_polarity_counts)


def match_datafiles(dataset_dir, end_substr=".txt"):
    """Get data's file path in directory with expected end sub string.

    Args:
        dataset_dir: str,
            directory of dataset
        end_substr: str, default=".txt"
            sub-string of file path ending, used to match the file
    Returns:
        filepaths: list of str
    """
    filepaths = []
    for dirpath, dirnames, filenames in walk(dataset_dir):
        for filename in filenames:
            if filename.endswith(end_substr) \
                    and not filename.startswith('Readme'):
                filepaths.append(path.join(dirpath, filename))
    logger.debug("Matched files in dir '{}' are: {}".format(
        dataset_dir, ", ".join(filepaths)))

    return filepaths


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
