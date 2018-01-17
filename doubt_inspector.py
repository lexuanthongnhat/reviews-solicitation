import argparse
from collections import namedtuple
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from synthetic import SyntheticReview, random_alpha_beta, beta_binomial
from reviews_soli import ReviewsSolicitation as ReviewSoli
import uncertainty


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

Doubt = namedtuple("Doubt", ["doubt", "doubt_drop"])


class DoubtInspector:
    """Explore the properties of rating doubt/uncertainty."""

    def __init__(self, doubt_metrics,
                 aspect_count=5, randomize_rating=False, star_rank=5,
                 run_count=10, user_count=300,
                 output="drop.pdf"):
        self.doubt_metrics = doubt_metrics
        self.aspect_count = aspect_count
        self.randomize_rating = randomize_rating
        self.star_rank = star_rank
        self.run_count = run_count
        self.user_count = user_count
        self.stars = np.arange(1, self.star_rank + 1, 1)
        self.output = output

    def plot_doubt_drop(self):
        aspect_profiles = self._build_aspect_profile()
        rating_generators = [
                ReviewSoli.rating_generator(self.stars, aspect_profile)
                for aspect_profile in aspect_profiles]

        # 3d array: aspect * metric * user_count
        aspect_doubts = []
        aspect_doubt_drops = []
        for generator in rating_generators:
            doubt = self._sample_doubt_drop(generator)
            aspect_doubts.append(doubt.doubt)
            aspect_doubt_drops.append(doubt.doubt_drop)

        self._plot_doubt_trend(aspect_profiles,
                               np.array(aspect_doubt_drops),
                               np.array(aspect_doubts))

    def _build_aspect_profile(self):
        """Build rating distribution (profile) of an aspect."""
        aspect_profiles = []
        for i in range(self.aspect_count):
            alpha, beta = random_alpha_beta() if self.randomize_rating else \
                          SyntheticReview.BETA_BINO_PARAMS[i, :]
            star_dist = np.array(beta_binomial(alpha, beta,
                                               self.star_rank - 1))
            aspect_profiles.append(star_dist)
        return aspect_profiles

    def _sample_doubt_drop(self, star_generator):
        """Sampling multiple time to record the change of rating.

        Return:
            doubt_drop: 2d narray,
                        shape: (#doubt_metric, #user_count)
        """
        metric_count = len(self.doubt_metrics)
        multi_run_doubt_drops = np.zeros((self.run_count, self.user_count,
                                          metric_count))
        multi_run_doubts = np.zeros((self.run_count, self.user_count,
                                     metric_count))

        # Each run is a full experiment that ask a number of users
        for run in range(self.run_count):
            per_run_drops = np.zeros((self.user_count, metric_count))
            per_run_doubts = np.zeros((self.user_count, metric_count))

            ratings = np.ones(self.star_rank)
            prev_doubts = np.array([doubt_metric(ratings)
                                   for doubt_metric in self.doubt_metrics])
            for user in range(self.user_count):
                star = next(star_generator)
                ratings[star - 1] += 1
                curr_doubts = np.array([doubt_metric(ratings)
                                       for doubt_metric in self.doubt_metrics])
                per_run_doubts[user, :] = prev_doubts

                per_user_drops = prev_doubts - curr_doubts
                prev_doubts = curr_doubts
                per_run_drops[user, :] = per_user_drops

            multi_run_doubt_drops[run, :, :] = per_run_drops
            multi_run_doubts[run, :, :] = per_run_doubts

        doubt_drops = np.mean(multi_run_doubt_drops, axis=0)
        doubts = np.mean(multi_run_doubts, axis=0)
        return Doubt(doubts.T, doubt_drops.T)

    def _plot_doubt_trend(self, aspect_profiles,
                          aspect_doubt_drops, aspect_doubts):
        """Plot doubt and its change with different metric.

        Args:
            aspect_profiles: list of aspect's rating distribution (profile)
            aspect_doubt_drops, aspect_doubts:
                both are 3d arrays: aspect * metric * user_count
        """
        np.set_printoptions(precision=1)
        metric_count = len(self.doubt_metrics)
        fig, axes = plt.subplots(nrows=metric_count, ncols=2,
                                 figsize=(14, metric_count * 6))
        for i, metric in enumerate(self.doubt_metrics):
            ax_left, ax_right = axes[i]
            doubt_drops = aspect_doubt_drops[:, i, :]
            doubts = aspect_doubts[:, i, :]
            for aspect in range(doubt_drops.shape[0]):
                ax_left.plot(doubt_drops[aspect, :],
                             label=str(aspect_profiles[aspect]))
                ax_left.set_title("Uncertainty DROP")
                ax_right.plot(doubts[aspect, :],
                              label=str(aspect_profiles[aspect]))
                ax_right.set_title("Uncertainty")
            for ax in axes[i]:
                ax.set_xlabel("Number of users asked")
                ax.set_ylabel(str(metric.__name__))
                ax.legend(loc="upper right")

        fig.suptitle("Uncertainty drop of different rating distribution - "
                     "{} stars system - ran {} times".format(
                         self.star_rank, self.run_count))
        fig.savefig(self.output)
        logger.info("Plot was exported to: {}".format(self.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Rating Doubt/Uncertainty Inspector")
    add_arg = parser.add_argument
    add_arg("--aspect-count", type=int, default=5,
            help="Number of aspect simulated, default=5")
    add_arg("-r", "--randomize-rating", action="store_true",
            help="Randomize the aspect's rating distribution")
    add_arg("--star-rank", type=int, default=5,
            help="Star system to rate, default=5")
    add_arg("--run-count", type=int, default=100,
            help="Number of experiment runs, default=100")
    add_arg("--user-count", type=int, default=200,
            help="Number of users to ask question, default=200")
    add_arg("--output", default="plot.pdf",
            help="File path of the output")
    args = parser.parse_args()
    logger.info(args)

    doubt_metrics = [
            uncertainty.expected_rating_var,
            uncertainty.dirichlet_var_sum,
            uncertainty.confidence_interval_len,
            uncertainty.entropy
            ]
    doubt_inspector = DoubtInspector(
            doubt_metrics,
            aspect_count=args.aspect_count,
            randomize_rating=args.randomize_rating,
            star_rank=args.star_rank,
            run_count=args.run_count,
            user_count=args.user_count,
            output=args.output)
    doubt_inspector.plot_doubt_drop()
