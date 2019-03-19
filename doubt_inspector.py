import argparse
from collections import namedtuple
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from reviews_soli import ReviewsSolicitation as ReviewSoli
from synthetic import SyntheticReview, random_alpha_beta, beta_binomial
import uncertainty as unc
from visualizer import GOLDEN_RATIO


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)

Doubt = namedtuple("Doubt", ["doubt", "doubt_drop", "expected_drop"])


class DoubtInspector:
    """Explore the properties of rating doubt/uncertainty."""

    def __init__(self, doubt_metrics,
                 aspect_count=5, randomize_rating=False, star_rank=5,
                 run_count=10, poll_count=300,
                 output="drop.pdf"):
        self.doubt_metrics = doubt_metrics
        self.aspect_count = aspect_count
        self.randomize_rating = randomize_rating
        self.star_rank = star_rank
        self.run_count = run_count
        self.poll_count = poll_count
        self.stars = np.arange(1, self.star_rank + 1, 1)
        self.output = output

    def plot_doubt_drop(self):
        aspect_profiles = self._build_aspect_profile()

        # 3d array: aspect * metric * poll_count
        aspect_doubts = []
        aspect_doubt_drops = []
        aspect_expected_drops = []
        for aspect_profile in aspect_profiles:
            doubt = self._sample_doubt_drop(aspect_profile)
            aspect_doubts.append(doubt.doubt)
            aspect_doubt_drops.append(doubt.doubt_drop)
            aspect_expected_drops.append(doubt.expected_drop)

        self._plot_doubt_trend(aspect_profiles,
                               np.array(aspect_doubts),
                               np.array(aspect_doubt_drops),
                               np.array(aspect_expected_drops))

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

    def _sample_doubt_drop(self, aspect_profile):
        """Sampling multiple time to record the change of rating.

        Return:
            Doubt named tuple: each element is a numpy 2d array with shape
                               (#doubt_metric, #poll_count)
        """
        star_generator = ReviewSoli.rating_generator(self.stars,
                                                     aspect_profile)
        metric_count = len(self.doubt_metrics)
        multi_run_doubts = np.zeros((self.run_count, self.poll_count,
                                     metric_count))
        multi_run_drops = np.zeros(
                (self.run_count, self.poll_count, metric_count))
        multi_run_expected_drops = np.zeros((self.run_count, self.poll_count,
                                             metric_count))

        # Each run is a full experiment that ask a number of polls
        for run in range(self.run_count):
            per_run_drops = np.zeros((self.poll_count, metric_count))
            per_run_doubts = np.zeros((self.poll_count, metric_count))
            per_run_expected_drops = np.zeros((self.poll_count, metric_count))

            ratings = np.ones(self.star_rank)
            # prev_doubts = np.array([doubt_metric(ratings)
            #                         for doubt_metric in self.doubt_metrics])
            prev_doubts = self._compute_doubt(ratings, aspect_profile)
            for poll in range(self.poll_count):
                # Expected uncertainty drop before getting a new actual rating
                # per_run_expected_drops[poll, :] = np.array(
                        # [unc.expected_uncertainty_drop(ratings,
                                # base_criterion=doubt_metric if doubt_metric !=
                                # unc.kl_divergence else
                                # unc.expected_rating_var)
                         # for doubt_metric in self.doubt_metrics]
                        # )
                per_run_expected_drops[poll, :] = self._compute_expected_drop(
                        ratings, aspect_profile)

                star = next(star_generator)
                ratings[star - 1] += 1
                curr_doubts = self._compute_doubt(ratings, aspect_profile)
                per_run_doubts[poll, :] = prev_doubts

                per_poll_drops = prev_doubts - curr_doubts
                prev_doubts = curr_doubts
                per_run_drops[poll, :] = per_poll_drops

            multi_run_drops[run, :, :] = per_run_drops
            multi_run_doubts[run, :, :] = per_run_doubts
            multi_run_expected_drops[run, :, :] = per_run_expected_drops

        doubts = np.mean(multi_run_doubts, axis=0)
        drops = np.mean(multi_run_drops, axis=0)
        expected_drops = np.mean(multi_run_expected_drops, axis=0)
        return Doubt(doubts.T, drops.T, expected_drops.T)

    def _compute_doubt(self, ratings, aspect_profile):
        """Compute doubt using appropriate metric.

        Returns:
            1d np array of length #doubt_metrics
        """
        return np.array([
            metric(ratings) if metric != unc.kl_divergence
            else metric(aspect_profile, ratings)
            for metric in self.doubt_metrics])

    def _compute_expected_drop(self, ratings, aspect_profile):
        """Compute expected doubt drop using appropriate metric.

        Returns:
            1d np array of length #doubt_metrics
        """
        expected_drops = [
                unc.expected_uncertainty_drop(
                    ratings,
                    base_criterion=metric if metric != unc.kl_divergence
                    else unc.expected_rating_var
                    )
                for metric in self.doubt_metrics
                ]
        return np.array(expected_drops)

    def _plot_doubt_trend(self, aspect_profiles, aspect_doubts,
                          aspect_doubt_drops, aspect_expected_drops):
        """Plot doubt and its change with different metric.

        Args:
            aspect_profiles: list of aspect's rating distribution (profile)
            aspect_doubts, aspect_doubt_drops, aspect_expected_drops:
                all are 3d arrays: aspect * metric * poll_count
        """
        np.set_printoptions(precision=1)
        metric_count = len(self.doubt_metrics)
        ncols = 3
        height = 4.5
        fig, axes = plt.subplots(nrows=metric_count, ncols=ncols,
                                 figsize=(ncols * height * GOLDEN_RATIO,
                                          metric_count * height))
        for i, metric in enumerate(self.doubt_metrics):
            doubt_amounts = [aspect_doubts[:, i, :],
                             aspect_doubt_drops[:, i, :],
                             aspect_expected_drops[:, i, :]]
            amount_names = ["Uncertainty",
                            "Uncertainty DROP",
                            "Uncertainty EXPECTED DROP"]
            for ax, amounts, name in zip(axes[i], doubt_amounts, amount_names):
                for aspect in range(amounts.shape[0]):
                    ax.plot(amounts[aspect, :],
                            label=str(aspect_profiles[aspect]))
                    ax.set_title(name)
                ax.set_xlabel("Number of polls")
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
    add_arg("--poll-count", type=int, default=200,
            help="Number of polls, default=200")
    add_arg("--output", default="plot.pdf",
            help="File path of the output")
    args = parser.parse_args()
    logger.info(args)

    doubt_metrics = [
            unc.expected_rating_var,
            unc.dirichlet_var_sum,
            unc.confidence_interval_len,
            unc.entropy,
            unc.shannon_entropy,
            unc.kl_divergence
            ]
    doubt_inspector = DoubtInspector(
            doubt_metrics,
            aspect_count=args.aspect_count,
            randomize_rating=args.randomize_rating,
            star_rank=args.star_rank,
            run_count=args.run_count,
            poll_count=args.poll_count,
            output=args.output)
    doubt_inspector.plot_doubt_drop()
