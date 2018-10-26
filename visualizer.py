import argparse
from collections import OrderedDict
import logging
from os import path
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd

import soli_start
from uncertainty import expected_rating_var
from bliu import BliuReview
from synthetic import SyntheticReview


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s - %(message)s'))
logger.addHandler(ch)


# Text width according to different conference templates
#   \the\textwidth from latex, unit: point
CONF_TEXT_WIDTHS = {
        "acm": 506.295,
        "siam": 492.0
        }
# Points per inch: https://en.wikibooks.org/wiki/LaTeX/Lengths
PTS_PER_INCH = 72.27
GOLDEN_RATIO = 1.618


def figsize(scale, conference="acm", ratio=GOLDEN_RATIO):
    """Figure size in inch.

    Returns:
        (fig_width, fig_height) tuple, unit: inch
    """
    text_width_pt = CONF_TEXT_WIDTHS[conference]
    text_width_in = text_width_pt / PTS_PER_INCH
    fig_width = text_width_in * scale
    fig_height = fig_width / ratio
    return (fig_width, fig_height)


def savefig(fig, filename):
    fig.savefig('{}.pdf'.format(filename), bbox_inches='tight')


def set_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.despine()

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=10)
    plt.rc("legend", fontsize=8)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)


palette = sns.color_palette()
HATCHES = ("/", "'", "|", "-", "+", "x", "o", "O", ".", "*")
MARKERS = ('o', 'd', 'x', '+', 'P', 'v',
           '1', '2', '3', '4',
           '^', '<', '>', 's', 'p', '*',
           'h', 'H', 'D')
MARKER_SIZE = 4
MARKER_WIDTH = 1


def _to_latex(raw_str):
    """Convert raw string to the latex-compliant string."""
    return raw_str.replace("_", " ")


def plot_experiment_result(experiment,
                           experiment_dir="output/",
                           dataset="edmunds",
                           plot_dir="plots/",
                           poll=299,
                           marker_step=20,
                           conference="acm",
                           scale=1,
                           ratio=GOLDEN_RATIO):
    """Plot complete result of an experiment.
    Args:
        experiment: str,
            which experiment to plot
        experiment_dir: str, default='output/'
            where does the experiment result is PICKLED
        plot_dir: str, default='plots/'
            where to store exported plots
        poll: int, default=299
            the end poll to plot
    """
    # product_to_aspect_stars: dict, default=None
    #     if available, then plot for each product, e.g., bliu dataset
    product_to_aspect_stars = None
    if dataset == "bliu":
        _, product_to_aspect_stars = BliuReview.preprocess_dataset(
            "anno-datasets/bliu-datasets")
    if dataset == "synthetic":
        product_to_aspect_stars = SyntheticReview.import_synthetic_profile()

    filename = path.join(plot_dir, experiment)
    experiment_path = path.join(experiment_dir, experiment + ".pickle")
    with open(experiment_path, 'rb') as f:
        product_to_stats = pickle.load(f)

    if not product_to_stats:
        logger.warn("Nothing at {} to plot!!!".format(experiment_path))
        return

    soliconfig_to_stats_average = soli_start.summary_product_to_config_stats(
        product_to_stats, plotted_poll_end=poll, ignore_rating=True)
    fig = plot_sim_stats(soliconfig_to_stats_average,
                         plot_dir=plot_dir,
                         poll=poll,
                         marker_step=marker_step,
                         plot_rating=False,
                         plot_pdf_prefix=experiment,
                         scale=scale,
                         conference=conference,
                         ratio=ratio)
    fig.suptitle('Average over all products', fontsize=15, fontweight='bold')
    savefig(fig, filename)

    if product_to_aspect_stars is not None:
        for product, goal_to_stats in product_to_stats.items():
            fig = plot_sim_stats(
                goal_to_stats,
                plot_dir=plot_dir,
                poll=poll,
                marker_step=marker_step,
                product=product,
                aspect_to_star_counts=product_to_aspect_stars[product]
                )
            fig.suptitle(product, fontsize=15, fontweight='bold')
            savefig(fig, filename + "_" + str(product))

    logger.info(f'Exported plots to "{filename}*.pdf"')


def plot_sim_stats(soliconfig_to_stats,
                   poll=100, marker_step=20, fig_w=16, subplt_fig_h=5,
                   plot_rating=True, product=None, aspect_to_star_counts=None,
                   scale=1, conference="acm", ratio=GOLDEN_RATIO,
                   plot_pdf_prefix=None, plot_dir='plots'):
    """Plot a product's rating distribution and simulation result statistics.
    Args:
        soliconfig_to_stats_average: dict,
            soliconfig (SoliConfig) -> SimulationStats
        poll: int, default=100
        fig_w: int, default=16,
            figure width
        subplt_fig_h: int, default=5,
            subplot height
        plot_rating: bool, default=True
        product: str
            apply when plot_rating=True
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
            apply when plot_rating=True
    """
    stats_sample = list(soliconfig_to_stats.values())[0]
    answer_to_goal_stats = partition_goal_by_answer(soliconfig_to_stats)
    answer_count = len(answer_to_goal_stats)
    metric_count = len(stats_sample.poll_to_report[1].metrics())

    # grid layout of 2 plot types
    uncertainty_col_count = 2
    uncertainty_row_count = metric_count * answer_count

    probe_poll_count = 6    # Number of polls to plot ratings at.
    # 4: first for dataset star counts, 3 for violet, sd and variance plots
    rating_row_count = answer_count * probe_poll_count + 4
    uncertainty_rating_subplt_ratio = 1.6

    uncertainty_h_unit = uncertainty_row_count
    rating_h_unit = rating_row_count / uncertainty_rating_subplt_ratio
    h_unit_count = uncertainty_h_unit + rating_h_unit \
        if plot_rating else uncertainty_h_unit
    fig = plt.figure(figsize=(fig_w, h_unit_count * subplt_fig_h))
    if plot_rating:
        gs = gridspec.GridSpec(
            2, 1, hspace=0.1,
            height_ratios=[rating_h_unit, uncertainty_h_unit])
    else:
        gs = gridspec.GridSpec(1, 1)

    # Plot dataset rating and ratings after simulation
    if plot_rating:
        aspect_to_raw_stars = get_raw_stars(aspect_to_star_counts)
        aspect_to_raw_stars = OrderedDict(
            sorted(aspect_to_raw_stars.items(), key=lambda kv: np.std(kv[1])))
        aspects = list(aspect_to_raw_stars.keys())

        rating_row_id = 0
        gs1 = gridspec.GridSpecFromSubplotSpec(rating_row_count, 1,
                                               hspace=0.4,
                                               subplot_spec=gs[0])

        ax = plt.subplot(gs1[rating_row_id, :])
        aspect_to_star_counts = OrderedDict([
            (aspect, aspect_to_star_counts[aspect]) for aspect in aspects])
        plot_aspect_star_counts(ax, product, aspect_to_star_counts)
        rating_row_id += 1

        axarr = [plt.subplot(gs1[rating_row_id, :]),
                 plt.subplot(gs1[rating_row_id + 1, :]),
                 plt.subplot(gs1[rating_row_id + 2, :])]
        plot_aspect_rating_dist(axarr, product, aspect_to_star_counts,
                                aspect_to_raw_stars)
        rating_row_id += 3

        axarr = [plt.subplot(gs1[rating_row_id + i, :])
                 for i in range(answer_count * probe_poll_count)]
        plot_picked_features(axarr, soliconfig_to_stats,
                             answer_count=answer_count, poll=poll,
                             sorted_features=aspects)

    # Plot uncertainties over polls
    gs0 = gridspec.GridSpecFromSubplotSpec(
        uncertainty_row_count, uncertainty_col_count,
        subplot_spec=gs[1] if plot_rating else gs[0])

    for i, answer in enumerate(answer_to_goal_stats.keys()):
        offset = metric_count * i
        uncertainty_axarr = [plt.subplot(gs0[row_id + offset, col_id])
                             for row_id in range(metric_count)
                             for col_id in range(uncertainty_col_count)]
        plot_cost_of_multi_picks(uncertainty_axarr, soliconfig_to_stats,
                                 poll=poll, marker_step=marker_step,
                                 answer=answer)
        if plot_pdf_prefix is not None:
            figure_size = figsize(scale, conference=conference, ratio=ratio)
            plot_cost_of_multi_picks_to_pdfs(
                plot_pdf_prefix,
                len(uncertainty_axarr),
                soliconfig_to_stats,
                figure_size,
                plot_dir=plot_dir,
                poll=poll,
                marker_step=marker_step,
                answer=answer)
        export_cost_of_multi_picks_same_answer(
            soliconfig_to_stats,
            poll=poll,
            answer=answer,
            filename=product if product else "overall")
    return fig


def export_cost_of_multi_picks_same_answer(soliconfig_to_stats,
                                           answer="answer_by_gen", poll=100,
                                           filename=None):
    """Export cost of multiple pick method with same answer to xlsx file.

    Args:
        axarr: list of Axes
            Each axes is a subplot of a SoliConfig
        soliconfig_to_stats: SoliConfig -> SimulationStats
    """
    stats_sample = list(soliconfig_to_stats.values())[0]
    metrics = stats_sample.poll_to_report[1].metrics()
    metric_answer_to_df = OrderedDict()

    for metric in metrics:
        goal_to_totals = OrderedDict()
        baselines = []
        for goal, stats in soliconfig_to_stats.items():
            goal_str = goal.pick_goal_str()
            if goal.baseline:
                reports = stats.uncertainty_reports[0:poll]
                goal_to_totals[goal_str] = [
                    report.get_uncertainty_total(metric)
                    for report in reports]
                df = pd.DataFrame(goal_to_totals)
                baselines.append(goal_str)

        for goal, stats in soliconfig_to_stats.items():
            goal_str = goal.pick_goal_str()
            reports = stats.uncertainty_reports[0:poll]
            totals = [report.get_uncertainty_total(metric)
                      for report in reports]

            if not goal.baseline:
                df[goal_str] = totals
                for baseline in baselines:
                    df[goal_str + " / " + baseline + " (%)"] = (
                            1 - df[goal_str] / df[baseline]) * 100

        metric_answer_to_df[str(metric)] = df

    filepath = "output/" + filename + "_" + answer + ".xlsx"
    with pd.ExcelWriter(filepath) as writer:
        for sheetname, df in metric_answer_to_df.items():
            df.to_excel(writer, sheet_name=sheetname[:31],
                        float_format="%.3f", freeze_panes=(1, 1))


def plot_cost_of_multi_picks(axarr,
                             soliconfig_to_stats,
                             answer="answer_by_gen",
                             poll=100,
                             marker_step=20,
                             std_poll_step=10):
    """Plot the uncertainty change of multiple pick method with same answer.
    Args:
        axarr: list of Axes
            Each axes is a cost/std subplot of a SoliConfig
            In total, #axes: #metric * 2 (cost/std)
        soliconfig_to_stats: SoliConfig -> SimulationStats
    """
    stats_sample = list(soliconfig_to_stats.values())[0]
    metrics = stats_sample.poll_to_report[1].metrics()

    X = list(stats_sample.polls)[0:poll]
    goals = soliconfig_to_stats.keys()
    for metric_idx, metric in enumerate(metrics):
        for goal, marker in zip(goals, MARKERS[:len(goals)]):
            goal_str = goal.pick_goal_str()
            stats = soliconfig_to_stats[goal]
            reports = stats.uncertainty_reports[0:poll]
            # Plot uncertainty result
            ax = axarr[metric_idx * 2]
            Y = [report.get_uncertainty_total(metric) for report in reports]
            ax.plot(X[::marker_step], Y[::marker_step],
                    label=_to_latex(goal_str),
                    marker=marker,
                    ms=MARKER_SIZE,
                    markeredgewidth=MARKER_WIDTH)
            ax.set_ylabel(_to_latex(str(metric)))
            ax.set_xlabel("Number of reviews")
            if metric.ratio:
                # avoid overlapping in case of high_confidence_ratio metric
                ax.legend(loc='lower right')
            else:
                ax.legend(loc='upper right')

            # Plot standard deviation
            ax_std = axarr[metric_idx * 2 + 1]
            Y_std = [report.metric_to_std[metric] for report in reports]
            ax_std.plot(X[::std_poll_step], Y_std[::std_poll_step],
                        label=_to_latex(goal.pick_goal_str()),
                        marker=marker,
                        ms=MARKER_SIZE,
                        markeredgewidth=MARKER_WIDTH)
            ax_std.set_ylabel("standard deviation")
            ax_std.set_xlabel("Number of reviews")
            ax_std.legend(loc='upper right')


def plot_cost_of_multi_picks_to_pdfs(
        pdf_prefix,
        subplot_count,
        soliconfig_to_stats,
        figsize,
        plot_dir="plots/",
        answer="answer_by_gen",
        poll=100,
        marker_step=5,
        std_poll_step=10):
    """Export to pdf plot.

    """
    filenames = [path.join(plot_dir, pdf_prefix + "_" + str(i))
                 for i in range(subplot_count)]
    figs = [plt.figure(figsize=figsize)
            for i in range(subplot_count)]
    axarr = [fig.add_subplot(1, 1, 1) for fig in figs]

    plot_cost_of_multi_picks(axarr, soliconfig_to_stats,
                             poll=poll, marker_step=marker_step, answer=answer)

    for filename, fig in zip(filenames, figs):
        fig.patch.set_alpha(0.)
        savefig(fig, filename)
        plt.close()     # Suppress showing inline in Jupyter notebook


def partition_goal_by_answer(goal_to_value):
    """Partition argument into different answer methods.

    Returns:
        answer_to_goal_values: dict
            answer -> goal_to_value
    """
    answer_to_goal_values = OrderedDict()
    for goal, value in goal_to_value.items():
        if goal.answer not in answer_to_goal_values:
            answer_to_goal_values[goal.answer] = OrderedDict()
        answer_to_goal_values[goal.answer][goal] = value

    return answer_to_goal_values


def plot_picked_features(axarr, soliconfig_to_stats,
                         answer_count=2, poll=100, sorted_features=None):
    """Bar plot of different methods' ratings at a specific poll.

    Args:
        axarr: list of Axes to draw
        soliconfig_to_stats_average: dict,
            soliconfig (SoliConfig) -> SimulationStats
        answer_count: int, default=2,
            Number of answer options in SoliConfig of the input
        poll: int, default=100,
        sorted_features: list of str
    """
    probe_poll_count = len(axarr) // answer_count
    probe_poll_interval = poll // probe_poll_count
    features = list(soliconfig_to_stats.values())[0].features
    if not sorted_features:
        features = sorted(features, key=lambda f: f.name)
    features = sorted(features, key=lambda f: sorted_features.index(f.name))

    configs = list(soliconfig_to_stats.keys())
    X = np.arange(len(sorted_features))

    answer_to_goal_stats = partition_goal_by_answer(soliconfig_to_stats)
    subpl_idx = 0
    for answer, goal_to_stats in answer_to_goal_stats.items():
        for i in range(1, probe_poll_count + 1):
            probe_poll = probe_poll_interval * i + 1

            feature_to_per_config_counts = get_aspect_picked_counts(
                goal_to_stats, features, poll=probe_poll)
            ax = axarr[subpl_idx]
            count_max = 0
            count_min = float("inf")
            for i, config in enumerate(configs):
                Y = [config_to_count[config]
                     for config_to_count in feature_to_per_config_counts.values()]
                correlation = goal_to_stats[config].correlation_at(probe_poll)
                ax.plot(X, Y,
                        label=_to_latex(config.pick_goal_str()) + ", cor:\n" +\
                                str(correlation),
                        marker=MARKERS[i],
                        ms=MARKER_SIZE,
                        markeredgewidth=MARKER_WIDTH
                        )
                count_max = max(count_max, np.max(Y))
                count_min = min(count_min, np.min(Y))

            ax.set_xticks(X)
            ax.set_xticklabels([_to_latex(feature.name)
                                for feature in features])
            ax.set_title("2. Rating count after {} polls ({})".format(
                probe_poll, _to_latex(answer)))
            ax.set_ylabel("\# Ratings")
            ax.legend(loc='upper left', ncol=5)
            ax.set_ylim(count_min * 0.95, count_max * 1.03)
            subpl_idx += 1


def get_aspect_picked_counts(soliconfig_to_stats, sorted_features, poll=100):
    feature_to_per_config_counts = OrderedDict()

    config_to_ratings = {config: stats.poll_to_report[poll - 1].ratings
                         for config, stats in soliconfig_to_stats.items()}
    for feature in sorted_features:
        feature_to_per_config_counts[feature] = {
            config: np.sum(ratings[feature.idx, :])
            for config, ratings in config_to_ratings.items()}

    return feature_to_per_config_counts


def plot_aspect_star_counts(ax, product, aspect_to_star_counts):
    """Bar plot of normalized rating distribution of a single product.

    Args:
        ax: Axes
        product: str
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
    """
    star_rank = max([max(star_counts.keys())
                     for star_counts in aspect_to_star_counts.values()])
    aspects = list(aspect_to_star_counts.keys())
    X = np.arange(len(aspects))
    width = 1 / (star_rank + 2)
    for star in range(1, star_rank + 1):
        X_left = X + (star - 1) * width
        Y = [star_to_count[star]
             for star_to_count in aspect_to_star_counts.values()]
        ax.bar(X_left, Y, width, label="{} stars".format(star))

    X_mid = X + (star_rank - 1) / 2 * width
    ax.set_xticks(X_mid)
    ax.set_xticklabels(aspects)
    ax.legend(loc='upper center', ncol=5)
    ax.set_ylabel('\# Ratings')
    ax.set_title("1a. Dataset profile: rating count (used in generator)")


def plot_aspect_rating_dist(axarr, product,
                            aspect_to_star_counts, aspect_to_raw_stars):
    """Bar plot of normalized rating distribution of a single product.

    Args:
        axarr: list of 2 Axes
            one Axes for std plot, one Axes for violin plot
        product: str
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
    """
    aspects = list(aspect_to_raw_stars.keys())
    raw_stars = [np.array(stars) for stars in aspect_to_raw_stars.values()]

    X = np.arange(len(aspects)) + 1
    ax_violin, ax_std, ax_erv = axarr

    # Violin plot for estimating distribution shape
    ax_violin.violinplot(raw_stars, showmeans=True)
    ax_violin.set_ylabel("Star")
    ax_violin.set_title(
        "1b. Dataset profile: rating distribution (used in generator)")

    # Standard Deviation of ratings
    aspect_stds = [np.std(stars) for stars in aspect_to_raw_stars.values()]
    ax_std.plot(X, aspect_stds,
                marker=MARKERS[-1],
                ms=MARKER_SIZE,
                markeredgewidth=MARKER_WIDTH)
    std_max = np.max(aspect_stds)
    for x, std in zip(X, aspect_stds):
        ax_std.annotate("{:.2f}".format(std), xy=(x, std + std_max * 0.01),
                        ha="center")
    ax_std.set_ylabel("Standard Deviation")
    ax_std.set_title("1c. Dataset profile: Standard Deviation "
                     "(used in generator)")
    ax_std.set_ylim(top=std_max * 1.05)

    # Expected Rating Variance
    aspect_exp_rating_vars = [
        expected_rating_var(np.array(list(star_counts.values())))
        for star_counts in aspect_to_star_counts.values()]
    ax_erv.plot(X, aspect_exp_rating_vars,
                marker=MARKERS[-1],
                ms=MARKER_SIZE,
                markeredgewidth=MARKER_WIDTH)
    erv_max = np.max(aspect_exp_rating_vars)
    for x, erv in zip(X, aspect_exp_rating_vars):
        ax_erv.annotate("{:.3f}".format(erv), xy=(x, erv + erv_max * 0.01),
                        ha="center")
    ax_erv.set_ylabel("Expected Rating Variance")
    ax_erv.set_title("1d. Dataset profile: Expected Rating Variance by"
                     " Bayesian (used in generator)")
    ax_erv.set_ylim(top=erv_max * 1.05)

    plt.setp(axarr, xticks=X, xticklabels=aspects)


def get_raw_stars(aspect_to_star_counts):
    """Get the raw star list of each aspect.
    Args:
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
    Returns:
        aspect_to_raw_stars: dict
            aspect -> list of stars
    """
    aspect_to_raw_stars = OrderedDict()
    for aspect, star_to_count in aspect_to_star_counts.items():
        stars = []
        for star, count in star_to_count.items():
            stars.extend([star] * count)
        aspect_to_raw_stars[aspect] = stars
    return aspect_to_raw_stars


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize experiment result")
    add_arg = parser.add_argument
    add_arg("--dataset", default="edmunds",
            help="Dataset name (default='edmunds')")
    add_arg("--experiment", default="bliu_p300_q5_r20_real_200",
            help="Experiment name (default='bliu_p300_q5_r20_real_200')")
    add_arg("--experiment-dir", default="output/",
            help="""Directory of experimental results, i.e. the pickle file
                    (default: output/)""")
    add_arg("--output-dir", default="plots/",
            help="Output dir of exported plots. Default: plots/")
    add_arg("--poll", type=int, default=299,
            help="last poll")
    add_arg("--marker-step", type=int, default=20,
            help="Interval to print marker in the plot, default=20")
    add_arg("--conference", default="acm", choices=CONF_TEXT_WIDTHS.keys(),
            help="""Conference name, used for calculating appropriate text
                    width in paper (default='acm').""")
    add_arg("--scale", type=float, default=1.0,
            help="Scale plot to a factor of text width (default=1).")
    add_arg("--ratio", type=int, default=GOLDEN_RATIO,
            help="""ratio between width and height of exported single plot (
                    default={}, i.e. golden ratio)""".format(GOLDEN_RATIO))

    args = parser.parse_args()

    # Style for publication friendly plots
    set_style()
    np.set_printoptions(precision=1)

    plot_experiment_result(
        args.experiment,
        experiment_dir=args.experiment_dir,
        dataset=args.dataset,
        plot_dir=args.output_dir,
        poll=args.poll,
        marker_step=args.marker_step,
        conference=args.conference,
        scale=args.scale,
        ratio=args.ratio
    )
