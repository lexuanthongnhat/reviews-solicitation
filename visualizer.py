from collections import OrderedDict
import math

import matplotlib.pyplot as plt
import numpy as np


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


def plot_pick_answer_goals(soliconfig_to_stats,
                           answer_count=2, poll_max=100, fig_w=20):
    """
    Args:
        soliconfig_to_stats: SoliConfig -> SimulationStats
    """
    stats_sample = list(soliconfig_to_stats.values())[0]
    metrics = stats_sample.poll_to_report[1].metrics()
    answer_to_goal_stats = partition_goal_by_answer(soliconfig_to_stats)

    subpl_col_count = 2
    subpl_row_count = len(metrics) if answer_count == 2 \
        else math.ceil(len(metrics) / subpl_col_count)

    X = list(stats_sample.polls)[0:poll_max]
    fig, axarr = plt.subplots(subpl_row_count, subpl_col_count,
                              figsize=(fig_w, subpl_row_count * 6),
                              subplot_kw=dict(xlabel='Number of polls'))
    subpl_idx = 0
    for metric_idx, metric in enumerate(metrics):
        for answer, soliconfig_to_stats in answer_to_goal_stats.items():
            # each goal is plotted by a curve
            for goal, stats in soliconfig_to_stats.items():
                ax = axarr[subpl_idx // subpl_col_count,
                           subpl_idx % subpl_col_count]
                Y = [report.get_uncertainty_total(metric)
                     for report in stats.uncertainty_reports[0:poll_max]]
                ax.plot(X, Y, label=goal.pick_goal_str())

                ax.set_title('Cost change over polls ({})'.format(answer))
                ax.set_ylabel(str(metric))
                ax.legend(loc='upper right')
            subpl_idx += 1

    plt.show()
    return fig


def plot_picked_features(soliconfig_to_stats,
                         answer_count=2, poll=100, fig_w=20, fig_h=4):
    """Bar plot of different methods' ratings at a specific poll.

    Args:
        soliconfig_to_stats_average: dict,
            soliconfig (SoliConfig) -> SimulationStats
        answer_count: int, default=2,
            Number of answer options in SoliConfig of the input
        poll: int, default=100,
        fig_w: int, default=10
            figure width
    """
    fig, axarr = plt.subplots(answer_count, 1,
                              figsize=(fig_w, fig_h * answer_count),
                              subplot_kw=dict(ylabel='Number of ratings'))
    features = list(soliconfig_to_stats.values())[0].features
    features = sorted(features, key=lambda f: f.name)
    configs = list(soliconfig_to_stats.keys())
    X = np.arange(len(features))
    width = 1 / (1 + len(configs))

    answer_to_goal_stats = partition_goal_by_answer(soliconfig_to_stats)
    subpl_idx = 0
    for answer, goal_to_stats in answer_to_goal_stats.items():
        feature_to_per_config_counts = get_aspect_picked_counts(goal_to_stats,
                                                                poll=poll)
        if answer_count == 1:
            ax = axarr
        else:
            ax = axarr[subpl_idx]
        # plot each goal's stats by a set of horizontal bars
        for i, config in enumerate(configs):
            X_pos = X + i * width
            Y = [config_to_count[config]
                 for config_to_count in feature_to_per_config_counts.values()]
            # ax.bar(X_pos, Y, width, label=config.pick_goal_str())
            ax.plot(X, Y, label=config.pick_goal_str())

        ax.set_xticks(X)
        ax.set_xticklabels([feature.name for feature in features])
        ax.set_title("Rating distribution after {} polls ({})".format(
            poll, answer))
        ax.legend(loc='upper right')
        subpl_idx += 1

    plt.show()
    return fig


def get_aspect_picked_counts(soliconfig_to_stats, poll=100):
    feature_to_per_config_counts = OrderedDict()

    features = list(soliconfig_to_stats.values())[0].features
    features = sorted(features, key=lambda f: f.name)
    config_to_ratings = {config: stats.poll_to_report[poll - 1].ratings
                         for config, stats in soliconfig_to_stats.items()}
    for feature in features:
        feature_to_per_config_counts[feature] = {
                config: np.sum(ratings[feature.idx, :])
                for config, ratings in config_to_ratings.items()}
    
    return feature_to_per_config_counts


def plot_ratings(soliconfig_to_stats, answer_count=2, poll=100, fig_w=10):
    """Bar plot of different methods' ratings at a specific poll.

    Args:
        soliconfig_to_stats_average: dict,
            soliconfig (SoliConfig) -> SimulationStats
        answer_count: int, default=2,
            Number of answer options in SoliConfig of the input
        poll: int, default=100,
        fig_w: int, default=10
            figure width
    """
    fig, axarr = plt.subplots(1, answer_count,
                              figsize=(fig_w, 10), sharey=True,
                              subplot_kw=dict(xlabel='Number of ratings'))
    features = list(soliconfig_to_stats.values())[0].features

    Y = np.arange(len(soliconfig_to_stats) / answer_count, 0, -1)
    width = 1 / (1 + len(features))

    answer_to_goal_stats = partition_goal_by_answer(soliconfig_to_stats)
    subpl_idx = 0
    for answer, goal_to_stats in answer_to_goal_stats.items():
        if answer_count == 1:
            ax = axarr
        else:
            ax = axarr[subpl_idx]
        # plot each goal's stats by a set of horizontal bars
        for feature in features:
            X = [np.sum(
                stats_average.poll_to_report[poll - 1].ratings[feature.idx, :])
                for stats_average in goal_to_stats.values()
                ]
            Y_pos = Y - feature.idx * width
            ax.barh(Y_pos, X, width, label=feature.name)

        Y_pos_mid = Y - len(features) / 2 * width
        ax.set_yticks(Y_pos_mid)
        ax.set_yticklabels([goal.pick_goal_str()
                            for goal in goal_to_stats.keys()])
        ax.set_title("Rating distribution after {} polls ({})".format(
            poll, answer))
        ax.legend(loc='upper right')
        subpl_idx += 1

    plt.show()
    return fig


def plot_aspect_star_counts(product, aspect_to_star_counts,
                            fig_w=20, fig_h=4):
    """Bar plot of normalized rating distribution of a single product.
    
    Args:
        product: str
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
        fig_w: int, default=20
        fig_h: int, default=4
    Returns:
        figure
    """
    aspect_to_star_counts = OrderedDict(
            sorted(aspect_to_star_counts.items(), key=lambda kv: kv[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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

    ax.set_title("Rating distribution of {}".format(product))
    ax.legend(loc='upper right')
    ax.set_ylabel('Rating Count (normalized to [0, 1])')
    plt.show()
    return fig


def plot_aspect_rating_dist(product, aspect_to_star_counts,
                            fig_w=20, fig_h=4):
    """Bar plot of normalized rating distribution of a single product.
    
    Args:
        product: str
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
        fig_w: int, default=20
        fig_h: int, default=4
    Returns:
        figure
    """
    aspect_to_star_counts = OrderedDict(
            sorted(aspect_to_star_counts.items(), key=lambda kv: kv[0]))
    fig, axarr = plt.subplots(2, 1, figsize=(fig_w, 2 * fig_h), sharex=False,
                              subplot_kw=dict(ylabel='Star'))
    ax_std, ax_violin = axarr
    star_rank = max([max(star_counts.keys())
                     for star_counts in aspect_to_star_counts.values()])
    aspects = list(aspect_to_star_counts.keys())
    aspect_to_raw_stars = get_raw_stars(aspect_to_star_counts)
    raw_stars = [np.array(stars) for stars in aspect_to_raw_stars.values()]

    ax_std.plot(np.arange(len(aspects)) + 1,
                [np.std(stars) for stars in aspect_to_raw_stars.values()])
    ax_std.set_ylabel("Standard Deviation")

    vi_plt = ax_violin.violinplot(raw_stars, showmeans=True, showmedians=True)
    plt.setp(axarr,
             xticks=[y + 1 for y in range(len(aspects))],
             xticklabels=aspects)

    # Draw standard deviation
    upper_y = 0
    for aspect, end_points in zip(aspects, vi_plt["cbars"].get_segments()):
        star_std = np.std(aspect_to_raw_stars[aspect]) 
        _, upper = end_points
        upper_y = upper[1] + 0.36
        ax_violin.text(upper[0], upper_y,
                      "{:.2f}".format(star_std), horizontalalignment="center")
    ax_violin.text(0.5, upper_y, u"std \u2192", horizontalalignment="right")

    vi_plt["cmedians"].set_color("orange")
    vi_plt["cmeans"].set_color("green")

    return fig


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


def _normalize_star_counts(aspect_to_star_counts):
    """Normalize star counts distribution.

    Args:
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
    Returns:
        aspect_to_star_counts_normalized: dict
        aspect_to_std: dict
            aspect -> standard deviation
        star_rank: int
    """
    star_rank = max([max(star_counts.keys())
                     for star_counts in aspect_to_star_counts.values()])

    aspect_to_star_counts_normalized = OrderedDict()
    aspect_to_std = OrderedDict()
    for aspect, star_counts in aspect_to_star_counts.items():
        count_total = sum(star_counts.values())
        aspect_to_star_counts_normalized[aspect] = {
                star: count / count_total
                for star, count in star_counts.items()}

        aspect_to_std[aspect] = np.std(
                list(aspect_to_star_counts_normalized[aspect].values()))

    return (aspect_to_star_counts_normalized, aspect_to_std, star_rank)
