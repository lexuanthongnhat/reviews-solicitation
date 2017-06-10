from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd


def plot_sim_stats(soliconfig_to_stats,
                   poll=100, fig_w=16, subplt_fig_h=5, plot_rating=True,
                   product=None, aspect_to_star_counts=None):
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
    rating_row_count = answer_count + 3   # 3: std, violin, bar plots
    uncertainty_rating_subplt_ratio = 2

    uncertainty_h_unit = uncertainty_row_count
    rating_h_unit = rating_row_count / uncertainty_rating_subplt_ratio
    h_unit_count = uncertainty_h_unit + rating_h_unit \
        if plot_rating else uncertainty_h_unit
    fig = plt.figure(figsize=(fig_w, h_unit_count * subplt_fig_h))
    if plot_rating:
        gs = gridspec.GridSpec(
                2, 1, hspace=0.1,
                height_ratios=[uncertainty_h_unit, rating_h_unit])
    else:
        gs = gridspec.GridSpec(1, 1)

    # Plot uncertainties over polls
    gs0 = gridspec.GridSpecFromSubplotSpec(uncertainty_row_count,
                                           uncertainty_col_count,
                                           subplot_spec=gs[0])

    for i, answer in enumerate(answer_to_goal_stats.keys()):
        offset = metric_count * i
        uncertainty_axarr = [plt.subplot(gs0[row_id + offset, col_id])
                             for row_id in range(metric_count)
                             for col_id in range(uncertainty_col_count)]
        plot_pick_answer_goals(uncertainty_axarr, soliconfig_to_stats,
                               poll=poll, answer=answer)
        export_pick_answer_goals(soliconfig_to_stats,
                                 poll=poll, answer=answer,
                                 filename=product if product else "overall")

    # Plot dataset rating and ratings after simulation
    if plot_rating:
        rating_row_id = 0
        gs1 = gridspec.GridSpecFromSubplotSpec(rating_row_count, 1,
                                               hspace=0.4,
                                               subplot_spec=gs[1])
        axarr = [plt.subplot(gs1[rating_row_id + i, :])
                 for i in range(answer_count)]
        plot_picked_features(axarr, soliconfig_to_stats,
                             answer_count=answer_count, poll=poll)

        rating_row_id += answer_count
        axarr = [plt.subplot(gs1[rating_row_id, :]),
                 plt.subplot(gs1[rating_row_id + 1, :])]
        plot_aspect_rating_dist(axarr, product, aspect_to_star_counts)

        rating_row_id += 2
        ax = plt.subplot(gs1[rating_row_id, :])
        plot_aspect_star_counts(ax, product, aspect_to_star_counts)

    plt.show()
    return fig


def export_pick_answer_goals(soliconfig_to_stats,
                             answer="answer_by_gen", poll=100,
                             filename=None):
    """
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
                    df[goal_str + " / " + baseline] = (
                            1 - df[goal_str] / df[baseline]) * 100

        metric_answer_to_df[str(metric)] = df

    filepath = "output/" + filename + "_" + answer + ".xlsx"
    with pd.ExcelWriter(filepath) as writer:
        for sheetname, df in metric_answer_to_df.items():
            df.to_excel(writer, sheet_name=sheetname[:31],
                        float_format="%.2f", freeze_panes=(1, 1))


def plot_pick_answer_goals(axarr, soliconfig_to_stats,
                           answer="answer_by_gen", poll=100,
                           uncertainty_poll_step=5, std_poll_step=10):
    """
    Args:
        axarr: list of Axes
            Each axes is a subplot of a SoliConfig
        soliconfig_to_stats: SoliConfig -> SimulationStats
    """
    stats_sample = list(soliconfig_to_stats.values())[0]
    metrics = stats_sample.poll_to_report[1].metrics()

    X = list(stats_sample.polls)[0:poll]
    for metric_idx, metric in enumerate(metrics):
        for goal, stats in soliconfig_to_stats.items():
            goal_str = goal.pick_goal_str()
            reports = stats.uncertainty_reports[0:poll]
            # Plot uncertainty result
            ax = axarr[metric_idx * 2]
            Y = [report.get_uncertainty_total(metric) for report in reports]
            ax.plot(X[::uncertainty_poll_step], Y[::uncertainty_poll_step],
                    label=goal_str)

            ax.set_title('Cost change over polls ({})'.format(answer))
            ax.set_ylabel(str(metric))

            # Plot standard deviation
            ax_std = axarr[metric_idx * 2 + 1]
            Y_std = [report.metric_to_std[metric] for report in reports]
            ax_std.plot(X[::std_poll_step], Y_std[::std_poll_step],
                        label=goal.pick_goal_str())
            ax_std.set_title('Std change over polls ({})'.format(answer))
            ax_std.set_ylabel("Standard Deviation")

    for ax in axarr:
        ax.legend(loc='upper right')


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


def plot_picked_features(axarr, soliconfig_to_stats, answer_count=2, poll=100):
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
    features = list(soliconfig_to_stats.values())[0].features
    features = sorted(features, key=lambda f: f.name)
    configs = list(soliconfig_to_stats.keys())
    X = np.arange(len(features))

    answer_to_goal_stats = partition_goal_by_answer(soliconfig_to_stats)
    subpl_idx = 0
    for answer, goal_to_stats in answer_to_goal_stats.items():
        feature_to_per_config_counts = get_aspect_picked_counts(goal_to_stats,
                                                                poll=poll)
        ax = axarr[subpl_idx]
        # plot each goal's stats by a set of horizontal bars
        for i, config in enumerate(configs):
            Y = [config_to_count[config]
                 for config_to_count in feature_to_per_config_counts.values()]
            ax.plot(X, Y, label=config.pick_goal_str())

        ax.set_xticks(X)
        ax.set_xticklabels([feature.name for feature in features])
        ax.set_title("Rating count after {} polls ({})".format(poll, answer))
        ax.set_ylabel("# Ratings")
        ax.legend(loc='upper right')
        subpl_idx += 1


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


def plot_aspect_star_counts(ax, product, aspect_to_star_counts):
    """Bar plot of normalized rating distribution of a single product.

    Args:
        ax: Axes
        product: str
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
    """
    aspect_to_star_counts = OrderedDict(
            sorted(aspect_to_star_counts.items(), key=lambda kv: kv[0]))
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
    ax.legend(loc='upper right')
    ax.set_ylabel('# Ratings')


def plot_aspect_rating_dist(axarr, product, aspect_to_star_counts):
    """Bar plot of normalized rating distribution of a single product.

    Args:
        axarr: list of 2 Axes
            one Axes for std plot, one Axes for violin plot
        product: str
        aspect_to_star_counts: dict,
            aspect -> star_counts
                star_counts: dict, star -> count
    """
    aspect_to_star_counts = OrderedDict(
            sorted(aspect_to_star_counts.items(), key=lambda kv: kv[0]))

    aspects = list(aspect_to_star_counts.keys())
    aspect_to_raw_stars = get_raw_stars(aspect_to_star_counts)
    raw_stars = [np.array(stars) for stars in aspect_to_raw_stars.values()]

    X = np.arange(len(aspects)) + 1
    aspect_stds = [np.std(stars) for stars in aspect_to_raw_stars.values()]

    ax_std, ax_violin = axarr
    ax_std.plot(X, aspect_stds)
    ax_std.set_ylabel("Standard Deviation")
    ax_std.set_title("Dataset's rating distribution (used in generator)")
    for x, std in zip(X, aspect_stds):
        ax_std.annotate("{:.2f}".format(std), xy=(x, std + 0.05), ha="center")

    ax_violin.violinplot(raw_stars, showmeans=True)
    ax_violin.set_ylabel("Star")
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
