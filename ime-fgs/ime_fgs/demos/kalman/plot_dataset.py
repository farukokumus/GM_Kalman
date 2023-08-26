from math import ceil
import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    mpl.interactive(False)
    sns.set_style("darkgrid")
except ImportError:
    raise RuntimeError('Optional packages must be installed for this operation to succeed.')


def add_markers_to_ax(ax, x_markers, color='r', linestyle='-'):

    for marker_location in x_markers:
        ax.axvline(marker_location, color=color, linestyle=linestyle)

    return ax


def row_based_idx(num_rows, num_cols, idx):
    return np.arange(1, num_rows * num_cols + 1).reshape((num_rows, num_cols)).transpose().flatten()[idx - 1]


def add_plot_row_to_fig(signal, x_series, num_plots, plot_idx, share_ax=None, fill_by_row=True,
                        markers=None, marker_colors=None, ref_sigs=None, scatter=False, num_cols=None, bare_plot=False):
    if num_cols is not None:
        num_rows = ceil(num_plots / num_cols)
    else:
        # Determine the number of columns automatically
        if num_plots <= 4:
            num_cols = 1
            num_rows = num_plots
        else:
            num_cols = 2
            num_rows = ceil(num_plots / 2)

    if fill_by_row:
        plot_idx = row_based_idx(num_rows, num_cols, plot_idx)

    ax = plt.subplot(num_rows, num_cols, plot_idx, sharex=share_ax)
    max_num_plots = num_cols * num_rows
    less_than_max = max_num_plots - num_plots
    if scatter:
        ax.scatter(x_series, signal)
    else:
        ax.plot(x_series, signal)
    if ref_sigs is not None:
        if type(ref_sigs) is pd.DataFrame:
            num_graphs = len(ref_sigs.columns) + 1
            for colname in ref_sigs.columns:
                if scatter:
                    ax.scatter(x_series, ref_sigs[colname])
                else:
                    ax.plot(x_series, ref_sigs[colname])

        elif type(ref_sigs) is list:
            num_graphs = len(ref_sigs) + 1
            for ref_sig_df in ref_sigs:
                assert type(ref_sig_df) is pd.DataFrame and len(ref_sig_df.columns) == 2
                cols = list(ref_sig_df.columns)
                cols.remove(x_series.name)
                # Now, there should be just a single element left: the name of the data column
                assert len(cols) == 1
                colname = cols[0]
                if scatter:
                    ax.scatter(ref_sig_df[x_series.name], ref_sig_df[colname])
                else:
                    ax.plot(ref_sig_df[x_series.name], ref_sig_df[colname])
        else:
            raise TypeError

        if not bare_plot:
            # TODO: The following legends look nice but don't work with too many signals. Implement automatic legend
            # line breaks.
            # TODO: Change relative to absolute calculations to enable correct spacings when resizing
            # Shrink current axis's height by 15% on the bottom to make space for a legend.
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15,
                             box.width, box.height * 0.85])

            # Put a legend below current axis, since legends inside axes (the default) do not work well. (They move
            # around while zooming and often are hardly visible due to the data.)
            if plot_idx in range(max_num_plots - num_cols + 1 - less_than_max, max_num_plots + 1):
                # Last subplot in this column -> leave space for x label
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fancybox=True, shadow=True, ncol=num_graphs)
            else:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                          fancybox=True, shadow=True, ncol=num_graphs)

    if not bare_plot:
        ax.set_ylabel(signal.name)

    # Only print x label if this is the bottom plot of this plot column

    if not bare_plot and plot_idx in range(max_num_plots - num_cols + 1 - less_than_max, max_num_plots + 1):
        ax.set_xlabel(x_series.name)
    ax.grid(True)

    # Clean up plot
    sns.despine()

    # Add markers
    if markers is not None:
        if marker_colors is None:
            marker_colors = ['r']
        for marker_series, marker_color in zip(markers, marker_colors):
            ax = add_markers_to_ax(ax, marker_series, marker_color)

    return ax


def plot_dataset(df_to_plot, markers=None, marker_colors=None, x_series=None, ref_sigs=None,
                 title=None, scatter=False, num_cols=None, bare_plot=False):
    """
    Plot a dataset consisting of multiple measurements with a shared x axis (often, but not necessarily: time).

    Plots the columns of a dataset into individual subplots (automatically deciding between a one-column and a two-
    column layout, based on the number of columns to plot), sharing the same x axis.
    Optionally, markers representing time points of interest can be added to individual subplots, and reference signals
    can be added to individual subplots as well.
    Usage examples include plotting a set of measurements versus time, and plotting the predictors in a regression
    problem versus the residual, in order to identify unmodeled behavior in a regression model.

    :param df_to_plot: Pandas dataframe. Each column in this dataframe will be shown in a separate subplot.
    :param markers: Either None (default), a dictionary containing series objects (or lists) as values, or a list of
    such dictionaries. The keys of each dictionary must be a subset of the columns in df_to_plot, to which markers
    should be added. In the plot corresponding to the key column, a vertical line marker will be added at the locations
    specified in the corresponding value series. The dictionary values must hence each be a series of time points (float
    values, not timestamp objects).
    If a list of dictionaries is passed, this will be done for each of the dictionaries. This may be of interest to add
    multiple, different types of markers to the same plots.
    :param marker_colors: List of color strings to be used for the different timestamp marker types. If None (default),
    colors 'r', 'b', 'g', 'y' are used in this order, and then recycled.
    :param x_series: Pandas series to plot each column in df_to_plot against. Shared among all sub plots. If None
    (default), the shared x axis will be the dataframe row index.
    :param ref_sigs: Either None or a dictionary that has the columns in df_to_plot to which one or multiple reference
    signals should be shown as keys. Values should be either a dataframe of reference signals, or a list of dataframes
    each containing a single reference signal and the corresponding values of the x_series.
    :param title: Title to be displayed above all subplots.
    :param scatter: If False (default), a line plot is created. Otherwise, a scatter plot is created.
    :param num_cols: If None (default), the number of plot columns is determined automtically. Otherwise, the provided
    number of columns is employed.
    :param bare_plot: If True, no labels or legends are added to the plot at all.
    :return: Matplotlib figure object.
    """
    if x_series is None:
        x_series = df_to_plot.index

    if markers is not None and not type(markers) is list:
        markers = [markers]

    if marker_colors is None:
        marker_colors = ['r', 'b', 'g', 'y']

    cols_to_plot = [col for col in df_to_plot.columns]
    num_plots = len(cols_to_plot)

    fig = plt.figure()
    meas_series_1 = df_to_plot[cols_to_plot[0]]
    ax1 = add_plot_row_to_fig(meas_series_1, x_series, num_plots, plot_idx=1,
                              markers=None if markers is None
                              else [marker_dict[meas_series_1.name] if meas_series_1.name in marker_dict.keys()
                                    else pd.Series() for marker_dict in markers],
                              marker_colors=marker_colors,
                              ref_sigs=None if ref_sigs is None or meas_series_1.name not in ref_sigs.keys()
                              else ref_sigs[meas_series_1.name],
                              scatter=scatter,
                              num_cols=num_cols,
                              bare_plot=bare_plot)

    for jj, colname in enumerate(cols_to_plot[1:]):
        meas_series = df_to_plot[colname]
        add_plot_row_to_fig(meas_series, x_series, num_plots, plot_idx=jj + 2,
                            markers=None if markers is None
                            else [marker_dict[colname] if colname in marker_dict.keys()
                                  else pd.Series() for marker_dict in markers],
                            marker_colors=marker_colors,
                            ref_sigs=None if ref_sigs is None or meas_series.name not in ref_sigs.keys()
                            else ref_sigs[meas_series.name],
                            share_ax=ax1,
                            scatter=scatter,
                            num_cols=num_cols,
                            bare_plot=bare_plot)
    if title is not None:
        plt.suptitle(title)

    return fig


if __name__ == '__main__':
    # TODO: Add simple usage example
    pass
