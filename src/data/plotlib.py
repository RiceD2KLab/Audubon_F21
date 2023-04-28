import matplotlib.pyplot as plt


def get_cmap(num, name='tab20c'):
    '''
    Return a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    '''
    return plt.cm.get_cmap(name, num)


def plot_distribution(data_frame, col_name,
                      x_label, y_label, title, path=None, filt=None):
    '''
    Plot a barchart of the value counts of a column in a dataframe.

    Input:
        data_frame : Pandas dataframe containing the column to plot.
        col_name : The name of the column to plot.
        x_label : The x-label of the plot.
        y_label : The y-label of the plot.
        title : The title of the plot.
        path : The directory in which to save the plot.
        filt : int or None. If not None, only show categories with a count of at least `filt`.

    Output:
        A barchart of the value counts for the specified column.
    '''

    plt.rcdefaults()
    val_counts = data_frame[col_name].value_counts()

    if filt:
        val_counts = val_counts[val_counts >= filt]
    idx_list = val_counts.index.tolist()
    val_list = val_counts.values.tolist()
    cmap = get_cmap(len(idx_list))
    color_list = [cmap(idx) for idx in range(len(idx_list))]

    # make plot
    fig, axs = plt.subplots(figsize=(10, 6))
    chart = axs.bar(idx_list, val_list, color=color_list)
    axs.set_title(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.set_xticks(range(len(idx_list)))
    axs.set_xticklabels(idx_list, rotation=45, ha='right', fontsize=12)
    axs.set_ylim(0, val_counts.max() * 1.2)
    for i in range(len(chart)):
        axs.text(i, chart[i].get_height() + 0.5, chart[i].get_height(), ha='center', fontsize=10)

    if path:
        fig.savefig(path + title + '.pdf', bbox_inches='tight')
    return fig
