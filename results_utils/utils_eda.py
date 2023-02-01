import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def plot_viola(dataframe, metric, title, xlabel, ylabel, display_hline=False):
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8) 

    plot_data = [dataframe[dataframe['model_display_name'] == name][metric] for name in dataframe['model_display_name'].unique()]

    mins = dataframe.groupby(['model_display_name'])[[metric]].min().sort_values('model_display_name', ascending=True, key=lambda col: col.str.lower())
    maxes = dataframe.groupby(['model_display_name'])[[metric]].max().sort_values('model_display_name', ascending=True, key=lambda col: col.str.lower())
    medians = dataframe.groupby(['model_display_name'])[[metric]].median().sort_values('model_display_name', ascending=True, key=lambda col: col.str.lower())

    fig, ax = plt.subplots(figsize=(6, 3.54), dpi=600)

    plt.title(title)

    xticklabels = dataframe['model_display_name'].unique()

    ax.annotate('local max', xy=(.5, .5), xytext=(0.5, 0.5), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top')

    ax.set_xticks([i for i in range(1, len(xticklabels)+1)])
    ax.set_xticklabels(xticklabels)

    plt.xticks(rotation=30, ha='right')

    my_plot = ax.violinplot(plot_data, showmedians=True)

    for i, v in enumerate(medians[metric]):
        plt.text((i+.84), (v+.2), str(round(v, 2)), fontsize=7)
    for i, v in enumerate(mins[metric]):    
        plt.text((i+.84), (v+.2), str(round(v, 2)), fontsize=7)
    for i, v in enumerate(maxes[metric]):
        plt.text((i+.84), (v+.2), str(round(v, 2)), fontsize=7)

    if display_hline == True:
        plt.axhline(y = 50, color = 'r', linestyle = 'dashed')

    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.show()


def plot_box(dataframe, metric, title, display_hline=False):
    plot_data = [dataframe[dataframe['model_display_name'] == name][metric] for name in dataframe['model_display_name'].unique()]


    mins = dataframe.groupby(['model_display_name'])[[metric]].min().sort_values('model_display_name', ascending=True, key=lambda col: col.str.lower())
    maxes = dataframe.groupby(['model_display_name'])[[metric]].max().sort_values('model_display_name', ascending=True, key=lambda col: col.str.lower())
    medians = dataframe.groupby(['model_display_name'])[[metric]].median().sort_values('model_display_name', ascending=True, key=lambda col: col.str.lower())

    fig, ax = plt.subplots(figsize=(12, 8))

    plt.title(title)

    xticklabels = dataframe['model_display_name'].unique()

    ax.annotate('local max', xy=(.5, .5), xytext=(0.5, 0.5), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top')

    # ax.set_xticks([i for i in range(1, len(xticklabels)+1)])
    ax.set_xticklabels(xticklabels)

    plt.xticks(rotation=45, ha='right')


    my_plot = ax.boxplot(plot_data)


    for i, v in enumerate(medians[metric]):
        plt.text((i+1.3), (v-.1), str(round(v, 3)), fontsize = 12)
    for i, v in enumerate(mins[metric]):    
        plt.text((i+.87), (v-.7), str(round(v, 3)), fontsize = 12)
    for i, v in enumerate(maxes[metric]):
        plt.text((i+.87), (v+.2), str(round(v, 3)), fontsize = 12)

    if display_hline == True:
        plt.axhline(y = 50, color = 'r', linestyle = 'dashed')

    plt.show()
    