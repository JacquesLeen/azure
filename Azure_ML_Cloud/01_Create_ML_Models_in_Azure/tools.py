import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

def show_density(data, column):
    """
    Plots the density distribution of the given data using seaborn's displot with KDE.

    Parameters:
    ----------
    data : pandas.DataFrame
        The data for which to plot the density distribution.
    column : str
        The column for which to represent the density plot.

    Returns:
    -------
    matplotlib.figure.Figure
        The figure object containing the density plot.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'StudyHours': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]})
    >>> fig = show_density(data, column='StudyHours')
    >>> fig.show()
    """
    
    plt.style.use('fivethirtyeight')
    # Create the density plot using seaborn
    ax = sns.displot(
        data=data,
        x=column,
        kind='kde',
        height=4,  # Set the height of the figure
        aspect=10/4,  # Set the aspect ratio to achieve the desired width
        color='powderblue'
    )

    #compute statistical indicators
    mean = data[column].mean()
    median = data[column].median()
    mode = data[column].mode()[0]

    # add vertical lines
    plt.axvline(x=mean, color='royalblue', linestyle='--', linewidth=2, label='mean')
    plt.axvline(x=median, color='mediumblue', linestyle='--', linewidth=2, label='median')
    plt.axvline(x=mode, color='navy', linestyle='--', linewidth=2, label='mode')
    plt.legend()
    plt.title(f'Densty Plot {column}')
    return ax

    


def show_distribution(data, column):
    """
    Plots the distribution of the given data using a histogram and a boxplot, and highlights key statistical measures.

    Parameters:
    ----------
    data : pandas.DataFrame or pandas.Series
        The data for which to plot the distribution.
    column : str
        The column for which to represent the distribution plot.

    Returns:
    -------
    matplotlib.figure.Figure
        The figure object containing the distribution plots.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'StudyHours': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]})
    >>> fig = show_distribution(data, column='StudyHours')
    >>> fig.show()
    """

    plt.style.use('fivethirtyeight')
    # compute stat measures
    min_val = data[column].min()
    max_val = data[column].max()
    mean= data[column].mean()
    median = data[column].median()
    mode = data[column].mode()[0]

    #create frame
    fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
    #histogram
    sns.histplot(ax=ax[0],
                data=data,
                x=column,
                color='powderblue'
    )
    ax[0].set_ylabel('Frequency')
    #add lines
    ax[0].axvline(x=min_val, color='deepskyblue', linestyle='--', linewidth=2, label='min')
    ax[0].axvline(x=max_val, color='dodgerblue', linestyle='--', linewidth=2, label='max')
    ax[0].axvline(x=mean, color='royalblue', linestyle='--', linewidth=2, label='mean')
    ax[0].axvline(x=median, color='mediumblue', linestyle='--', linewidth=2, label='median')
    ax[0].axvline(x=mode, color='navy', linestyle='--', linewidth=2, label='mode')
    ax[0].legend()
    #boxplot
    sns.boxplot(ax= ax[1],
                x=data[column],
                color= 'powderblue',
                linewidth=2)
    ax[1].set_xlabel(f'{column}')

    fig.suptitle('Distribution Data')
    fig.show()