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
    plt.xlabel(f'{column.capitalize()}')
    plt.title(f'Density Plot {column.capitalize()}')
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
    ax[1].set_xlabel(f'{column.capitalize()}')

    fig.suptitle(f'{column.capitalize()} Distribution Data')
    return fig


def show_distribution_cat_feature(df, column):
    """
    Plots the distribution of a categorical feature using a barplot and highlights the count of each category.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    column : str
        The column representing the categorical feature to plot.

    Returns:
    -------
    matplotlib.axes._subplots.AxesSubplot
        The Axes object containing the barplot.

    Example:
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> data = pd.DataFrame({'City': ['New York', 'New York', 'LA', 'LA', 'New York', 'LA']})
    >>> ax = show_distribution_cat_feature(data, column='City')
    >>> plt.show()
    """
    data = df[column].value_counts().sort_index().reset_index()
    palette_length = data.shape[0]
    diverging_colors = sns.color_palette("Blues", n_colors=palette_length)
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data, 
                x=column, 
                y='count',
                palette= diverging_colors
                )
    ax.set_ylabel('Count')
    ax.set_xlabel(column.capitalize())
    ax.set_title(f'{column.capitalize()} Distribution')
    return fig

def show_correlation_with_target(data, target, column):
    """
    Plots the correlation between a specified column and the target column using a scatter plot, and displays the correlation coefficient.

    Parameters:
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    target : str
        The target column to correlate with.
    column : str
        The column for which to plot the correlation against the target.

    Returns:
    -------
    matplotlib.figure.Figure
        The figure object containing the scatter plot.

    Example:
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> data = pd.DataFrame({'Target': [1, 2, 3, 4, 5], 'Feature': [2, 3, 4, 5, 6]})
    >>> fig = show_correlation_with_target(data, target='Target', column='Feature')
    >>> plt.show()
    """
    plt.style.use('fivethirtyeight')
    corr = data[column].corr(data[target])
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=data,
                    x=target,
                    y=column,
                    color='royalblue',
                    ax=ax)
    ax.set_ylabel(column.capitalize())
    ax.set_xlabel(target.capitalize())
    ax.set_title(f'{column.capitalize()} Correlation with {target.capitalize()} ({round(corr, 2)})')
    return fig

def show_correlation_cat_with_target(data, target, column):
    """
    Plots the correlation between a categorical column and the target column using a boxplot.

    Parameters:
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    target : str
        The target column to correlate with.
    column : str
        The categorical column for which to plot the correlation against the target.

    Returns:
    -------
    matplotlib.figure.Figure
        The figure object containing the boxplot.

    Example:
    --------
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> data = pd.DataFrame({'Target': [1, 2, 3, 4, 5], 'Category': ['A', 'B', 'A', 'B', 'A']})
    >>> fig = show_correlation_cat_with_target(data, target='Target', column='Category')
    >>> plt.show()
    """
    
    plt.style.use('fivethirtyeight')
    # Determine the number of unique categories in the column
    n_colors = data[column].value_counts().sort_index().reset_index().shape[0]
    
    # Generate a color palette based on the number of categories
    palette = sns.color_palette("Blues", n_colors=n_colors)
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create the boxplot on the axes
    sns.boxplot(data=data,  # Corrected 'bike_data' to 'data'
                x=column,
                y=target,
                palette=palette,
                linewidth=2,
                ax=ax)
    
    # Set the title and labels
    ax.set_title(f'Boxplot for Correlation between {column.capitalize()} and {target.capitalize()}')
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel(target.capitalize())
    
    # Return the figure object
    return fig