"""
    Utility functions
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover


"""
Data Preprocessing
"""


def transform_id(df, var):
    df[var] = df[var].str.replace(",", "").astype(int)
    return df


def transform_gender(df, var, male="M"):
    df[var] = (df[var] == male).astype(int)
    return df


def transform_age_bin(df, dataset_name, var, save_encoding=True):
    unique_values = sorted(df[var].unique().tolist())
    age_bin_encoding = dict(zip(unique_values, [format(x, "d") for x in range(len(unique_values))]))
    df[var] = df[var].replace(age_bin_encoding).astype(int)

    if save_encoding:
        with open(f"utils/encodings/{dataset_name}_age_bin_encoding.json", "w") as f:
            json.dump(age_bin_encoding, f)
    return df


def transform_categorical_column(df, column_name, dataset_name, save_encoding=True):
    unique_values = sorted(df[column_name].unique().tolist(), key=lambda x: str(x))
    encoding = dict(zip(unique_values, [format(x, "d") for x in range(len(unique_values))]))
    df[column_name] = df[column_name].replace(encoding).astype(int)

    if save_encoding:
        with open(
            f"utils/encodings/{dataset_name}_{column_name}_encoding.json", "w"
        ) as f:
            json.dump(encoding, f)

    return df, encoding




def process_full_dataset(df, dataset_name):
    df_processed = df.copy()
    df_processed = transform_gender(df_processed, var="gender")
    df_processed = transform_age_bin(df_processed, dataset_name, var="age_bin")
    for column in ["diagnosis"]:
        df_processed, _ = transform_categorical_column(
            df_processed, column, dataset_name
        )

    return df_processed


def plot_correlation_matrix(df, selected_columns, cmap="bwr", figsize=(10, 10)):
    """
    Plot the correlation matrix for selected columns of a DataFrame.

    Parameters:
    - df: DataFrame to analyze.
    - selected_columns: List of column names to include in the correlation matrix.
    - cmap: Color map for the matrix plot.
    - figsize: Tuple indicating figure size.
    """
    # Calculate correlation matrix
    correlation = df[selected_columns].corr().round(2)

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(correlation, vmin=-1, vmax=1, cmap=cmap)
    fig.colorbar(cax)

    # Add text annotations
    for (i, j), val in np.ndenumerate(correlation.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center")

    # Set ticks and labels
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)

    plt.tight_layout()
    plt.show()


