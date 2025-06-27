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


DEFAULT_COLS = [
    "Diagnosis",
    "Disparate_Impact",
    "Statistical_Parity_Difference",
    # "DIDI",
]


"""
Data Exploration
"""


def get_rank_n_candidates(dataset, match_rank):
    return dataset[dataset["match_rank"] == match_rank]


def discretize_feature(data):
    distances_km_discrete = np.zeros(10)
    for dist in data:
        distances_km_discrete[int(dist // 10)] += 1
    return distances_km_discrete


def create_dictionary_from_series(series):  # in percentage
    dict_series = {}
    total = np.sum(series.values)
    for idx, val in zip(series.index, series.values):
        dict_series[idx] = np.around((val / total), 4)
    return dict_series


def create_dicts_rank_n(dataset, cols):
    dict_list = []
    distances_km = discretize_feature(dataset.distance_km)
    total_distances = np.sum(distances_km)
    dict_distances = {}
    for i in range(10):
        dict_distances[i] = np.around(distances_km[i] / total_distances, 4)
    dict_list.append(dict_distances)

    for col in cols:
        dict_list.append(create_dictionary_from_series(dataset[col].value_counts()))

    return dict_list


def create_table_for_feature(list_dict, idx=0):
    selected_dicts = [sublist[idx] for sublist in list_dict]

    total_keys = selected_dicts[0].keys()
    for dictionary in selected_dicts[1:]:
        for key in total_keys:
            if key not in dictionary.keys():
                dictionary[key] = 0

    data = [list(d.values()) for d in selected_dicts]

    return pd.DataFrame(np.vstack(data), columns=list(selected_dicts[0].keys()))


def show_global_distribution(df, feature):
    value_counts = df[feature].value_counts()

    plt.bar(value_counts.index, value_counts.values, color="skyblue")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()


def print_feature_distribution(dataframe, title):  # across rank
    dataframe.plot(kind="bar", stacked=True)
    plt.title(f"Distribution of {title} by Rank")
    plt.xlabel("Rank")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(title=title)
    plt.show()


def plot_2_features(
    df,
    feature1,
    feature2,
    num_ranks=[1, 2],
    num_cols=2,
    legend_outside=False,
    response=None,
    x_axis_rotation=90,
):
    data = []
    if num_ranks == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
        distribution = (
            df.groupby(feature1)[feature2].value_counts(normalize=True).unstack()
        )
        data.append(distribution)
        distribution.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Full dataset")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if legend_outside:
            ax.legend(
                title=feature2,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="small",
            )
        else:
            ax.legend(title=feature2)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_axis_rotation)
        plt.show()
    else:

        fig, axs = plt.subplots(1, num_cols, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(
            f"{feature1} Distribution by {feature2} for Different Ranks", fontsize=16
        )
        for i, rank in enumerate(num_ranks):
            if len(num_ranks) == 1:  # Handle single rank differently
                ax = axs
            else:
                ax = axs[i]
            new_df = df[df.match_rank == rank]
            distribution = (
                new_df.groupby(feature1)[feature2]
                .value_counts(normalize=True)
                .unstack()
            )
            distribution.plot(kind="bar", stacked=True, ax=ax)
            data.append(distribution)
            ax.set_title(f"Rank {rank}")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if legend_outside:
                ax.legend(
                    title=feature2,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize="small",
                )
            else:
                ax.legend(title=feature2)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=x_axis_rotation)


        plt.show()

    if response != None:
        return data


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


"""
Data Analysis
"""

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


"""
Bias Analysis
"""

def test_bias(df, protected_attr_col, attr_favorable_value,
              target_var: str = 'diagnosis'):
    """
    Test bias for each diagnosis in the DataFrame.

    Because this code is repurposed from an earlier analysis in the context of
    HR use cases, the terminology sector now refers to a diagnosis instead of
    an employement sector.

    Parameters:
    - df: DataFrame to analyze.
    - protected_attr_col: Protected attribute column name.
    - attr_favorable_value: Protected attribute value.

    Returns:
    - DataFrame containing fairness metrics for each diagnosis.
    """
    results = []

    for diagnosis_name, group in df.groupby(target_var):
        group_df = group.copy()

        bld = BinaryLabelDataset(
        df=group_df[['present', protected_attr_col]],
            label_names=['present'],
            protected_attribute_names=[protected_attr_col]
        )

        metric = BinaryLabelDatasetMetric(
            bld,
            privileged_groups=[{protected_attr_col: attr_favorable_value}],
            unprivileged_groups=[{protected_attr_col: 0}],
        )

        didi = DIDI_r(group_df, group_df["present"], {protected_attr_col: [1]})

        results.append({
            'diagnosis': diagnosis_name,
            'disparate_impact': metric.disparate_impact(),
            'statistical_parity_difference': metric.statistical_parity_difference(),
            'DIDI': didi
        })

    return pd.DataFrame(results)


def DIDI_r(data, pred, protected):
    res, avg = 0, np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = data[aname] == val
            res += abs(avg - np.mean(pred[mask]))
    return res


def show_bias(df, protected_attr_col, attr_favorable_value, target_var='diagnosis',
              plot_histogram=False):
    """
    Show bias analysis for the specified protected attribute column.

    Because this function is repurposed from an earlier analysis on a more
    extensive data set, the meaning of sector changed to diagnosis.

    Parameters:
    - df: DataFrame to analyze.
    - protected_attr_col: Protected attribute column name.
    - attr_favorable_value: Protected attribute value.
    - plot_histogram: Boolean indicating whether to plot histograms for each sector.

    Returns:
    - DataFrame containing fairness metrics for each sector.
    """

    all_sector_metrics = test_bias(df, protected_attr_col, attr_favorable_value)
    all_sector_metrics.to_csv("Results/bias_analysis_" + protected_attr_col + ".csv")

    if plot_histogram:
        for sector in df[target_var].unique():
            plot_histogram_metric(
                all_sector_metrics,
                "Disparate_Impact",
                sector,
                protected_attr_col,
                save=True,
            )
            plot_histogram_metric(
                all_sector_metrics,
                "Statistical_Parity_Difference",
                sector,
                protected_attr_col,
                save=True,
            )
            plot_histogram_metric(
                all_sector_metrics, "DIDI", sector, protected_attr_col, save=True
            )

    return all_sector_metrics


def plot_histogram_metric(df, metric, sector, protected_attr_col, target_var='diagnosis',
                          save=True):
    """
    Plot a histogram for the specified metric in the specified sector.

    Parameters:
    - df: DataFrame containing the metrics.
    - metric: Metric to plot.
    - sector: Sector to analyze.
    - protected_attr_col: Protected attribute column name.
    - save: Boolean indicating whether to save the plot. Default is True. If False, the plot is displayed.
    """
    df_sector = df[df[target_var] == sector]
    plt.figure(figsize=(8, 6))
    plt.hist(df_sector[metric], color="skyblue", bins=20, edgecolor="black")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.title(f"{metric} Distribution for {protected_attr_col} in {sector}")
    plt.tight_layout()

    if save:
        plt.savefig(
            f"Results/Plots/Histogram_{protected_attr_col}_{metric}_{sector}.png"
        )
        plt.clf()
        plt.close()
    else:
        plt.show()


def compute_repaired_df(df, sector, protected_attribute, level=0.8):
    """
    Compute the repaired DataFrame for the specified sector and protected attribute.

    Because this code is repurposed from an earlier analysis in the context of
    HR use cases, the term "sector" now refers to a diagnosis instead of
    an employement sector.

    Parameters:
    - df: DataFrame to analyze.
    - sector: Diagnosis to analyze.
    - protected_attribute: Protected attribute column name.

    Returns:
    - Original DataFrame for the specified sector and protected attribute.
    - Repaired DataFrame for the specified sector and protected attribute.
    """
    sector_df = df[df["diagnosis"] == sector]

    binaryLabelDataset = BinaryLabelDataset(
        df=sector_df[['present', 'gender']],
        label_names=['present'],
        protected_attribute_names=[protected_attribute]
    )

    di = DisparateImpactRemover(repair_level=level)

    bld_repaired = di.fit_transform(binaryLabelDataset)

    job_df_orig = binaryLabelDataset.convert_to_dataframe()[0]
    job_df_repaired = bld_repaired.convert_to_dataframe()[0]

    return job_df_orig, job_df_repaired


def compute_bias_differences_percentage(df, sectors, protected_attribute, columns, mode='percentage'):
    """
    Compute the differences between the original and repaired DataFrames for each sector.

    Because this code is repurposed from an earlier analysis in the context of
    HR use cases, the term "sector" now refers to a diagnosis instead of
    an employement sector.

    Parameters:
    - df: DataFrame to analyze.
    - sectors: List of diagnoses to analyze.
    - protected_attribute: Protected attribute column name.
    - columns: List of columns to analyze.
    - mode: percentage or total.

    Returns:
    - DataFrame containing the differences between the original and repaired DataFrames for each sector.
    """

    results_df = pd.DataFrame(columns=columns)

    for sector in sectors:
        job_df_orig, job_df_repaired = compute_repaired_df(
            df, sector, protected_attribute
        )
        differences_list = []
        for column in job_df_orig.columns[:-1]:  # do not compute for idoneous
            differences = job_df_orig[column] != job_df_repaired[column]
            num_differences = differences.sum()
            if mode == 'percentage':
                total_count = job_df_orig.shape[0]
                percentage = round((num_differences / total_count) * 100, 2)
                differences_list.append(percentage)
            else:
                differences_list.append(num_differences)

        differences_df = pd.DataFrame([differences_list], columns=columns)
        results_df = pd.concat([results_df, differences_df], ignore_index=True)

    return results_df


def plot_series(series, title, xlabel, ylabel="Count"):
    plt.bar(
        series.index,
        series.values,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.show()


def compare_plot(
    original, repaired, labels, title, xlabel, ylabel="Count", size=(6, 6)
):
    width = 0.4
    plt.figure(figsize=size)
    x = np.arange(len(labels))
    plt.bar(x - width / 2, original, width, label="Original", color="skyblue", alpha=1)
    plt.bar(x + width / 2, repaired, width, label="Repaired", color="orange", alpha=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.show()


def prepare_different_series(series1, series2):
    all_index = sorted(set(series1.index).union(set(series2.index)))
    orig_counts = series1.reindex(all_index, fill_value=0)
    repaired_counts = series2.reindex(all_index, fill_value=0)
    return orig_counts, repaired_counts, all_index
