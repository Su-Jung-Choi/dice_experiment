"""
// Written by: Sujung Choi
// Date: 9/25/2024
// for run1 data
// this script generate cf with three different methods: 1. random, 2. genetic, 3. kdtree
// generate cfs for each of the methods for each of the feature sets: 1. feature_set1, 2. feature_set2, 3. all_features with pre-defined permitted ranges
// for each of the methods, run kmeans clustering with k = 2, 3, 4, or 5
// run DBSCAN
// calculate euclidean distances with a bad sample and the rest samples and visualize the distance graph
// find the min and max values within the cluster 0 and label 0 (good design) and set a new permitted range
// generate cfs for the new permitted range
"""

import timeit
import glob
import os
import pandas as pd
import numpy as np
import random
import pickle
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # to create a custom line in a plot
import seaborn as sns

# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# DiCE imports
import dice_ml
from dice_ml.utils import helpers  # helper functions

np.random.seed(42)
random.seed(42)


# load the dataset
def read_data(file_name, target_class):
    # Read the first row of the file to know the number of columns
    first_row = pd.read_csv(file_name, nrows=1, index_col=0)
    num_columns = len(first_row.columns)

    # Read the data into pandas df, set appropriate column names, and drop the first index column
    data = pd.read_csv(
        file_name,
        names=[f"c{i+1}" if i < 4 else f"x{i-3}" for i in range(num_columns)],
        index_col=0,
    )

    # For classification labels 'c2', 'c3', 'c4', replace all values greater than 0 with 1
    for label in ["c2", "c3", "c4"]:
        data[label] = data[label].apply(lambda x: 1 if x > 0 else 0)

    # prepare data without any labels
    data_wo_label = data.drop(columns=["c1", "c2", "c3", "c4"])

    if target_class == "c2":
        c2_data = data.drop(columns=["c1", "c3", "c4"])

        return c2_data, data_wo_label

    elif target_class == "c3":
        c3_data = data.drop(columns=["c1", "c2", "c4"])

        return c3_data, data_wo_label

    elif target_class == "c4":
        c4_data = data.drop(columns=["c1", "c2", "c3"])

        return c4_data, data_wo_label

    else:
        raise ValueError("Invalid target_class. Expected 'c2', 'c3', or 'c4'.")


# compute the kmeans and return the updated df with the number of counted labels in each cluster
def kmeans_compute(df_name, df_wo_label_name, col_name):
    # use K-Means clustering
    X = df_wo_label_name
    # change the n clusters to 3, 4, or 5
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)

    # extract the target label column
    label = df_name[col_name]

    # add a new column named 'cluster' into the original df for the cluster labels from kmeans
    df_name["cluster"] = kmeans.labels_

    # rename the 0 and 1 into 'cluster 0' and cluster 1', respectively
    df_name["cluster"] = df_name["cluster"].replace({0: "cluster 0", 1: "cluster 1"})

    # group by clusters and count 0 and 1 in each cluster
    cluster_label_counts = (
        df_name.groupby(["cluster", label]).size().unstack(fill_value=0)
    )

    # rename the 0 (good design) and 1 (bad design) labels into 'label 0' and 'label 1', respectively
    cluster_label_counts = cluster_label_counts.rename(
        columns={0: "label 0", 1: "label 1"}
    )

    return df_name, cluster_label_counts

# generate plots for DBSCAN clustering
def DBSCAN_compute(df_name, df_wo_label_name, col_name):
    # use DBSCAN clustering
    X = df_wo_label_name
    X = StandardScaler().fit_transform(X)

    # test PCA to reduce the dimensionality
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    dbscan = DBSCAN() # default eps=0.5, min_samples=5
    labels = dbscan.fit_predict(X)
    
    # to do similar to kmeans_compute
    #dbscan.fit(X)
    #labels = dbscan.labels_
    #print(labels)

    # create scatter plot
    plt.figure(figsize=(8,6))
    # get the unique labels
    unique_labels = np.unique(labels)
    # plot each cluster with a different color
    for label in unique_labels:
        if label == -1:
            # plot noise points in black
            color = 'k'
            marker = 'x'
            label_text = "Noise"
        else:
            color = plt.cm.Spectral(float(label) / len(unique_labels))
            marker = 'o'
            label_text = f"Cluster {label}"
        # plot the points that belong to the current cluster
        plt.scatter(X[labels==label][:,0], X[labels==label][:,1], c=[color], label=label_text, marker=marker, s=50, edgecolor='k')
    plt.title("DBSCAN Clustering", fontsize=16)
    plt.legend(loc='upper right')
    # save the plot to a file
    plt.savefig("dbscan_plot.png")
    plt.close()

# plot the kmeans results and save each plot in the 'cluster' folder
def plot_kmeans_results(cluster_label_counts, dataset_name, label_name):
    title_name = dataset_name.split("_")[0] + "_" + label_name
    # create a new figure
    plt.figure(figsize=(6, 4))
    # visualize the bar chart
    cluster_label_counts.plot(kind="bar", color=["blue", "red"])
    plt.title(f"Count of Labels in Each Cluster ({title_name})")
    plt.xlabel("Clusters")
    plt.ylabel("Count")
    plt.xticks(range(len(cluster_label_counts)), cluster_label_counts.index, rotation=0)
    plt.legend(title="Actual Label")

    # save the figure to the 'cluster' folder
    if not os.path.exists("cluster"):
        os.makedirs("cluster")

    # change the k name according to the number of clusters
    plot_filename = os.path.join("cluster", f"{title_name}_k2_plot.png")
    plt.savefig(plot_filename)
    plt.close()


# compute the distance from sample to the rest of the points within the cluster 0 and order them from shortest to longest
def compute_distance(clustering_df, label):
    # filter only cluster 0
    cluster0 = clustering_df[clustering_df["cluster"] == "cluster 0"]
    # select a random bad design sample (denoted by 1 in the label) within cluster 0
    sample = cluster0[cluster0[label] == 1].sample(n=1, random_state=42)

    # get the index for the sample
    sample_index = sample.index[0]

    # remove the sample from the cluster1
    rest_data = cluster0.drop(index=sample_index)

    # delete the label and 'cluster' columns before calculating euclidean distances
    sample_wo_labels = sample.drop(
        columns=[label, "cluster"]
    ).values  # convert df to np array

    rest_data_wo_labels = rest_data.drop(
        columns=[label, "cluster"]
    ).values  # convert df to np array

    # compute the euclidean distances for each column between the sample and the rest of values
    distances = np.linalg.norm(rest_data_wo_labels - sample_wo_labels, axis=1)
  

    # add a new column named 'distance' into the rest_data df
    rest_data["distance"] = distances

    # separate the data into two groups based on the label and sort them by the distance
    group_blue = rest_data[rest_data[label] == 0].sort_values("distance")
    group_red = rest_data[rest_data[label] == 1].sort_values("distance")

    # concatenate the sorted groups back together
    ordered_data = pd.concat([group_blue, group_red])

    return ordered_data, sample_index


def generate_barplot(ordered_data, label, plot_name):
    # extract the 'distance' column from the ordered data and convert the df to np array
    ordered_distance = ordered_data["distance"].values

    # extract the label column to set aside the actual labels for later use
    ordered_labels = ordered_data[[label]]

    # set the radius to be the maximum of distances
    # radius = np.max(distances)
    plt.figure(figsize=(12, 8))

    # Define colors for different labels
    colors = ["blue" if label == 0 else "red" for label in ordered_labels[label]]

    # Plot the bar chart
    plt.bar(range(len(ordered_distance)), ordered_distance, color=colors)

    # Set labels and title
    plt.xlabel("Points", fontsize=14)
    plt.ylabel("Distance from Sample", fontsize=14)
    plt.title("Distances from Sample Point to Other Points", fontsize=16)

    # Create custom legend
    handles = [
        Line2D(
            [0], [0], color="blue", linestyle="-", linewidth=10, label="Label 0: Blue"
        ),
        Line2D(
            [0], [0], color="red", linestyle="-", linewidth=10, label="Label 1: Red"
        ),
    ]

    plt.legend(handles=handles, fontsize=12)

    # create a directory to save the plot
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # save the plot to a file
    output_path = os.path.join("plots", plot_name)
    plt.savefig(output_path)
    # plt.show()


# to get the lower bound and upper bound for each actual label (0 and 1) within cluster 0
def find_min_max(clustering_df, label_name):
    cluster0 = clustering_df[clustering_df["cluster"] == "cluster 0"]

    # extract only the ones with actual label 0
    cluster0_label0 = cluster0[cluster0[label_name] == 0]
    cluster0_label0 = cluster0_label0.drop(columns=[label_name, "cluster"])

    # get the minimum and maximum values for each column
    label0_min_values = cluster0_label0.min().tolist()  # convert to list
    label0_max_values = cluster0_label0.max().tolist()

    # extract only the ones with actual label 1
    cluster0_label1 = cluster0[cluster0[label_name] == 1]
    cluster0_label1 = cluster0_label1.drop(columns=[label_name, "cluster"])

    # pair the min and max for each feature
    label0_min_max_pairs = [
        [min_val, max_val]
        for min_val, max_val in zip(label0_min_values, label0_max_values)
    ]

    return label0_min_max_pairs


def generate_object(target_class, dataset):
    # extract the target label
    target = dataset[target_class]
    # split the data into training and testing sets
    train_dataset, test_dataset, y_train, y_test = train_test_split(
        dataset, target, test_size=0.2, random_state=0, stratify=target
    )

    # construct a data object for DiCE with train dataset.
    # Specify the names of the continuous features and the name of the output variable that the ML model will predict.
    # (this dataset contains only continuous features so specify all of them)
    continuous_features = [f"x{i}" for i in range(1, 35)]
    data_object = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=continuous_features,
        outcome_name=target_class,
    )
    return data_object


def create_data_object(file_name, target_class, sample_idx):
    # Read the first row of the file to get the number of columns
    first_row = pd.read_csv(file_name, nrows=1, index_col=0)
    num_columns = len(first_row.columns)

    # Read the data ('run*_data.csv') into pandas dataframe, set appropriate column names, and drop the first index column
    data = pd.read_csv(
        file_name,
        names=[f"c{i+1}" if i < 4 else f"x{i-3}" for i in range(num_columns)],
        index_col=0,
    )
    # create a new label 'c5' by adding the values of 'c2', 'c3', 'c4'
    data["c5"] = data["c2"] + data["c3"] + data["c4"]

    # For classification labels, replace all values greater than 0 with 1
    for label in ["c2", "c3", "c4", "c5"]:
        data[label] = data[label].apply(lambda x: 1 if x > 0 else 0)

    # extract the bad design sample for later use
    sample = data.iloc[[sample_idx]]
    sample = sample.drop(columns=["c1", "c2", "c3", "c4", "c5"])
    # drop the bad design sample from the original df
    data.drop(sample_idx, inplace=True)

    # Split the data into four different dataframes, each with one label
    c2_data = data.drop(columns=["c1", "c3", "c4", "c5"])
    c3_data = data.drop(columns=["c1", "c2", "c4", "c5"])
    c4_data = data.drop(columns=["c1", "c2", "c3", "c5"])
    c5_data = data.drop(columns=["c1", "c2", "c3", "c4"])

    if target_class == "c2":
        data_object = generate_object("c2", c2_data)

    elif target_class == "c3":
        data_object = generate_object("c3", c3_data)

    elif target_class == "c4":
        data_object = generate_object("c4", c4_data)

    elif target_class == "c5":
        data_object = generate_object("c5", c5_data)

    else:
        raise ValueError("Invalid target_class. Expected 'c2', 'c3', 'c4', or 'c5'.")

    return data_object, sample


def generate_cf(
    model_name,
    label,
    feature_set,
    feature_set_name,
    permitted_ranges,
    data_object,
    sample,
    file,
):
    try:
        with open(model_name, "rb") as f:
            model = pickle.load(f)

        # Using sklearn backend
        dice_model = dice_ml.Model(model=model, backend="sklearn")

        # create an instance of the Dice class, which is used to generate counterfactual explanations
        explanation_instance = dice_ml.Dice(data_object, dice_model, method="random")

        # generate 10 counterfactuals that can change the original outcome (1) to desired class (0)
        counterfactuals = explanation_instance.generate_counterfactuals(
            sample,
            total_CFs=10,
            desired_class="opposite",
            features_to_vary=feature_set,
            permitted_range=permitted_ranges,
        )

        # get the counterfactuals as a pandas dataframe
        counterfactuals = counterfactuals.cf_examples_list[0].final_cfs_df

        # if the counterfactuals df is not empty
        if not counterfactuals.empty:
            # drop the label
            cf_without_label = counterfactuals.drop(columns=[label])
            # convert the counterfactuals to a list of lists
            counterfactual_list = cf_without_label.values.tolist()

            extracted_model_name = model_name.split(".")[0]
            # save the counterfactuals to a txt file
            file.write(
                f"# {extracted_model_name} {feature_set_name} 10 counterfactuals:\n"
            )

            if len(counterfactual_list) == 10:
                for i, row in enumerate(counterfactual_list):
                    comma_separated_row = ",".join(map(str, row))
                    file.write(f"genome = [{comma_separated_row}]\n")

            else:  # if less than 10 cfs are found
                for i, row in enumerate(counterfactual_list):
                    comma_separated_row = ",".join(map(str, row))
                    file.write(f"genome = [{comma_separated_row}]\n")
                file.write("No further cfs found.\n")

        else:  # if no cfs are found
            file.write(
                f"{extracted_model_name} {feature_set_name} No counterfactuals found.\n"
            )

    except Exception as e:
        file.write(f"Error in generating cfs for {feature_set_name}: {str(e)}\n")


# to define the fixed permitted range
def predefined_range(file_name):
    # to save the permitted_ranges depending on the dataset
    permitted_ranges = {}

    # set the permitted range for x1 to x20
    for i in range(1, 21):
        if i % 2 != 0:  # odd numbers
            permitted_ranges[f"x{i}"] = [0, 100]
        else:  # even numbers
            permitted_ranges[f"x{i}"] = [0, 43]

    # set the permitted range for x21 to x34
    for i in range(21, 35):
        if file_name == "run1_data.csv":
            permitted_ranges[f"x{i}"] = [0.5, 1.5]  # for run1 data
        elif file_name == " run2_data.csv":
            permitted_ranges[f"x{i}"] = [0.5, 1.2]  # for run2 data
        elif file_name == "run3_data.csv":
            permitted_ranges[f"x{i}"] = [0.5, 1]  # for run3 data
        elif file_name == "run4_data.csv":
            permitted_ranges[f"x{i}"] = [0.3, 0.8]  # for run4 data
        else:
            raise ValueError("Invalid file name for permitted_ranges.")
    return permitted_ranges


# to define the permitted range based on the min and max values for good design that grouped into the right cluster
def define_range(min_max_pairs):
    permitted_ranges = {}
    # set the permitted range for x1 to x34
    for i in range(1, 35):
        permitted_ranges[f"x{i}"] = min_max_pairs[i - 1]
    return permitted_ranges


def save_cf_to_txt(
    file_name,
    target_class,
    saved_model,
    feature_sets,
    sample_index,
    min_max_range,
    folder_name,
):
    permitted_ranges = min_max_range
    data_object, sample = create_data_object(file_name, target_class, sample_index)

    # create a directory to save the cf results
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save the counterfactuals to a txt file
    extracted_model_name = saved_model.split(".")[0]

    txt_file_path = os.path.join(folder_name, f"{extracted_model_name}_cfs.txt")

    # Check if the txt file already exists
    if os.path.exists(txt_file_path):
        print(
            f"File {txt_file_path} already exists, skipping counterfactual generation."
        )
        return  # Skip counterfactual generation and exit the function

    # If file doesn't exist, proceed to generate counterfactuals
    with open(txt_file_path, "w") as file:
        for feature_set, feature_set_name in feature_sets:
            # generate cf for each feature set
            generate_cf(
                saved_model,
                target_class,
                feature_set,
                feature_set_name,
                permitted_ranges,
                data_object,
                sample,
                file,
            )


def process_steps(file_name, target_class, saved_model, feature_sets):
    data, data_wo_label = read_data(file_name, target_class)

    # call kmeans_compute function to get the df with kmeans clustering results for the dataset
    clustering_df, cluster_counts = kmeans_compute(data, data_wo_label, target_class)

    # call DBSCAN_compute function to generate plots for DBSCAN clustering
    #DBSCAN_compute(data, data_wo_label, target_class)

    # plot the kmeans results
    plot_kmeans_results(cluster_counts, file_name, target_class)
    ordered_data, sample_index = compute_distance(clustering_df, target_class)

    # generate bar plot
    extracted_file_name = file_name.split("_")[0]
    generate_barplot(
        ordered_data, target_class, f"{extracted_file_name}_{target_class}_barplot.png"
    )

    # display the upper and lower bounds for each label within cluster 1
    min_max_range = find_min_max(clustering_df, target_class)

    save_cf_to_txt(
        file_name,
        target_class,
        saved_model,
        feature_sets,
        sample_index,
        predefined_range(file_name),
        "dice_results",
    )
    new_min_max_range = define_range(min_max_range)
    save_cf_to_txt(
        file_name,
        target_class,
        saved_model,
        feature_sets,
        sample_index,
        new_min_max_range,
        "results",
    )


def main():
    start = timeit.default_timer()
    feature_set1 = [f"x{i}" for i in range(1, 21)]
    feature_set2 = [f"x{i}" for i in range(21, 35)]
    all_features = feature_set1 + feature_set2

    feature_sets = [
        (feature_set1, "feature_set1"),
        (feature_set2, "feature_set2"),
        (all_features, "all_features"),
    ]

    # run1 c2 MLP
    process_steps(
        "run1_data.csv", "c2", "run1_data_c2_mlp_classifier.pkl", feature_sets
    )

    # run1 c3 KNN
    process_steps(
        "run1_data.csv", "c3", "run1_data_c3_knn_classifier.pkl", feature_sets
    )

    # run1 c4 MLP
    process_steps(
        "run1_data.csv", "c4", "run1_data_c4_mlp_classifier.pkl", feature_sets
    )

    # run1 c5 DTC
    # process_steps("run1_data.csv", "c5", "run1_data_c5_dtc.pkl", feature_sets)

    stop = timeit.default_timer()
    time_seconds = stop - start
    time_minutes = time_seconds / 60
    print(f"Time: {time_minutes} minutes")


if __name__ == "__main__":
    main()
