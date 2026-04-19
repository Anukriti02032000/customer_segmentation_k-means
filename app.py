import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# TITLE
st.title("Customer Segmentation using K-Means Clustering")

st.write(
    "Upload any customer dataset containing numeric features. "
    "The system automatically detects numeric columns, removes missing values, "
    "applies scaling, performs clustering and generates customer segment insights."
)

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # LOAD DATASET
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # DATASET PREVIEW
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # DATASET SHAPE
    st.subheader("Dataset Shape")
    st.write(df.shape)

    # NUMERIC COLUMN SELECTION
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Numeric Columns Used for Clustering")
    st.write(numeric_cols)

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns.")
        st.stop()

    # REMOVE MISSING VALUES
    numeric_df = df[numeric_cols].dropna()

    st.subheader("Dataset After Removing Missing Values")
    st.write(numeric_df.shape)

    # FEATURE SCALING
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # ELBOW METHOD
    inertia = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    st.subheader("Elbow Method for Optimal Cluster Selection")

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

    # CLUSTER SLIDER
    k = st.slider("Select Number of Clusters", 2, 10, 3)

    # APPLY KMEANS
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    numeric_df["Cluster"] = clusters

    # SHOW CLUSTERED DATA
    st.subheader("Clustered Dataset Preview")
    st.dataframe(numeric_df.head())

    # CLUSTER SUMMARY
    st.subheader("Cluster Insights Summary")

    cluster_summary = numeric_df.groupby("Cluster").mean()
    st.dataframe(cluster_summary)

    # AUTOMATIC SEGMENT INTERPRETATION
    st.subheader("Automatic Customer Segment Interpretation")

    cluster_labels = {}

    sorted_clusters = cluster_summary.mean(axis=1).sort_values().index

    cluster_labels[sorted_clusters[0]] = "Low Value Customers"

    if len(sorted_clusters) > 1:
        cluster_labels[sorted_clusters[1]] = "Medium Value Customers"

    if len(sorted_clusters) > 2:
        cluster_labels[sorted_clusters[-1]] = "High Value Customers"

    numeric_df["Customer Segment"] = numeric_df["Cluster"].map(cluster_labels)

    st.dataframe(numeric_df.head())

    # SEGMENT DISTRIBUTION CHART
    st.subheader("Customer Segment Distribution")

    segment_counts = numeric_df["Customer Segment"].value_counts()

    st.bar_chart(segment_counts)

    # CLUSTER VISUALIZATION
    st.subheader("Cluster Visualization (First Two Features)")

    fig2, ax2 = plt.subplots()

    scatter = ax2.scatter(
        numeric_df.iloc[:, 0],
        numeric_df.iloc[:, 1],
        c=clusters
    )

    ax2.set_xlabel(numeric_cols[0])
    ax2.set_ylabel(numeric_cols[1])

    st.pyplot(fig2)

    # DOWNLOAD BUTTON
    csv = numeric_df.to_csv(index=False)

    st.download_button(
        label="Download Segmented Dataset",
        data=csv,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload dataset to begin segmentation.")