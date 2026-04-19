import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation using K-Means Clustering",
    layout="wide"
)

# Title
st.title("Customer Segmentation using K-Means Clustering")

st.write(
    "Upload a customer dataset to generate behavior-based segmentation insights using the K-Means clustering algorithm."
)

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset shape
    st.subheader("Dataset Shape")
    st.success(df.shape)

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Selected Features for Clustering")
    st.write(numeric_cols)

    if len(numeric_cols) >= 2:

        # Use first two numeric columns
        selected_features = numeric_cols[:2]

        X = df[selected_features].dropna()

        # Standardization (important for clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df = df.loc[X.index]
        df["Cluster"] = clusters

        st.success("Customer segmentation completed successfully!")

        # Cluster summary
        st.subheader("Cluster Summary")

        cluster_counts = df["Cluster"].value_counts().sort_index()
        st.write(cluster_counts)

        # Scatter plot
        st.subheader("Cluster Visualization")

        fig, ax = plt.subplots()

        ax.scatter(
            X[selected_features[0]],
            X[selected_features[1]],
            c=clusters,
            cmap="viridis"
        )

        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])

        st.pyplot(fig)

        # Bar chart
        st.subheader("Customers per Cluster")

        st.bar_chart(cluster_counts)

        # Download segmented dataset
        st.subheader("Download Segmented Dataset")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV file",
            data=csv,
            file_name="customer_segments_output.csv",
            mime="text/csv"
        )

    else:
        st.error("Dataset must contain at least two numeric columns.")

else:
    st.info("Please upload dataset to begin segmentation.")