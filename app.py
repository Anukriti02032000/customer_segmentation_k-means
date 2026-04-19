import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

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
    st.success(f"{df.shape}")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Selected Features for Clustering")
    st.write(numeric_cols)

    if len(numeric_cols) >= 2:

        # Select first 2 numeric features
        X = df[numeric_cols[:2]].dropna()

        try:
            # Correct path for deployment
            MODEL_PATH = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "kmeans_model.pkl"
            )

            model = pickle.load(open(MODEL_PATH, "rb"))

            # Predict clusters
            clusters = model.predict(X)

            df = df.loc[X.index]
            df["Cluster"] = clusters

            st.success("Customer segmentation completed successfully!")

            # Cluster summary
            st.subheader("Cluster Summary")

            cluster_counts = df["Cluster"].value_counts().sort_index()

            st.write(cluster_counts)

            # Cluster visualization
            st.subheader("Cluster Visualization")

            fig, ax = plt.subplots()

            scatter = ax.scatter(
                X.iloc[:, 0],
                X.iloc[:, 1],
                c=clusters,
                cmap="viridis"
            )

            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])

            st.pyplot(fig)

            # Customers per cluster chart
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

        except FileNotFoundError:
            st.error(
                "Model file not found. Please ensure kmeans_model.pkl exists inside models folder."
            )

    else:
        st.error(
            "Dataset must contain at least two numeric columns for clustering."
        )

else:
    st.info("Please upload dataset to begin segmentation.")