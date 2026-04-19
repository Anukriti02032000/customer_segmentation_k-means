import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Title
st.title("AI-Based Customer Segmentation Analytics Dashboard")

st.write(
    "Upload a customer transaction dataset to generate behavior-based segmentation insights using K-Means clustering."
)

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Selected Features for Clustering")
    st.write(numeric_cols)

    if len(numeric_cols) >= 2:

        # Use first two numeric columns
        X = df[numeric_cols[:2]]

        # Remove missing values
        X = X.dropna()

        # Load trained model
        model = pickle.load(open("models/kmeans_model.pkl", "rb"))

        # Predict clusters
        clusters = model.predict(X)

        df["Cluster"] = clusters

        st.success("Customer segmentation completed successfully!")

        # Cluster summary
        st.subheader("Cluster Summary")

        cluster_counts = df["Cluster"].value_counts().sort_index()
        st.write(cluster_counts)

        # Plot cluster visualization
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

        # Cluster count chart
        st.subheader("Customers per Cluster")

        st.bar_chart(cluster_counts)

        # Download button
        st.subheader("Download Segmented Dataset")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV file",
            data=csv,
            file_name="customer_segments_output.csv",
            mime="text/csv",
        )

    else:
        st.error("Dataset must contain at least two numeric columns for clustering.")

else:
    st.info("Please upload dataset to begin segmentation.")