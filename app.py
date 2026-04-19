import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Page setup
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

    # Ensure minimum 2 numeric columns exist
    if len(numeric_cols) >= 2:

        # IMPORTANT FIX: Always select ONLY 2 features
        selected_features = numeric_cols[:2]

        X = df[selected_features].dropna()

        try:

            # Load trained model
            model = pickle.load(open("kmeans_model.pkl", "rb"))

            # Predict clusters
            clusters = model.predict(X)

            df = df.loc[X.index]
            df["Cluster"] = clusters

            st.success("Customer segmentation completed successfully!")

            # Cluster summary
            st.subheader("Cluster Summary")
            cluster_counts = df["Cluster"].value_counts().sort_index()
            st.write(cluster_counts)

            # Visualization
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

            # Download button
            st.subheader("Download Segmented Dataset")

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV file",
                data=csv,
                file_name="customer_segments_output.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Model prediction failed. Please check dataset format.")

    else:
        st.error("Dataset must contain at least two numeric columns.")

else:
    st.info("Please upload dataset to begin segmentation.")