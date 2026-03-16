import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import io

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("🛍️ Mall Customer Segmentation using KMeans Clustering")

st.markdown("Upload a customer dataset to visualize clusters, discover spending patterns, and download segmented data.")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    features = ['Age', 'AnnualIncome (k$)', 'SpendingScore (1-100)', 
                'TotalSpending', 'RecencyDays', 'PurchaseFrequency', 'TimeSpent']

    missing = [col for col in features if col not in df.columns]
    if missing:
        st.error(f"Required columns missing: {missing}")
        st.stop()

    X = df[features]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal number of clusters
    wcss = []
    silhouette_scores = []
    max_clusters = st.sidebar.slider("Max Clusters for Elbow/Silhouette", 3, 15, 10)

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    st.subheader("📊 Elbow & Silhouette Method")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(range(2, max_clusters + 1), wcss, 'bo-')
    ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("WCSS")

    ax[1].plot(range(2, max_clusters + 1), silhouette_scores, 'go-')
    ax[1].set_title("Silhouette Scores")
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Score")

    st.pyplot(fig)

    # Choose number of clusters
    optimal_clusters = st.sidebar.slider("🔢 Select Number of Clusters (k)", 2, max_clusters, 5)

    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Cluster Summary
    st.subheader("📈 Cluster Summary")
    st.dataframe(df.groupby('Cluster')[features].mean().round(2))

    # Visualizations
    st.subheader("📌 Cluster Visualizations")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.scatterplot(data=df, x='AnnualIncome (k$)', y='SpendingScore (1-100)', hue='Cluster', palette='viridis', ax=axes[0, 0])
    axes[0, 0].set_title("Income vs Spending Score")

    sns.scatterplot(data=df, x='Age', y='TotalSpending', hue='Cluster', palette='viridis', ax=axes[0, 1])
    axes[0, 1].set_title("Age vs Total Spending")

    sns.scatterplot(data=df, x='RecencyDays', y='PurchaseFrequency', hue='Cluster', palette='viridis', ax=axes[1, 0])
    axes[1, 0].set_title("Recency vs Purchase Frequency")

    sns.countplot(x='Cluster', data=df, palette='viridis', ax=axes[1, 1])
    axes[1, 1].set_title("Customers per Cluster")

    plt.tight_layout()
    st.pyplot(fig)

    # Download segmented CSV
    st.subheader("⬇️ Download Clustered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "customer_segments.csv", "text/csv")

    st.success("Clustering completed successfully! 🎉")

else:
    st.info("Please upload a `.csv` file to begin.")
