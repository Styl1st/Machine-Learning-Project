import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# =============================================
# 1. LOAD DATA
# =============================================
df = pd.read_csv("combined_survey_data.csv")

print("\n===== First rows of the dataset =====")
print(df.head())

print("\n===== Dataset Info =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Summary Statistics (numerical only) =====")
print(df.describe())

# =============================================
# 2. PREPROCESSING
# =============================================

# Drop rows with missing 'Value' because it is numeric and essential
df = df.dropna(subset=["Value"])

# Drop rows with missing 'Value' because it is numeric and essential
# CLEAN THE "Value" COLUMN
df["Value"] = df["Value"].replace("***", pd.NA)
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

print("\n===== Summary Statistics for 'Value' =====")

mean_value = df["Value"].mean()
median_value = df["Value"].median()
std_value = df["Value"].std()

print(f"Mean: {mean_value:.2f}")
print(f"Median: {median_value:.2f}")
print(f"Standard Deviation: {std_value:.2f}")

# Identify categorical & numerical columns
categorical_cols = ["Group", "Education Level", "Year", "Domain", "Indicator"]
numeric_cols = ["Value"]

# One-hot encoding for categoricals
ohe = OneHotEncoder(handle_unknown="ignore")

# Standard scaler for numeric values
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, categorical_cols),
        ("num", scaler, numeric_cols)
    ]
)

df_processed = preprocessor.fit_transform(df)

print("\n===== Shape after preprocessing =====")
print(df_processed.shape)

# =============================================
# Save processed dataset (optional)
# =============================================
import numpy as np
df_processed = pd.DataFrame(
    df_processed.toarray() if hasattr(df_processed, "toarray") else df_processed
)
df_processed.to_csv("processed_dataset.csv", index=False)

print("\nProcessed dataset saved as 'processed_dataset.csv'")

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================
# K-MEANS CLUSTERING
# =============================================

print("\n===== K-Means Clustering =====")

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(df_processed)

kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(df_processed, kmeans_labels)

print("Inertia:", kmeans_inertia)
print("Silhouette Score:", kmeans_silhouette)

pca = PCA(n_components=2)
reduced = pca.fit_transform(df_processed)

# =============================================
# Agglomerative Clustering
# =============================================

print("\n===== Agglomerative Clustering =====")

agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(df_processed)

agg_silhouette = silhouette_score(df_processed, agg_labels)

print("Silhouette Score:", agg_silhouette)

# =============================================
# DBSCAN Clustering
# =============================================

print("\n===== DBSCAN Clustering =====")

dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(df_processed)

# DBSCAN often produces -1 (noise), so check if enough clusters exist
unique_labels = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

print("Number of clusters found:", unique_labels)

if unique_labels > 1:
    dbscan_silhouette = silhouette_score(df_processed, dbscan_labels)
    print("Silhouette Score:", dbscan_silhouette)
else:
    print("Silhouette Score: Not applicable (DBSCAN found too few clusters)")

# =============================================
# Generate a clean result table
# ===========================================

import pandas as pd

# Build results table
results_table = pd.DataFrame({
    "Model": ["K-Means", "Agglomerative", "DBSCAN"],
    "Silhouette Score": [
        kmeans_silhouette,
        agg_silhouette,
        "N/A" if unique_labels <= 1 else dbscan_silhouette
    ],
    "Inertia (Only for K-Means)": [
        kmeans_inertia,
        "N/A",
        "N/A"
    ],
    "Clusters Found": [
        len(set(kmeans_labels)),
        len(set(agg_labels)),
        unique_labels
    ]
})

print("\n===== Combined Results Table =====")
print(results_table)

# =============================================
# ALL GRAPHS SECTION — Display at the end
# =============================================

print("\nShowing all cluster visualizations...\n")

# K-Means plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=reduced[:,0], y=reduced[:,1],
    hue=kmeans_labels, palette="tab10"
)
plt.title("K-Means Clusters (PCA Projection)")
plt.show()

# Agglomerative plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=reduced[:,0], y=reduced[:,1],
    hue=agg_labels, palette="tab10"
)
plt.title("Agglomerative Clusters (PCA Projection)")
plt.show()

# DBSCAN plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=reduced[:,0], y=reduced[:,1],
    hue=dbscan_labels, palette="tab10"
)
plt.title("DBSCAN Clusters (PCA Projection)")
plt.show()