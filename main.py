import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# === Load processed dataset if available, otherwise preprocess ===
base_dir = os.path.dirname(__file__)
processed_path = os.path.join(base_dir, 'processed_dataset.csv')
if os.path.exists(processed_path):
    # Load preprocessed numeric CSV directly
    df_processed = pd.read_csv(processed_path)

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
# Use a non-interactive backend to avoid GUI-related errors (e.g. gi.require_version)
# This is safe for scripts that only save or show plots in non-GUI environments.
matplotlib.use('Agg')
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