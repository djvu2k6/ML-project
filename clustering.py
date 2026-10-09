# ============================================================
# WINE CLUSTERING
# K-MEANS vs DBSCAN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("WineQT.csv")

print("\n========== DATASET INFORMATION ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. SELECT NUMERICAL FEATURES
# ============================================================

# Select only numerical columns
X = df.select_dtypes(include=[np.number]).copy()

# Remove quality column because it is the target/quality label
if "quality" in X.columns:
    X = X.drop(columns=["quality"])

# Remove Id if present because it is only an identifier
if "Id" in X.columns:
    X = X.drop(columns=["Id"])

print("\n========== FEATURES USED FOR CLUSTERING ==========")

print(X.columns.tolist())

print("\nFeature Shape:")
print(X.shape)


# ============================================================
# 3. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardization completed.")


# ============================================================
# 4. K-MEANS CLUSTERING
#    Test K = 2 to 10
# ============================================================

k_values = range(2, 11)

inertia_values = []
kmeans_silhouette = []

print("\n========== K-MEANS RESULTS ==========")

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    inertia_values.append(inertia)
    kmeans_silhouette.append(silhouette)

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.2f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 5. ELBOW CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")

plt.grid()

plt.show()


# ============================================================
# 6. K-MEANS SILHOUETTE CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    kmeans_silhouette,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Score")

plt.grid()

plt.show()


# ============================================================
# 7. SELECT BEST K BASED ON SILHOUETTE
# ============================================================

best_k = k_values[
    np.argmax(kmeans_silhouette)
]

print("\nBest K based on silhouette score:", best_k)


# ============================================================
# 8. FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering completed.")


# ============================================================
# 9. K-MEANS EVALUATION
# ============================================================

kmeans_silhouette_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

kmeans_db_score = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

kmeans_ch_score = calinski_harabasz_score(
    X_scaled,
    kmeans_labels
)

print("\n========== FINAL K-MEANS EVALUATION ==========")

print(
    "Silhouette Score:",
    round(kmeans_silhouette_score, 4)
)

print(
    "Davies-Bouldin Score:",
    round(kmeans_db_score, 4)
)

print(
    "Calinski-Harabasz Score:",
    round(kmeans_ch_score, 4)
)

print(
    "Inertia:",
    round(kmeans.inertia_, 4)
)


# ============================================================
# 10. DBSCAN PARAMETER SELECTION
#     Nearest Neighbour Distance Plot
# ============================================================

neighbors = NearestNeighbors(
    n_neighbors=5
)

neighbors.fit(X_scaled)

distances, indices = neighbors.kneighbors(
    X_scaled
)

# Take distance to 5th nearest neighbour
k_distances = np.sort(
    distances[:, 4]
)

plt.figure(figsize=(8, 5))

plt.plot(k_distances)

plt.xlabel("Data Points")
plt.ylabel("5th Nearest Neighbour Distance")
plt.title("DBSCAN Neighbour Distance Plot")

plt.grid()

plt.show()


# ============================================================
# 11. APPLY DBSCAN
# ============================================================

dbscan = DBSCAN(
    eps=2.0,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(
    X_scaled
)


# ============================================================
# 12. DBSCAN CLUSTER INFORMATION
# ============================================================

unique_labels = np.unique(
    dbscan_labels
)

print("\n========== DBSCAN RESULTS ==========")

print(
    "Cluster labels:",
    unique_labels
)

print(
    "Number of clusters:",
    len(
        unique_labels[
            unique_labels != -1
        ]
    )
)

noise_count = np.sum(
    dbscan_labels == -1
)

print(
    "Number of noise points:",
    noise_count
)


# ============================================================
# 13. DBSCAN EVALUATION
# ============================================================

# Remove noise points for internal validation
dbscan_mask = dbscan_labels != -1

dbscan_valid_labels = dbscan_labels[
    dbscan_mask
]

dbscan_valid_data = X_scaled[
    dbscan_mask
]


if (
    len(np.unique(dbscan_valid_labels)) >= 2
    and len(dbscan_valid_labels) > 1
):

    dbscan_silhouette = silhouette_score(
        dbscan_valid_data,
        dbscan_valid_labels
    )

    dbscan_db_score = davies_bouldin_score(
        dbscan_valid_data,
        dbscan_valid_labels
    )

    dbscan_ch_score = calinski_harabasz_score(
        dbscan_valid_data,
        dbscan_valid_labels
    )

    print("\n========== FINAL DBSCAN EVALUATION ==========")

    print(
        "Silhouette Score:",
        round(dbscan_silhouette, 4)
    )

    print(
        "Davies-Bouldin Score:",
        round(dbscan_db_score, 4)
    )

    print(
        "Calinski-Harabasz Score:",
        round(dbscan_ch_score, 4)
    )

else:

    print(
        "\nDBSCAN produced insufficient clusters "
        "for validation."
    )

    dbscan_silhouette = None
    dbscan_db_score = None
    dbscan_ch_score = None


# ============================================================
# 14. PCA FOR VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

print("\n========== PCA ==========")

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total variance explained:",
    round(
        sum(pca.explained_variance_ratio_),
        4
    )
)


# ============================================================
# 15. K-MEANS PCA VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=40
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Clustering - PCA Visualization"
)

plt.colorbar(
    label="Cluster"
)

plt.grid()

plt.show()


# ============================================================
# 16. DBSCAN PCA VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=dbscan_labels,
    cmap="viridis",
    s=40
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "DBSCAN Clustering - PCA Visualization"
)

plt.colorbar(
    label="Cluster (-1 = Noise)"
)

plt.grid()

plt.show()


# ============================================================
# 17. FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nK-MEANS")
print("--------------------")

print("Number of clusters:", best_k)

print(
    "Inertia:",
    round(kmeans.inertia_, 4)
)

print(
    "Silhouette:",
    round(kmeans_silhouette_score, 4)
)

print(
    "Davies-Bouldin:",
    round(kmeans_db_score, 4)
)

print(
    "Calinski-Harabasz:",
    round(kmeans_ch_score, 4)
)


print("\nDBSCAN")
print("--------------------")

print(
    "Number of clusters:",
    len(
        unique_labels[
            unique_labels != -1
        ]
    )
)

print(
    "Noise points:",
    noise_count
)

print(
    "Silhouette:",
    dbscan_silhouette
)

print(
    "Davies-Bouldin:",
    dbscan_db_score
)

print(
    "Calinski-Harabasz:",
    dbscan_ch_score
)


# ============================================================
# 18. ADD CLUSTER LABELS TO DATASET
# ============================================================

df["KMeans_Cluster"] = kmeans_labels

df["DBSCAN_Cluster"] = dbscan_labels

print("\n========== FINAL DATA ==========")

print(
    df[
        [
            "KMeans_Cluster",
            "DBSCAN_Cluster"
        ]
    ].head(20)
)