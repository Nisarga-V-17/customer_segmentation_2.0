import pandas as pd
from sklearn.cluster import KMeans

# Sample mall data
data = {
    "Income": [15, 16, 17, 40, 42, 43, 70, 72, 74, 90, 92, 95],
    "Spending": [39, 40, 42, 60, 61, 65, 20, 21, 19, 80, 82, 85]
}

df = pd.DataFrame(data)

# KMeans model
kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

df["Cluster"] = kmeans.labels_

print(df)