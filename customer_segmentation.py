import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_excel("mall_customers.xlsx")

# Clean column names (just in case)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# Use last 2 columns
X = df.iloc[:, -2:]

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print(df.head())
