
from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from DBSCAN import DBSCAN
import time

data = pd.read_pickle('./data1.pkl')
x_train, x_test, y_train, y_test = train_test_split(data["embed"], data["label"], test_size=0.2, shuffle=True)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train[:8000]
y_train = y_train[:8000]
x_test = x_test[:2000]
y_test = y_test[:2000]

pca = PCA(n_components=50, random_state=11)

x_trained_pca = pca.fit_transform(x_train)


tsne = TSNE(n_components=2, random_state=11)
x_trained_ts = tsne.fit_transform(x_trained_pca)


start = time.perf_counter()
dbscan = DBSCAN()
clusters, noise, labels = dbscan.fit(data_points=x_trained_ts, eps=9.5, minpts=150)

end = time.perf_counter()
print(f"DBSCAN ran in {end - start:0.4f} seconds")

print(len(noise),"points of noise")
print(len(clusters), "clusters")
score = silhouette_score(x_trained_ts, labels)
score1 = adjusted_rand_score(y_train, labels)
print("Silhouette Score:", score) 
print("Adjusted Rand Score:", score1)

