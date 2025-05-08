from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from skimage import metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.cluster import KMeans
import umap
import numpy as np
from sklearn.metrics import silhouette_score


data = pd.read_pickle('./data2.pkl')
x_train, x_test, y_train, y_test = train_test_split(data["file"], data["label"], test_size=0.2, shuffle=True, random_state=11)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train[:8000]
y_train = y_train[:8000] #these splits arent actually necessary with DBSCAN but will keep around for now as well
x_test = x_test[:2000]
y_test = y_test[:2000]

nsamples, nx, ny, nrgb = x_train.shape
x_trained = x_train.reshape((nsamples, nx * ny * nrgb))
nsamples, nx, ny, nrgb = x_test.shape
x_tested = x_test.reshape((nsamples, nx * ny * nrgb))


pca = PCA(n_components=50)
x_trained_ts = pca.fit_transform(x_trained)
x_tested_ts = pca.transform(x_tested)
#previously used PCA dim reduction, might use again so keeping it around

tsne = TSNE(n_components=2, random_state=1)
x_trained_ts = tsne.fit_transform(x_trained_ts)

#umap_model = umap.UMAP(n_components=2, random_state=42)
#x_trained_ts = umap_model.fit_transform(x_trained_ts)

k = 3  # You have 3 classes: cats, dogs, snakes
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(x_trained_ts)

print("Adjusted Rand Index (ARI):", adjusted_rand_score(y_train, y_pred))
print("Normalized Mutual Info (NMI):", normalized_mutual_info_score(y_train, y_pred))
print("Adjusted Mutual Info (AMI):", adjusted_mutual_info_score(y_train, y_pred))
print("Homogeneity:", homogeneity_score(y_train, y_pred))
print("Completeness:", completeness_score(y_train, y_pred))
print("V-measure:", v_measure_score(y_train, y_pred))

score = silhouette_score(x_trained_ts, y_pred)

print("Silhouette Score:", score)
# Separate noise from clusters
is_noise = y_pred == -1
not_noise = ~is_noise

plt.figure(figsize=(10, 6))

# Plot clustered points
scatter = plt.scatter(x_trained_ts[not_noise, 0], x_trained_ts[not_noise, 1],
                      c=y_pred[not_noise], cmap='viridis', marker='.')

# Plot noise points in red
plt.scatter(x_trained_ts[is_noise, 0], x_trained_ts[is_noise, 1],
            color='red', marker='.', label='Noise')

plt.colorbar(scatter, label='Cluster label')
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

'''scatter = plt.scatter(x_trained_ts[:, 0], x_trained_ts[:, 1], c=y_pred, cmap='viridis', marker='.')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()'''