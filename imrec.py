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

from kmeans import KMeans as kme

import os
import time
from DBKMEANS import KMeans as dbk


data = pd.read_pickle('./datae1.pkl')
x_train, x_test, y_train, y_test = train_test_split(data["embed"], data["label"], test_size=0.2, shuffle=True, random_state=11)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train[:8000]
y_train = y_train[:8000] #these splits arent actually necessary with DBSCAN but will keep around for now as well
x_test = x_test[:2000]
y_test = y_test[:2000]


# File to store preprocessed data
reduced_file = "pca_tsne_outputs.npz"

to_save = False

if os.path.exists(reduced_file) and to_save is True:
    print("Loading cached PCA and t-SNE outputs")
    data_reduced = np.load(reduced_file)
    x_trained_ts = data_reduced["x_tsne"]
    # Optional: x_pca = data_reduced["x_pca"] if you need it later
else:

    # Step 1: PCA
    pca = PCA(n_components=50)
    x_pca = pca.fit_transform(x_train)

    # Step 2: t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    x_trained_ts = tsne.fit_transform(x_pca)

    # Save results
    np.savez(reduced_file, x_pca=x_pca, x_tsne=x_trained_ts)
    print(f"Saved PCA and t-SNE outputs to {reduced_file}")

#umap_model = umap.UMAP(n_components=2, random_state=42)
#x_trained_ts = umap_model.fit_transform(x_trained_ts)

#kmeans = KMeans(n_clusters=8, algorithm='elkan')
kmeans = kme(n_clusters=8,init='kmeans++', alg='elkan')
#kmeans = dbk(n_clusters=8,init='random', alg='elkan')
#kmeans = kma(k=8, method='classic')
#kmeans = dbs()

'''
start_time = time.time()
y_pred = kmeans.fit_predict(x_trained_ts)
end_time = time.time()
elapsed_time = end_time - start_time


print(f"KMeans clustering took {elapsed_time:.4f} seconds")

print("Adjusted Rand Index (ARI):", adjusted_rand_score(y_train, y_pred))

score = silhouette_score(x_trained_ts, y_pred)
print("Silhouette Score:", score)'''

results = []
def k_test():
    i=0
    while i <1:
        i+=1
        kmeans =  kme(n_clusters=2,init='random' )

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time = time.time() - start_time

        ari = adjusted_rand_score(y_train, y_pred)
        score = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=4,init='random')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time2 = time.time() - start_time

        ari2 = adjusted_rand_score(y_train, y_pred)
        score2 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=8,init='random')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time3 = time.time() - start_time

        ari3 = adjusted_rand_score(y_train, y_pred)
        score3 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=16,init='random')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time4 = time.time() - start_time
        ari4 = adjusted_rand_score(y_train, y_pred)
        score4 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=32,init='random')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time5 = time.time() - start_time
        ari5 = adjusted_rand_score(y_train, y_pred)
        score5 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=64,init='random')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time6 = time.time() - start_time
        ari6 = adjusted_rand_score(y_train, y_pred)
        score6 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=128,init='random')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time7 = time.time() - start_time
        ari7 = adjusted_rand_score(y_train, y_pred)
        score7 = silhouette_score(x_trained_ts, y_pred)

        results.append({
            "k": 2,
            "runtime_sec": elapsed_time,
            "adjusted_rand_index": ari,
            "silhouette_score":score,
            "n":"",

            "runtime_sec2": elapsed_time2,
            "adjusted_rand_index2": ari2,
            "silhouette_score2": score2,
            "n2": "",

            "runtime_sec3": elapsed_time3,
            "adjusted_rand_index3": ari3,
            "silhouette_score3": score3,
            "n3": "",
            "runtime_sec4": elapsed_time4,
            "adjusted_rand_index4": ari4,
            "silhouette_score4": score4,
            "n4": "",
            "runtime_sec5": elapsed_time5,
            "adjusted_rand_index5": ari5,
            "silhouette_score5": score5,
            "n5": "",
            "runtime_sec6": elapsed_time6,
            "adjusted_rand_index6": ari6,
            "silhouette_score6": score6,
            "n6": "",
            "runtime_sec7": elapsed_time7,
            "adjusted_rand_index7": ari7,
            "silhouette_score7": score7,
        })

    results.append({
            "k": "",
            "runtime_sec": "",
            "adjusted_rand_index": "",
            "silhouette_score": ""
        })
def dbk_test():
    kmeans = dbk(n_clusters=8, init='random', alg='elkan')
    start_time = time.time()
    y_pred = kmeans.fit_predict(x_trained_ts)
    elapsed_time = time.time() - start_time
    ari = adjusted_rand_score(y_train, y_pred)
    score = silhouette_score(x_trained_ts, y_pred)

    results.append({
            "k": 2,
            "runtime_sec": elapsed_time,
            "adjusted_rand_index": ari,
            "silhouette_score":score,
            "n":"",
        })

dbk_test()
kmeans = dbk(n_clusters=8,init='random', alg='elkan')
# === Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("kmeans_test_results_k2.csv", index=False)
print("Results saved to kmeans_test_results.csv")

