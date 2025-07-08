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
import sys

from kmeans import KMeans as kme

import os
import time
from DBKMEANS import KMeans as dbk
results = []
results2 = []
def dataprep(dim=2):

    data = pd.read_pickle('./datae1.pkl')
    x_train, x_test, y_train, y_test = train_test_split(data["embed"], data["label"], test_size=0.2, shuffle=True)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train2=x_train
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train[:8000]
    y_train = y_train[:8000] #these splits arent actually necessary with DBSCAN but will keep around for now as well
    x_test = x_test[:2000]
    y_test = y_test[:2000]


        # File to store preprocessed data
    reduced_file = "pca_tsne_outputs.npz"

    to_save = False


            # Step 1: PCA
    pca = PCA(n_components=50)
    x_pca = pca.fit_transform(x_train)

            # Step 2: t-SNE
    tsne = TSNE(n_components=dim)
    return tsne.fit_transform(x_pca),y_train




def k_test(iter=50,dim = 2):
    i=0
    while i <iter:
        i+=1
        kmeans =  kme(n_clusters=2,init='random' )
        x_trained_ts,y_train = dataprep(dim)
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

        results2.append({
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
    results_df2 = pd.DataFrame(results2)
    results_df2.to_csv("kmeans_k" + str(dim) + ".csv", index=False)


def k_alg_test(iter=50,dim=2):
    i=0
    while i<iter:
        i+=1
        kmeans = kme(n_clusters=8, alg='elkan')
        x_trained_ts, y_train = dataprep(dim)
        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time1 = time.time() - start_time

        ari1 = adjusted_rand_score(y_train, y_pred)
        score1 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=8, alg='lloyd')

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time2 = time.time() - start_time

        ari2 = adjusted_rand_score(y_train, y_pred)
        score2 = silhouette_score(x_trained_ts, y_pred)

        results2.append({
            "a1": "elkan",
            "runtime_sec": elapsed_time1,
            "adjusted_rand_index": ari1,
            "silhouette_score": score1,

            "a2": "lloyd",
            "runtime_sec2": elapsed_time2,
            "adjusted_rand_index2": ari2,
            "silhouette_score2": score2,
        })
        results_df2 = pd.DataFrame(results2)
        results_df2.to_csv("kmeans_alg"+str(dim)+".csv", index=False)

def k_init_test(iter=50,dim=2,maxit=500):
    i=0
    while i<iter:
        i+=1
        kmeans = kme(n_clusters=8, alg='lloyd', init='random',max_iter=maxit)
        x_trained_ts, y_train = dataprep(dim)
        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time1 = time.time() - start_time

        ari1 = adjusted_rand_score(y_train, y_pred)
        score1 = silhouette_score(x_trained_ts, y_pred)

        kmeans = kme(n_clusters=8, alg='lloyd', init='kmeans++',max_iter=maxit)

        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time2 = time.time() - start_time

        ari2 = adjusted_rand_score(y_train, y_pred)
        score2 = silhouette_score(x_trained_ts, y_pred)

        results2.append({
            "a1": "elkan",
            "runtime_sec": elapsed_time1,
            "adjusted_rand_index": ari1,
            "silhouette_score": score1,

            "a2": "lloyd",
            "runtime_sec2": elapsed_time2,
            "adjusted_rand_index2": ari2,
            "silhouette_score2": score2,
        })
        results_df2 = pd.DataFrame(results2)
        results_df2.to_csv("kmeans_init"+str(dim)+"_maxit"+str(maxit)+".csv", index=False)

def dbk_test(iter=50,dim=2):
    i=0
    while i<iter:
        i+=1
        x_trained_ts, y_train = dataprep(dim=2)
        kmeans = dbk(n_clusters=8, init='random', alg='elkan')
        start_time = time.time()
        y_pred = kmeans.fit_predict(x_trained_ts)
        elapsed_time = time.time() - start_time
        a=np.array(y_pred)
        np.set_printoptions(threshold=sys.maxsize)
        c=np.array(y_pred)
        print(a)
        print(y_train)
        ari = adjusted_rand_score(y_train, y_pred)
        score = silhouette_score(x_trained_ts, y_pred)

        results2.append({
                "k": 2,
                "runtime_sec": elapsed_time,
                "adjusted_rand_index": ari,
                "silhouette_score":score,
                "n":"",
            })
    results_df2 = pd.DataFrame(results2)
    results_df2.to_csv("dbkmeans_alg" + str(dim) + ".csv", index=False)

##k_test()
k_test(iter=50)
#k_alg_test(iter=3)
#k_alg_test(iter=50,dim=3)
#k_init_test(iter=50)
#k_init_test(iter=50,maxit=1)

