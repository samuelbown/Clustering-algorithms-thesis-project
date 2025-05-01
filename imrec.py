
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



data = pd.read_pickle('./data.pkl')
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


'''pca = PCA(n_components=2) 
x_trained_pca = pca.fit_transform(x_trained) 
x_tested_pca = pca.transform(x_tested)'''
#previously used PCA dim reduction, might use again so keeping it around

tsne = TSNE(n_components=2, random_state=11)
x_trained_ts = tsne.fit_transform(x_trained)

clf = DBSCAN(eps=3, min_samples=8) #these parameters need to be tweaked
y_pred = clf.fit_predict(x_trained_ts)
accuracy_score(y_pred, y_train)
print(classification_report(y_pred,y_train))

scatter = plt.scatter(x_trained_ts[:, 0], x_trained_ts[:, 1], c=y_pred, cmap='viridis', marker='.')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
