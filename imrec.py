
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from sklearn.decomposition import PCA


data = pd.read_pickle('./data.pkl')
x_train, x_test, y_train, y_test = train_test_split(data["file"], data["label"], test_size=0.2, shuffle=True, random_state=11)


x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train[:8000]
y_train = y_train[:8000]
x_test = x_test[:2000]
y_test = y_test[:2000]

nsamples, nx, ny, nrgb = x_train.shape
x_trained = x_train.reshape((nsamples, nx * ny * nrgb))

nsamples, nx, ny, nrgb = x_test.shape
x_tested = x_test.reshape((nsamples, nx * ny * nrgb))

pca = PCA(n_components=100, random_state=11) 
x_trained_pca = pca.fit_transform(x_trained)
x_tested_pca = pca.transform(x_tested)


clf = svm.SVC(kernel='linear')
clf.fit(x_trained_pca, y_train)

y_pred = clf.predict(x_tested_pca)

accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))
