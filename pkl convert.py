import os
from skimage.io import imread
from skimage.transform import resize
import pickle

data_path = fr'./animals/Dataset Of animal Images/'
data = dict()
data["label"] = []
data["file"] = []

for subdir in os.listdir(data_path):
    new_data_path = data_path + subdir + "/train/images"
    for file in os.listdir(new_data_path):
        print(file)
        im = imread(os.path.join(new_data_path, file))
        im = resize(im, (64, 64))
        data["file"].append(im)
        data["label"].append(subdir)

with open("data.pkl", "wb") as fp:
    pickle.dump(data, fp)




