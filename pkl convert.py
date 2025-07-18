import os
from skimage.io import imread
from skimage.transform import resize
import pickle
from transformers import AutoImageProcessor, AutoModel

model_ckpt = "MichaelMM2000/vit-base-animals10"
extractor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)
model = AutoModel.from_pretrained(model_ckpt)

def extract_embeddings(image):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()

data_path = fr'./animals/Dataset Of animal Images/'
data = dict()
data["label"] = []
data["file"] = []
data["embed"] = []

for subdir in os.listdir(data_path):
    new_data_path = data_path + subdir + "/train/images"
    for file in os.listdir(new_data_path):
        print(file)
        im = imread(os.path.join(new_data_path, file))
        data["embed"].append(extract_embeddings(im))
        im = resize(im, (64, 64))
        data["file"].append(im)
        match subdir:
            case "Cat":
                data["label"].append(0)
            case "Cow":
                data["label"].append(1)
            case "Deer":
                data["label"].append(2)
            case "Dog":
                data["label"].append(3)
            case "Goat":
                data["label"].append(4)
            case "Hen":
                data["label"].append(5)
            case "Rabbit":
                data["label"].append(6)
            case "Sheep":
                data["label"].append(7)
with open("data.pkl", "wb") as fp:
    pickle.dump(data, fp)




