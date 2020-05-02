import os
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 64
CAT_PATH = "./kagglecatsanddogs_3367a/PetImages/Cat"
DOG_PATH = "./kagglecatsanddogs_3367a/PetImages/Dog"
LABELS = {CAT_PATH:0, DOG_PATH:1}
TEST_RATIO = 0.2
data = []
    
for label in LABELS:
    for f in tqdm(os.listdir(label)):
        try:
            path = os.path.join(label,f)
            #reads the image
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            #Scale down the pixel values by a factor of 255 i.e the max value
            data.append([img/255, LABELS[label]])
        except Exception:
            pass

#shuffle the data and split it into trian_data and test_data         
np.random.shuffle(data)
test_size = int(len(data)*TEST_RATIO)
train_data = data[:-test_size]
test_data = data[-test_size:]

#check that the ratio of cats and dogs in train_data is not very skewed
catcount = 0
dogcount = 0
for i in train_data:
    if(i[1] == LABELS[CAT_PATH]):
        catcount += 1
    elif(i[1] == LABELS[DOG_PATH]):
        dogcount += 1
print("Ratio of cats and dogs in train set: ",catcount/dogcount)

#saving the data
np.save("train_data.npy", train_data)
np.save("test_data.npy", test_data)
print("completed")
