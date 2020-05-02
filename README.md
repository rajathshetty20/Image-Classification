# Image Classification

Binary classification of images of cats and dogs, using a convolutional neural network.

Dataset is from : https://www.microsoft.com/en-us/download/details.aspx?id=54765.
Download and extract it.

#### Data Extraction
run `python data_extraction.py` with the 'kagglecatsanddogs_3367a' folder in the present working directory. This converts the images into square matrices with element values in range (0,1), representing the intensity of pixels in the rgb channels. The shape of each image matrix is (IMG_SIZE, IMG_SIZE, 3). Then the data is shuffled and split into train_data and test_data, and saved in the present working directory as .npy files.

#### Training and testing the model
`cnn.ipynb` Obtained about 96% accuracy on an untrained dataset, using a convolutional neural network.
