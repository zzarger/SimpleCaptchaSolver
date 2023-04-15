import cv2
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from utility import resize
import pickle

IMG_FOLDER = 'finessed_letters'
MODEL_FILE = 'captcha_model.hdf5'
MODEL_LABELS_FILE = 'model_labels.dat'

data = []
labels = []

for file in paths.list_images(IMG_FOLDER):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = resize(image, 20, 20)
    image = np.expand_dims(image, axis=2)

    label = file.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(xtrain, xtest, ytrain, ytest) = train_test_split(data, labels, test_size=0.25, random_state=0)

lb = LabelBinarizer().fit(ytrain)
ytrain = lb.transform(ytrain)
ytest = lb.transform(ytest)

with open(MODEL_LABELS_FILE, "wb") as f:
    pickle.dump(lb, f)

model = Sequential()

model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500, activation="relu"))

model.add(Dense(32, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=32, epochs=10, verbose=1)

model.save(MODEL_FILE)
