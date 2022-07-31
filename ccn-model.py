# Common imports
import os
import imutils
from imutils import paths
import cv2
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# to make this notebook's output stable across runs
np.random.seed(42)

# Loading images
images = []
labels = []
image_folder_Path = ""
for image_file_path in imutils.paths.list_images(image_folder_Path):
    image_file = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(image_file,(760,1000))
    label = image_file_path.split('/')[9]
    images.append(resized_img)
    labels.append(label)

pdf_labels_dict = {
    'Q01_unreadable_text': 0,
    'Q02_unclear_text_breaky_sticky': 1,
    'Q03_semi_clear_text': 2,
    'Q04_clear_text': 3,
    'Q05_perfectly_clear_text': 4,
}
numbered_Labels = []
for label in labels:
    label = pdf_labels_dict[label]
    numbered_Labels.append(label)
labels =numbered_Labels

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Preprocessing: Scaling Images
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# Training Convolutional Neural Network
num_classes = 5
input_shape=(5, 1000, 760,1)
model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape[1:]),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit to training set             
model.fit(X_train_scaled, y_train, epochs=30) 

# Fit to test set
model.evaluate(X_test_scaled,y_test)

# Add Data Augmentation to the model
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(1000, 
                                                              760,
                                                              1)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# Train the model using data augmentation and a drop out layer
num_classes = 5
input_shape=(5, 1000, 760,1)

model2 = Sequential([
  data_augmentation,

  #Convolutional Network
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape[1:]),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Conv2D(16, 3, padding='same', activation='relu',),
  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  #Dense network
  layers.Flatten(),
  layers.Dense(5000, activation='sigmoid'),
  layers.Dense(500, activation='sigmoid'),
  layers.Dense(50, activation='sigmoid'),
  layers.Dense(num_classes,activation='softmax' )
])

model2.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit Model to training set             
model2.fit(X_train_scaled, y_train, epochs=500) 

# Fit Model to test set
model2.evaluate(X_test_scaled,y_test)

# Building a confusion matrix
y_pred = model2.predict(X_test_scaled)

