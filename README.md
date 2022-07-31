# PDF-Noise-Classification

> To decrease the time taken by a manual pre-OCR pdf labeling process. I built a CNN image classification model using OpenCV, Tensorflow and Keras in Python using Jyputer Notebooks. The CNN model classifies PDFs according to the amount of noise, 1 - representing very high noise and 5 - representing clear pdfs.

## loading PDFS
``` {.python}
images = []
labels = []
image_folder_Path =""
for image_file_path in imutils.paths.list_images(image_folder_Path):
    image_file = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(image_file,(760,1000))
    label = image_file_path.split('/')[9]
    images.append(resized_img)
    labels.append(label)
```

## Fixing PDF Labels
``` {.python}
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
``` 

## PDF examples
![alt text](https://github.com/Kovenda/CNN-PDF-Noise-Classification/blob/main/pdf1.png?raw=true)

# Train Test Split
``` {.python}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
``` 

# Model Building
``` {.python}
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
``` 

## Fit Training Set
``` {.python}
model.fit(X_train_scaled, y_train, epochs=30) 
``` 
|Epoch|Accuracy|
| 1 | 0.2634 |
| 7 | 0.7737 |
| 11 | 0.9918 |
| 15 | 1.0000 |
| 30 | 1.0000 |

## Fit Test Set
``` {.python}
model.evaluate(X_test_scaled,y_test)
``` 
**accuracy:** 0.3537

## Add Data Augmentation
``` {.python}
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
``` 

## Rebuild Model with Data Augmentation

``` {.python}
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
``` 

## ReFit Training Set
``` {.python}
model2.fit(X_train_scaled, y_train, epochs=500) 
``` 
|Epoch|Accuracy|
| 1 | 0.1152 |
| 2 | 0.2510|
| 3 | 0.3292 |
| 300 | 0.3292 |
| 500 | 0.3292 |

## Refit Test Set
``` {.python}
model2.evaluate(X_test_scaled,y_test)
``` 
**accuracy:** 0.4268



