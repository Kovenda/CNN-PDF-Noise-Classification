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
![alt text](https://github.com/Kovenda/CNN-PDF-Noise-Classification/blob/main/images-and-plots/pdf1.png?raw=true)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Model Building
``` {.python}
``` 



