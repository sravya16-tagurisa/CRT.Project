# Project 1: Healthcare Disease Prediction Using Machine Learning

# 1. Project Content
Develop a machine learning model to predict possible diseases based on patient health-related data.

# 2. Project Code

~~~python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_csv("healthcare_dataset.csv")
df.drop(columns=['Name'], inplace=True)
label_encoders = {col: LabelEncoder().fit(df[col]) for col in df.select_dtypes(include=['object']).columns}
for col, le in label_encoders.items():
    df[col] = le.transform(df[col])

X = df.drop("Disease", axis=1)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

with open('disease_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
~~~

# 3. Key Technologies
- Python
- Pandas, NumPy (Data manipulation)
- Scikit-learn (Logistic Regression, Label Encoding, Evaluation)
- Pickle (Model saving)

# 4. Description
This project aims to simplify early diagnosis in healthcare. It uses logistic regression, a classification algorithm, to predict diseases based on features like symptoms, age, and gender. The data is preprocessed to remove non-numerical or irrelevant fields, and encoded for training the model. The model is then evaluated and saved for deployment.

# 5. Output
![image](https://github.com/user-attachments/assets/5903acd9-9279-4d64-b98c-007d139f981c)


# 6. Further Research
- Expand dataset to include more symptoms and rare diseases.
- Try ensemble models like Random Forest or XGBoost.
- Build a web interface using Flask or Django.
- Integrate real-time patient data using IoT or EHRs.

# Project 2: IMDB Sentiment Analysis Using Deep Learning

# 1. Project Content
Build a deep learning model to classify IMDB movie reviews as positive or negative based on textual content.

# 2. Project Code

~~~python
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('stopwords')
movie_reviews = pd.read_csv("/content/IMDB Dataset.csv")

# Clean reviews
def preprocess(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text().lower()
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(tokens)

movie_reviews['cleaned'] = movie_reviews['review'].apply(preprocess)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(movie_reviews['cleaned'])
sequences = tokenizer.texts_to_sequences(movie_reviews['cleaned'])
X = pad_sequences(sequences, maxlen=200)
y = pd.get_dummies(movie_reviews['sentiment']).values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LSTM Model
model = Sequential([
    Embedding(5000, 64, input_length=200),
    LSTM(64),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
~~~

# 3. Key Technologies
- Python
- NLTK, BeautifulSoup (Text preprocessing)
- Keras/TensorFlow (Neural Networks: Embedding, LSTM)
- Pandas (Data handling)
- Seaborn/Matplotlib (for EDA - if included)

# 4. Description
This project takes raw movie reviews, cleans them using HTML parsing and stopword removal, and converts them into padded sequences. An LSTM-based neural network is trained to detect sentiment, providing a powerful text classification pipeline. The model learns to associate patterns in text with positive or negative sentiments.

# 5. Output
![image](https://github.com/user-attachments/assets/757fb4c9-e4f0-45a5-af4a-d70b5f3789e0)



# 6. Further Research
- Use pre-trained word embeddings (like GloVe or Word2Vec).
- Try different RNN variants like GRU or Bidirectional LSTM.
- Deploy the model as an API using FastAPI or Flask.
- Visualize word importance using attention mechanisms.

# Project 3: Dog and Cat Image Classification Using CNN

# 1. Project Content
This project focuses on detecting whether an image contains a dog or a cat using a machine learning model. It involves loading datasets, preprocessing images, training a model, and evaluating its accuracy.

# 2. Project Code

~~~python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set image dimensions and paths
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Building the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Plotting accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
~~~

# 3. Key Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Deep Learning (CNN)

# 4. Description
The goal of the project is to develop a classification model that can accurately distinguish between dog and cat images. The dataset is split into training and validation sets, with images being resized and normalized. A CNN model is constructed with multiple convolutional, pooling, and dense layers. The model is trained using binary crossentropy as the loss function and accuracy as the metric.

# 5. Output
![image](https://github.com/user-attachments/assets/b05b555b-5c89-4f41-b444-da956464177f)

# 6. Further Research
- Improve accuracy using more advanced architectures like ResNet, Inception, or EfficientNet
- Hyperparameter tuning for optimal performance
- Implement transfer learning with pre-trained models
- Expand to multi-class classification including more animal types
- Deploy the model as a web app using Flask or Streamlit
