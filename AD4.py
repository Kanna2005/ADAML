#Practical 4
#Machine Learning for Text classification
#code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load dataset
data = pd.read_csv(
    'https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-',
    encoding='latin-1'
)

# Drop unnecessary columns
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Rename columns
data.columns = ['label', 'text']

# Basic info
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data['label'].value_counts())

# Plot distribution
data['label'].value_counts(normalize=True).plot.bar(title="Target Distribution")
plt.show()

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing
lemmatizer = WordNetLemmatizer()
corpus = []

for text in data['text']:
    r = re.sub('[^a-zA-Z]', ' ', text)  # remove special characters
    r = r.lower()                      # lowercase
    r = r.split()                      # tokenize
    r = [lemmatizer.lemmatize(word) for word in r if word not in stopwords.words('english')]
    corpus.append(' '.join(r))

# Add cleaned text
data['clean_text'] = corpus

# Features and labels
X = data['clean_text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Vectorization
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Model training
lr = LogisticRegression()
lr.fit(X_train_cv, y_train)

# Prediction
y_pred = lr.predict(X_test_cv)

# Evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))
