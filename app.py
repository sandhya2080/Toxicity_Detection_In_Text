# app.py
import pandas as pd
import numpy as np
import re
import string
import pickle

from flask import Flask, render_template, request

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('tcc_model.pkl', 'rb'))

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()/@;:{}`+=~|.!?,]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("(\W)", " ", text)
        text = re.sub(r"\b(\w+)\b", r"\1", text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    custom_statement = request.form['comment']
    preprocessed_statement = clean_text(custom_statement)

    # Vectorize the preprocessed statement
    vectorized_statement = vect.transform([preprocessed_statement])

    # Make predictions on the vectorized statement
    predicted_labels = model.predict(vectorized_statement)

    # Calculate the sum of predicted toxic labels
    is_toxic = int(predicted_labels.sum())  # Convert to regular Python integer

    # Convert predicted_labels into a list of strings
    predicted_labels_dict = {}
    for i, category in enumerate(cols_target):
        if predicted_labels[0][i] == 1:
            predicted_labels_dict[category] = '1'
        else:
            predicted_labels_dict[category] = '0'

    # Get the prediction probabilities for the bar chart
    y_test_pred = model.predict_proba(vectorized_statement)
    a_values = y_test_pred[0].tolist()  # Convert NumPy array to Python list

    return render_template('result.html', statement=custom_statement, predicted_labels=predicted_labels_dict, 
                           cols_target=cols_target, a_values=a_values, is_toxic=is_toxic)




    

if __name__ == '__main__':
    # Apply TF-IDF vectorization to the training data (same as before)
    df = pd.read_csv("/Users/uditnath/Documents/Artificial Intelligence/Toxic comment classifier/train.csv")
    df['clean_text'] = df['comment_text'].apply(lambda text: clean_text(text))
    cols_target = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

    X_train, y_train = df['clean_text'], df[cols_target]

    vect = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b\w{1,}\b',
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True
    )

    X_train = X_train.astype(str)
    X_train = vect.fit_transform(X_train)

    # Train the multilabel classifier using the One-vs-Rest strategy with Multinomial Naive Bayes
    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train, y_train)

    

    app.run(port=8000, debug=True)
