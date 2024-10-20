from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from pymongo import MongoClient

app = Flask(__name__)

MAX_WORDS = 5000
MAX_LEN = 300
DATASET_PATH = 'news.csv'
MODEL_PATHS = {
    'passive_aggressive_model': 'passive_aggressive_model.pkl',
    'vectorizer': 'vectorizer.pkl',
    'lstm_model': 'lstm_model.keras',
    'tokenizer': 'tokenizer.pkl'
}

client = MongoClient('mongodb://localhost:27017/')
db = client['newsdb']
collection = db['news']

def fetch_data_from_mongo():
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

def add_data_to_csv():
    existing_data = pd.read_csv(DATASET_PATH)
    mongo_data = fetch_data_from_mongo()
    mongo_data = mongo_data[['title']]
    mongo_data.rename(columns={'title': 'text'}, inplace=True)
    mongo_data['label'] = 'REAL'
    combined_data = pd.concat([existing_data, mongo_data], ignore_index=True)
    combined_data.to_csv(DATASET_PATH, index=False)
    print(f"Combined data saved to {DATASET_PATH}")

def create_and_train_models():
    dataframe = pd.read_csv(DATASET_PATH)
    dataframe.dropna(subset=['text', 'label'], inplace=True)
    x = dataframe['text']
    y = dataframe['label'].replace({'FAKE': 0, 'REAL': 1}).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    vectorizer = TfidfVectorizer(max_features=MAX_WORDS)
    x_train_tfidf = vectorizer.fit_transform(x_train).toarray()
    x_test_tfidf = vectorizer.transform(x_test).toarray()
    pac_model = PassiveAggressiveClassifier()
    pac_model.fit(x_train_tfidf, y_train)
    accuracy = pac_model.score(x_test_tfidf, y_test)
    print(f"Passive-Aggressive Classifier accuracy: {accuracy * 100:.2f}%")
    with open(MODEL_PATHS['passive_aggressive_model'], 'wb') as pac_file:
        pickle.dump(pac_model, pac_file)
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(x_train)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_LEN)
    x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_LEN)
    lstm_model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=100),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(x_train_pad, np.array(y_train), epochs=5, batch_size=64, validation_data=(x_test_pad, np.array(y_test)))
    loss, accuracy = lstm_model.evaluate(x_test_pad, np.array(y_test))
    print(f"LSTM Model accuracy: {accuracy * 100:.2f}%")
    lstm_model.save(MODEL_PATHS['lstm_model'])
    with open(MODEL_PATHS['tokenizer'], 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)
    print("Models and vectorizers saved successfully.")

def load_models_and_tokenizers():
    with open(MODEL_PATHS['passive_aggressive_model'], 'rb') as pac_file:
        pac_model = pickle.load(pac_file)
    with open(MODEL_PATHS['vectorizer'], 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    lstm_model = tf.keras.models.load_model(MODEL_PATHS['lstm_model'])
    with open(MODEL_PATHS['tokenizer'], 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return pac_model, vectorizer, lstm_model, tokenizer

if not all(os.path.exists(path) for path in MODEL_PATHS.values()):
    choice = input("Models or tokenizers not found. Do you want to create and train the models? (y/n)").lower()
    if choice == 'y':
        add_data_to_csv()
        create_and_train_models()
    else:
        print("Cannot proceed without trained models and tokenizers.")
        exit()

pac_model, vectorizer, lstm_model, tokenizer = load_models_and_tokenizers()

def fake_news_det(news, model_type):
    if model_type == 'pac':
        input_data = [news]
        vectorized_input_data = vectorizer.transform(input_data)
        prediction = pac_model.predict(vectorized_input_data)
        prediction_label = "REAL" if prediction[0] == 1 else "FAKE"
    elif model_type == 'lstm':
        input_seq = tokenizer.texts_to_sequences([news])
        input_pad = pad_sequences(input_seq, maxlen=MAX_LEN)
        prediction = lstm_model.predict(input_pad)
        prediction_label = "REAL" if prediction >= 0.5 else "FAKE"
    else:
        raise ValueError("Invalid model type")
    return prediction_label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    model_type = request.form['model_type']
    pred = fake_news_det(message, model_type)
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
