from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os
import nltk

# make sure all of these are present
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and tokenizer paths
LSTM_MODEL_PATH = os.path.join("models", "lstm_model.keras")
GRU_MODEL_PATH = os.path.join("models", "gru_model.keras")
TEXTCNN_MODEL_PATH = os.path.join("models", "textcnn_model.keras")
TOKENIZER_PATH = os.path.join("models", "preprocessed_data.pkl")
MAX_SEQUENCE_LENGTH = 60

# Load models and tokenizer at startup
lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
gru_model = tf.keras.models.load_model(GRU_MODEL_PATH)
textcnn_model = tf.keras.models.load_model(TEXTCNN_MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    preprocessed_data = pickle.load(f)
tokenizer = preprocessed_data["tokenizer"]

# Preprocessing function (copied from notebook)
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = " ".join(tokens)
    return cleaned_text

def combine_fields(payload):
    # Concatenate fields in the same order as the notebook
    return (
        str(payload["statement"]) + " " +
        str(payload["subject"]) + " " +
        str(payload["speaker"]) + " " +
        str(payload["speakers_job_title"]) + " " +
        str(payload["location"]) + " " +
        str(payload["party"]) + " " +
        str(payload["context"])
    )

class PredictionRequest(BaseModel):
    statement: str
    subject: str
    speaker: str
    speakers_job_title: str
    location: str
    party: str
    context: str

# Helper to preprocess and vectorize input
def prepare_input(payload):
    combined = combine_fields(payload)
    processed = preprocess_text(combined)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    return padded

@app.post("/predict/lstm")
def predict_lstm(request: PredictionRequest):
    try:
        x = prepare_input(request.dict())
        prob = float(lstm_model.predict(x, verbose=0)[0][0])
        return {"probability_lstm": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/gru")
def predict_gru(request: PredictionRequest):
    try:
        x = prepare_input(request.dict())
        prob = float(gru_model.predict(x, verbose=0)[0][0])
        return {"probability_gru": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/textcnn")
def predict_textcnn(request: PredictionRequest):
    try:
        x = prepare_input(request.dict())
        prob = float(textcnn_model.predict(x, verbose=0)[0][0])
        return {"probability_textcnn": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))