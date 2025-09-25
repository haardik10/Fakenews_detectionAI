from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import re
import string
from nltk.corpus import stopwords


model = load_model("fakenews2.keras")

with open("tokenizer2.pkl", "rb") as f:  
    tokenizer = pickle.load(f)

maxlen = 300  
best_thresh = 0.0020

stop_words = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop_words.update(punctuation)

def clean_text(text):
   
    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d", "", text)

    text = " ".join(word for word in text.split() if word not in stop_words)

    return text

def predict_news(text):
    text = clean_text(text)  
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=maxlen, padding="post"
    )
    prob = model.predict(padded)[0][0]
    print("Raw probability:", prob)  

    label = "Real" if prob > best_thresh else "Fake"
    return f"{label} (prob={prob:.4f}, threshold={best_thresh})"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_text = request.form["news_text"]
        result = predict_news(user_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
