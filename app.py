from flask import Flask, render_template, request
import numpy as np
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model, graph
    # Load the pre-trained Keras model
    model = load_model('sentiment_analysis.h5')
    graph = tf.get_default_graph()

# Initialize the model when the application starts
init()

# Code for Sentiment Analysis
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods=['POST', 'GET'])
def sent_anly_prediction():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = ''
        probability = 0.0
        img_filename = ''

        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())

        words = text.split()  # Split string into a list
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=500)  # Should be the same as you used for training data
        vector = np.array([x_test.flatten()])
        
        global graph
        with graph.as_default():
            probability = model.predict(array([vector][0]))[0][0]
            class1 = model.predict_classes(array([vector][0]))[0][0]

        if class1 == 0:
            sentiment = 'Negative'
            img_filename = 'Sad_Emoji.png'
        else:
            sentiment = 'Positive'
            img_filename = 'Smiling_Emoji.png'

    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)

# Run the app if this script is executed
if __name__ == "__main__":
    app.run()
