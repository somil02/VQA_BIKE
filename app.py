import os
import pickle
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
import pandas as pd

app = Flask(__name__)

data = pd.read_excel("qa3.xlsx")

model = VGG16(weights='imagenet')
feature_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def parse_function(image):
    image = image/255
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    return  image

model = tf.keras.models.load_model('model_bike.h5')

with open('ohe.pkl', 'rb') as file:
    ohe = pickle.load(file)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['question'])

def predict_answer(image, question):
    tokenized_question = tokenizer.texts_to_sequences([question])
    tokenized_question = pad_sequences(tokenized_question, maxlen=24)

    image_features = feature_model(image)
    tokenized_question = tokenized_question.reshape(1, -1)

    prediction = model.predict([image_features, tokenized_question])
    predicted_answer = ohe.inverse_transform(prediction)
    predicted_answer = predicted_answer[0][0]
    return predicted_answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file and question from the form
        image_file = request.files['image']
        question = request.form['question']

        # Save the uploaded image temporarily
        image_path = 'uploaded_image.jpg'
        image_file.save(image_path)

        # Load and preprocess the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = parse_function(image)

        # Predict the answer
        predicted_answer = predict_answer(image, question)

        # Delete the temporary image file
        os.remove(image_path)

        # return render_template('index.html', predicted_answer=predicted_answer)
        if predicted_answer.lower() == 'yes':
            return render_template('yes.html')
        elif predicted_answer.lower() == 'no':
            return render_template('no.html')
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
