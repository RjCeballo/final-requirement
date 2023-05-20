import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True, hash_funcs={tf.keras.models.Model: id})
def load_model():
    model = tf.keras.models.load_model('chess.h5')
    return model

st.write("""
# Chess Pieces Image Classifier
""")

file = st.file_uploader("upload any chess pieces", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (32, 32)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    # Preprocess the image if required
    # img_preprocessed = preprocess_image(img_reshape)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = load_model()
    prediction = import_and_predict(image, model)
    
    class_names = [
        'rook', 'queen', 'pawn', 'knight', 'king', 'bishop'
    ]
    string = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(string)
