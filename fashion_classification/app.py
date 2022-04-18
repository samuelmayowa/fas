import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow import keras
import cnn_model
import Seq_model
import pandas as pd
import pickle
import time
#from sequential_model import history_graph

import numpy as np
from PIL import Image, ImageOps

fas_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fas_data.load_data()


cnn_model = tf.keras.models.load_model("rheamamodel.h5")
class_names = ['Tshirt/TOP', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandel', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# -------------- Sidebarr--------------------->
add_selectbox = st.sidebar.selectbox(
    'select the model for classification',
    ('CNN', 'Please choose CNN for now, we will add another model with time')
)


st.title("Fashion MNIST dataset Classification")


def explore_data(train_images, train_label, test_images):
    st.write('Train Images shape:', train_images.shape)
    st.write('Test images shape:', test_images.shape)
    # st.write(train_labels[0:20])
    st.write('Training Classes', len(np.unique(train_labels)))
    st.write('Testing Classes', len(np.unique(test_labels)))


def about_data(cnn_model):

    if st.button("Explore Data"):
        explore_data(train_images, train_labels, test_images)

    if st.button('CNN ModelSumarry'):
        CNN_model_summary()



if add_selectbox == 'Demo Images':
    st.write("Please Upload the Following type of images of cloths for classification")
    image = Image.open("Demo Images/shirt.jpeg")
    image = image.resize((180, 180))
    st.image(image)
    image1 = Image.open("Demo Images/bag.jpg")
    image1 = image1.resize((180, 180))
    st.image(image1)
    image2 = Image.open("Demo Images/sneaker.jpg")
    image2 = image2.resize((180, 180))
    st.image(image2)
    image3 = Image.open("Demo Images/t-shirt.jfif")
    image3 = image3.resize((180, 180))
    st.image(image3)
    image4 = Image.open("Demo Images/blazer.jpg")
    image4 = image4.resize((180, 180))
    st.image(image4)
    image5 = Image.open("Demo Images/pant.jpg")
    image5 = image5.resize((180, 180))
    st.image(image5)


if add_selectbox == 'About data':
    about_data(cnn_model, Seq_model)

if add_selectbox == 'Pretrained Neural network':
    st.info("working on it, updated soon!")
if add_selectbox == 'Working Demo':
    video_file = open('fashion-working-demo.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

if(add_selectbox == 'Contact us'):
    image = Image.open('sandeep yadav.jpg')
    image = image.resize((400, 400))
    st.image(image)
    st.write('Patience')
    st.write('contact:patience@gmail.com')


# file uploder

if(add_selectbox == 'CNN' or add_selectbox == 'Sequential'):
    file_uploader = st.file_uploader('Upload cloth Image for Classification:')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file_uploader is not None:
        image = Image.open(file_uploader)
        text_io = io.TextIOWrapper(file_uploader)
        image = image.resize((180, 180))
        st.image(image, 'Uploaded image:')

        def classify_image(image, model):
            st.write("classifying......")
            img = ImageOps.grayscale(image)

            img = img.resize((28, 28))
            if(add_selectbox == 'Sequential'):
                img = np.expand_dims(img, 0)
            else:
                img = np.expand_dims(img, 0)
                img = np.expand_dims(img, 3)
            img = (img/255.0)

            img = 1-img

            pred = model.predict(img)

            st.write("The Predicted image is:", class_names[np.argmax(pred)])
            st.write('Prediction probability :{:.2f}%'.format(
                np.max(pred)*100))
        st.write('Click for classify the image')
        if st.button('Classify Image'):
            if(add_selectbox == 'CNN'):
                st.write("You are choosen Image classification with CNN Model")
                classify_image(image, cnn_model)
                st.success('This Image successufully classified!')
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    st.success('Done!')
                    st.balloons()
    else:
        st.write("Please select image:")
