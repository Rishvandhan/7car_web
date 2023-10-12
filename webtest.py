import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
st.set_option('deprecation.showfileUploaderEncoding',False)
st.cache(allow_output_mutation = True)
st.title("Car prediction AI model")
uploaded_file = st.file_uploader("Choose a image file", type=['jpg' ,'png'])
#st.image(uploaded_file)
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    #st.image(opencv_image,channels='RBG')
    st.image(uploaded_file)
    img=cv2.resize(opencv_image,(240,240))
    # Now do something with the image! For example, let's display it:
    #st.image(img,channels='RGB')

def load_model():
    path=r'D:\python\workspace_machineLearning\7_car_web\model5_7cars.h5'
    model=tf.keras.models.load_model(path)
    return model
model=load_model()

def predict(img,model):
    pred = model.predict(img.reshape(1,240,240,3))
    return pred
if st.button('Predict'):
    pred=predict(img,model)

    pred=pred.flatten()
#print(pred)
    pred=pred>0.5

    if(pred[0]==True):
        st.write("Model's prediction is Audi")
    elif(pred[1]==True):
        st.write("Model's prediction is Hyundai Creta")
    elif(pred[2]==True):
        st.write("Model's prediction is Mahindra Scorpio") 
    elif(pred[3]==True):
        st.write("Model's prediction is Rolls Royce")
    elif(pred[4]==True):
        st.write("Model's prediction is Swift")
    elif(pred[5]==True):
        st.write("Model's prediction is Tata Safari")
    else:
        st.write("Model's prediction is Toyota Innova")
