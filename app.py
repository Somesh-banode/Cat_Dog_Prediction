import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
print("File exists:", os.path.exists("models\cnn.keras"))


model = tf.keras.models.load_model("models\cnn.keras")
class_labels = ['cat', 'dog']

st.title("Image Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    # result = class_labels[np.argmax(prediction)]
    prediction=""
    if  result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    st.markdown("### 🧠 Prediction Result:")
    st.write(f"**Predicted Class:** `{prediction}`")

    st.spinner("Loading...")	#Show a spinner during processing
    #st.progress(87)	            #Progress bar
    st.balloons()	                #Show balloons 🎈
    #st.toast("Message this is toat notification")	            #Toast notification (new)
    


