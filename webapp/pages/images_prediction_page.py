import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array


from sqlite.database import get_single_data
from cache.models import (
    cache_load_model,
    face_detect_model,
    LBPHRecognizer_model,
    left_right_eye_detect_model,
    cache_load_model,
    
)
from constants.constants import (
    age_labels,
    emotion_labels,
    gender_classes,
    age_model_path,
    gender_model_path,
    emotion_model_path,
)


# ==================================================
        #side Bar Options
# ==================================================

with st.sidebar:
    st.sidebar.subheader("Slider For Labels")
    label_size = st.sidebar.select_slider('Label Size',options=[1,2,3,4,5])

    st.sidebar.subheader('Chart')
    add_selectbox = st.sidebar.selectbox(
            "Do you like to see chart?",
            ("None", "Barplot")
        )
    
    st.subheader('Categories : ')

    gray_image = st.sidebar.checkbox(
        'Gray Level Photo'
    )
    detect_face = st.sidebar.checkbox(
        'Detect Face'
    )

    region_of_interest = st.sidebar.checkbox(
        "Region Of Interest"
    )
    age_prediction_checkbox = st.sidebar.checkbox(
        "Age Range Prediction",
            # ("Email", "Home phone", "Mobile phone")
        )
    gender_prediction_checkbox = st.sidebar.checkbox(
        "Gender Classification"
    )
    facial_emotion_detection = st.sidebar.checkbox(
        'Facial Emotion Prediction'
    )
    face_recognition = st.sidebar.checkbox(
        "Face Recognition"
    )
    spoof_detection = st.sidebar.checkbox(
        'Spoof Detection'
    )

placeholder = st.empty()                                            # creating a single-empty-element container
placeholder1 = st.empty()                                            # creating a single-empty-element container
placeholder2 = st.empty()                                            # creating a single-empty-element container
placeholder3 = st.empty()                                            # creating a single-empty-element container
placeholder4 = st.empty()   
wide_df = px.data.medals_wide()


# ==================================================
        # Model Initilization Section
# ==================================================


w,h = 48,48
# path_model = './Modelos/model_dropout.hdf5'

# Load Anti-Spoofing Model graph
json_file = open('E:/FINAL YEAR PROJECT/spoofing detection/models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
spoof_model = model_from_json(loaded_model_json)
# load antispoofing model weights 
spoof_model.load_weights('E:/FINAL YEAR PROJECT/spoofing detection/models/antispoofing_model.h5')



model_face = LBPHRecognizer_model()
face_detection = face_detect_model()
leye,reye = left_right_eye_detect_model()
age_model = cache_load_model(age_model_path)
gender_model = cache_load_model(gender_model_path)
emotion_model = cache_load_model(emotion_model_path)
print("Model loaded from disk")



# ===============================================
                # For Image Upload
#  ==============================================
st.title('Image Upload.')
image = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])

def upload_image(uploadedfile):
    with open(os.path.join("media",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
         

# ==================================================
        # Image preview (Column 1st) 
# ==================================================
col1, col2 = st.columns(2)

with col1:
    if image:
        upload_image(image)
        st.image(image, caption='Original Image')

        btn = st.button('Predict')


# ==================================================
        # Predicted Image (Column 2nd) 
# ==================================================

with col2:
    if image:
        if btn:
            image = Image.open(image)
            image = np.asarray(image)
            image_copy = image.copy()
            gray_image_ = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            faces = face_detection.detectMultiScale(gray_image_,1.1,5)

            if gray_image:
                gray_image_converted = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
            if detect_face:
                for (x,y,w,h) in faces:
                    area = w * h
                    if area > 7000:
                        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),4)
                    
                           
            
            for (x,y,w,h) in faces:
                area = w * h

                if area > 1000:
                    # roi_gray = gray_image[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]



                    if facial_emotion_detection:
                        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)
                        roi_color = cv2.resize(roi_color,(48,48), interpolation=cv2.INTER_AREA)
                        roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
                        roi_gray = roi_gray.reshape(1,48,48,1)

                        roi_gray = roi_gray.astype("float") / 255.0

                        
                        facial_emotion_pred = prediction = emotion_model.predict(roi_gray)
                        prediction_label = emotion_labels[np.argmax(prediction)]
                        cv2.putText(image,f'Emotion:{prediction_label}',(x+w, y),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)

                    if gender_prediction_checkbox:
                        # preprocessing for gender detection model
                        face_crop = cv2.resize(roi_color, (96,96))
                        face_crop = face_crop.astype("float") / 255.0
                        face_crop = img_to_array(face_crop)
                        face_crop = np.expand_dims(face_crop, axis=0)
                        # predict
                        gender_prediction = gender_model.predict(face_crop)
                        gender_classification = gender_classes[np.argmax(gender_prediction)]
                        cv2.putText(image,f'Gender: {gender_classification}',(x, y-20),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)
                    
                    if age_prediction_checkbox:
                        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),10)
                        roi_color = cv2.resize(roi_color, (200,200), interpolation=cv2.INTER_AREA)
                        roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_RGB2GRAY)
                        # roi_gray = roi_gray.astype('float64') / 255.
                        roi_gray = roi_gray.reshape(1,200,200,1)
                        age_prediction = pred = age_model.predict(roi_gray)
                        predicted_age = age_labels[np.argmax(pred)]
                        cv2.putText(image,f"Age: {predicted_age}",(x+w, y+150),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)
                    
                    
                    if spoof_detection:
                        print("0-------------------------------0")
                        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
                        resized_face = cv2.resize(image,(160,160))
                        resized_face = resized_face.astype("float") / 255.0
                        resized_face = np.expand_dims(resized_face, axis=0)
                        preds = spoof_model.predict(resized_face)[0]
                        if preds> 0.5:
                            label = 'spoof'
                        else:
                            label = 'real'

                        cv2.putText(image,label,(x, y+h+50),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)
                    
                    if face_recognition:
                        print("+++++++++------------------+++++++++++++++++++")
                        model_face = LBPHRecognizer_model()
                        model_face.read('E:/FINAL YEAR PROJECT/face recognition/outputs/classifier.yaml')
                        roi_color = cv2.resize(roi_color,(224,224))
                        roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
                        prediction, conf = model_face.predict(roi_gray)
                        print(prediction,conf)
                        data = get_single_data(prediction)
                        print(data)
                        cv2.putText(image,f'pred: {str(data[1])} ({int(conf)}%)',(x+w, y+h), cv2.FONT_HERSHEY_COMPLEX, label_size,(255,0,0),2)


                    # Plot 
                    if add_selectbox == 'Barplot' and age_prediction_checkbox:
                        x = age_labels
                        y = age_prediction[0] * 100
                        y = [str(int(i)) for i in y]
                        fig = px.bar(
                        wide_df,
                        x=x,
                        y=y,
                        text=y,
                        color=x,
                        title="Age Prediction",
                        height=400,
                        labels = {
                            'x':'Age class',
                            'y': 'Age Prediction in Perctange'
                            }
                        )
                        placeholder1.plotly_chart(fig)

                    if add_selectbox == 'Barplot' and facial_emotion_detection:
                        x = emotion_labels
                        y = facial_emotion_pred[0] * 100
                        y = [int(i) for i in y]
                        # fig = px.bar(x=x, y=y)
                        fig = px.bar(
                            wide_df,
                            x=x,
                            y=y,
                            text=y,
                            color=x,
                            title="Facial Emotion Prediction",
                            height=400,
                            labels = {
                                'x':'Facial Emotion class',
                                'y': 'Facial Emotion Prediction in Perctange'
                            }
                        )
                        placeholder2.plotly_chart(fig)

                    if add_selectbox == 'Barplot' and spoof_detection:
                        x = ['spoof','real']
                        y = [preds[0], 1 - preds[0]]
                        y = [int(i *  100) for i in y]

                        fig = px.bar(
                            wide_df,
                            x=x,
                            y=y,
                            text=y,
                            color=x,
                            title="Spoof Prediction",
                            height=400,
                            labels = {
                                'x':'Spoof Detection class',
                                'y': 'Spoof Prediction in Perctange'
                            }
                        )
                        placeholder3.plotly_chart(fig)

                    if add_selectbox == 'Barplot' and gender_prediction_checkbox:
                        x = gender_classes
                        y = gender_prediction[0] * 100
                        y = [int(i) for i in y]
                        print(x,y)
                        fig = px.bar(
                            wide_df,
                            x=x,
                            y=y,
                            text=y,
                            color=gender_classes,
                            title="Gender Prediction",
                            height=400,
                            labels = {
                                'x':'Gender',
                                'y': 'Gender Prediction in Perctange'
                            }
                        )
                        placeholder4.plotly_chart(fig)
                
            st.image(image,caption='Predicted Image') 

            if gray_image:
                st.image(gray_image_converted,caption='Predicted Image')

            if region_of_interest:
                if len(faces) > 0:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(5,5))

                    for i, (x,y,w,h) in enumerate(faces):
                        plt.subplot(int(len(faces)/3) +1,3, i+1)
                        roi_color = image_copy[y:y+h, x:x+w]
                        roi_color = cv2.resize(roi_color,(500,500))
                        plt.axis("off")
                        plt.imshow(roi_color)

                    st.pyplot(fig)
            

            
            


