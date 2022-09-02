import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.express as px
from tensorflow.keras.models import model_from_json


from sqlite.database import get_single_data
from cache.models import (
    cache_load_model,
    face_detect_model,
    LBPHRecognizer_model,
    left_right_eye_detect_model,
    cache_load_model,
    
)
from constants.constants import (
    emotion_labels,
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
            ("None", "Graph")
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
    facial_emotion_detection = st.sidebar.checkbox(
        'Facial Emotion Prediction'
    )
    face_recognition = st.sidebar.checkbox(
        "Face Recognition"
    )

# ==================================================
        # Model Initilization Section
# ==================================================


w,h = 48,48

# Load Anti-Spoofing Model graph
json_file = open('../spoofing detection/models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
spoof_model = model_from_json(loaded_model_json)
# load antispoofing model weights 
spoof_model.load_weights('../spoofing detection/models/antispoofing_model.h5')



model_face = LBPHRecognizer_model()
face_detection = face_detect_model()
leye,reye = left_right_eye_detect_model()
emotion_model = cache_load_model(emotion_model_path)
print("Model loaded from disk")


placeholder = st.empty()                                            # creating a single-empty-element container
placeholder1 = st.empty()                                            # creating a single-empty-element container
placeholder2 = st.empty()                                            # creating a single-empty-element container
placeholder3 = st.empty()                                            # creating a single-empty-element container
placeholder4 = st.empty()  
wide_df = px.data.medals_wide()   



# ===============================================
                # For Image Upload
#  ==============================================
st.title('Image Upload.')
image = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])

def upload_image(uploadedfile):
    with open(os.path.join("media",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())




if face_recognition:
    algorithm_selection1 = st.selectbox(
        "Select Algorithm for Recognition",
        ("LBPH Algorithm","Eigen Faces Algorithm","Fisher Faces Algorithm")
    )


@st.cache(allow_output_mutation=True)
def model_face_func(model_path, type):
    if type == 'eigen':
        model_face = cv2.face.EigenFaceRecognizer_create()
        model_face.read(model_path)
        return model_face
    elif type == 'fisherman':
        model_face = cv2.face.FisherFaceRecognizer_create()
        model_face.read(model_path)
        return model_face
    elif type == 'lbph':
        model_face = LBPHRecognizer_model()
        model_face.read(model_path)
        return model_face
    else:
        model_face = cv2.face.EigenFaceRecognizer_create()
        model_face.read(model_path)
        return model_face

         

# ==================================================
        # Image preview (Column 1st) 
# ==================================================
col1, col2 = st.columns(2)

with col1:
    if image:
        upload_image(image)
        st.image(image, caption='Original Image')

        btn = st.button('Predict')



# Predicted Image (Column 2nd) 
# ==================================================

with col2:
    if image and btn:
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
            if area > 3000:
                roi_color = image[y:y+h, x:x+w]

                if facial_emotion_detection:
                    img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)
                    roi_color = cv2.resize(roi_color,(48,48), interpolation=cv2.INTER_AREA)
                    roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
                    roi_gray = roi_gray.reshape(1,48,48,1)

                    roi_gray = roi_gray.astype("float") / 255.0

                    
                    emotion_prediction = prediction = emotion_model.predict(roi_gray)
                    prediction_label = emotion_labels[np.argmax(prediction)]
                    cv2.putText(image,f'{prediction_label}',(x+w, y),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)

                if face_recognition:
                    model_face = LBPHRecognizer_model()
                    model_face.read('../face recognition/outputs/classifier.yaml')
                    roi_color = cv2.resize(roi_color,(224,224))
                    roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
                    try:
                        if algorithm_selection1 == 'Fisher Faces Algorithm':
                            model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/fisher_face_classifier.yaml','fisherman')
                        elif algorithm_selection1 == 'Eigen Faces Algorithm':
                            model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/eigen_face_classifier.yaml','eigen')
                        else:
                            model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/classifier.yaml','lbph')
                    except:
                        model_face = model_face_func('E:/FINAL YEAR PROJECT/face recognition/outputs/classifier.yaml','lbph')

                    prediction, conf = model_face.predict(roi_gray)
                    print(prediction,conf)
                    data = get_single_data(prediction)
                    print(data)
                    cv2.putText(image,f'pred: {str(data[1])} ({int(conf)}%',(x+w, y+h), cv2.FONT_HERSHEY_COMPLEX, label_size,(255,0,0),2)
                    cv2.putText(image,str(data[2]),(x-10, y), cv2.FONT_HERSHEY_COMPLEX, label_size,(255,0,0),2)
                    cv2.putText(image,str(data[3]),(x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, label_size,(255,0,0),2)

            
            if add_selectbox == 'Barplot' and facial_emotion_detection:
                x = emotion_labels
                y = emotion_prediction[0] * 100
                y = [int(i) for i in y]
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



