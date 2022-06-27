import os
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json




with st.sidebar:
    st.sidebar.subheader("Slider For Labels")
    label_size = st.sidebar.select_slider('Label Size',options=[1,2,3,4,5])

    st.sidebar.subheader('Chart')
    add_selectbox = st.sidebar.selectbox(
            "Do you like to see chart?",
            ("None", "Barplot", "Countplot")
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
    race_classification = st.sidebar.checkbox(
        'Race Classification'
    )
    face_recognition = st.sidebar.checkbox(
        "Face Recognition"
    )
    spoof_detection = st.sidebar.checkbox(
        'Spoof Detection'
    )



labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
w,h = 48,48
path_model = './Modelos/model_dropout.hdf5'
# detect_frontal_face = '../../haar cascade files/haarcascade_frontalface_default.xml'
# face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_detection = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_frontalface_default.xml')
model = load_model('E:/FINAL YEAR PROJECT/emotion detection/models/model_dropout.hdf5')


gender_model_path = 'E:/FINAL YEAR PROJECT/gender classification/models/gender_detection.model'
gender_model = load_model(gender_model_path)
gender_classes = ['man','woman']


# age_model_path = 'E:/FINAL YEAR PROJECT/age prediction/models/gender_detection.model'
age_model_path = 'E:/FINAL YEAR PROJECT/age prediction/models/age_model_pretrained.h5'
age_model = load_model(age_model_path)
age_labels = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']


# ethinicity_model_path = 'E:/FINAL YEAR PROJECT/enthinicity detection/models/ethinicity_model.h5'
# ethinicity_model = load_model(ethinicity_model_path)


# Load Anti-Spoofing Model graph
json_file = open('E:/FINAL YEAR PROJECT/spoofing detection/models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
spoof_model = model_from_json(loaded_model_json)
# load antispoofing model weights 
spoof_model.load_weights('E:/FINAL YEAR PROJECT/spoofing detection/models/antispoofing_model.h5')
print("Model loaded from disk")



st.title('Image Upload.')

image = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])


def upload_image(uploadedfile):
    with open(os.path.join("media",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
         


col1, col2 = st.columns(2)

with col1:
    if image:
        upload_image(image)
        st.image(image, caption='Original Image')

        btn = st.button('Predict')

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
                    
                # st.image(image,caption='Predicted Image')
                           
            
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

                        
                        prediction = model.predict(roi_gray)
                        prediction_label = labels[np.argmax(prediction)]
                        cv2.putText(image,prediction_label,(x+w, y),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)

                    if gender_prediction_checkbox:
                        # preprocessing for gender detection model
                        face_crop = cv2.resize(roi_color, (96,96))
                        face_crop = face_crop.astype("float") / 255.0
                        face_crop = img_to_array(face_crop)
                        face_crop = np.expand_dims(face_crop, axis=0)
                        # predict
                        gender_prediction = gender_model.predict(face_crop)
                        gender_classification = gender_classes[np.argmax(gender_prediction)]
                        cv2.putText(image,gender_classification,(x+w, y+100),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)
                    
                    if age_prediction_checkbox:
                        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),10)
                        roi_color = cv2.resize(roi_color, (200,200), interpolation=cv2.INTER_AREA)
                        roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_RGB2GRAY)
                        # roi_gray = roi_gray.astype('float64') / 255.
                        roi_gray = roi_gray.reshape(1,200,200,1)
                        pred = age_model.predict(roi_gray)
                        predicted_age = age_labels[np.argmax(pred)]
                        cv2.putText(image,predicted_age,(x+w, y+150),cv2.FONT_HERSHEY_COMPLEX,label_size,(255,0,0),2)
                    
                    if race_classification:
                        pass
                        # img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),10)
                        # roi_color = cv2.resize(roi_color, (200,200), interpolation=cv2.INTER_AREA)
                        # roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_RGB2GRAY)
                        # # roi_gray = roi_gray.astype('float64') / 255.
                        # roi_gray = roi_gray.reshape(1,200,200,1)
                        # pred = ethinicity_model.predict(roi_gray)
                        # print("------------------------------")
                        # print(pred)
                        # # predicted_age = age_labels[np.argmax(pred)]
                        # # cv2.putText(image,predicted_age,(x+w, y+150),cv2.FONT_HERSHEY_COMPLEX,4,(255,0,0),4)
                    
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

                        cv2.putText(image,label,(x+w, y),cv2.FONT_HERSHEY_COMPLEX,label_size,(0,255,0),2)

                
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
            


            # if len(prediction) != 0:
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt

            #     fig = plt.figure(figsize=(9,7))

            #     x = labels
            #     y = prediction * 100
            #     sns.barplot(x=x, y = y[0])
            #     st.pyplot(fig)



