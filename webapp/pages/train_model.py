import streamlit as st 
import cv2
import os


face_detection = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_frontalface_default.xml')

file_path = f"E:/FINAL YEAR PROJECT/face_recognition_train_image_folder"


st.subheader("Train Face Recognition Model")

with st.sidebar:
    st.subheader("Train Model")

    train_model_option = st.sidebar.selectbox(
            "Select Category to Train Model",
            ("None", "Face Recognition", "Emotion Detection")
        )


if train_model_option == 'Face Recognition':
    image_folder = st.sidebar.radio("Select Perfer Category",('Use Folder Image', 'Use Webcam'))

    if image_folder == 'Use Folder Image':
        image_folder_path = st.text_input('File Path', 'E:/folder1/subfolder1',key='9989989')
        train_btn = st.button('Train')

    i= 0
    if image_folder == 'Use Webcam':
        run = st.checkbox('Open or Close WebCam', key='12')
        FRAME_WINDOW_CAPTURE = st.image([])
        cam = cv2.VideoCapture(0)

        name = st.text_input('Enter Your Name *', '')
        # if len(name) > 2:
        captuer_image = st.button('Capture Image')
        image_directory = f'{file_path}/{name}'
        print('----------------------sdsfsddddddddddddddd-------------------')
        print(image_directory)
        if not os.path.exists(image_directory):
            os.mkdir(image_directory)
        # else:
        #     st.write("Please Enter Your Display Name")

        

        while run:
            res, frame = cam.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)

            FRAME_WINDOW_CAPTURE.image(frame)

            if captuer_image:
                if i < 100:
                    faces = face_detection.detectMultiScale(frame,1.1,5)
                    if len(faces) != 0:
                        for (x,y,w,h) in faces:
                            roi_color = frame[y:y+h, x:x+w]
                            roi_color = cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB)
                            cv2.imwrite(f'{file_path}/{name}/image_{i}.png',roi_color)
                            print("save save save save")
                            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
                        i += 1
                    else:
                        cv2.putText(frame,'No Face',(0, 0),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)




    

