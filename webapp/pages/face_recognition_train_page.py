import streamlit as st 
import os
import numpy as np
import cv2

# from webapp.test import model_func

st.subheader("Train Face Recognition Model")

@st.experimental_singleton 
def model_func():
    model = cv2.face.LBPHFaceRecognizer_create()
    return model

model_face = model_func()

fisher_faces_model =  cv2.face.FisherFaceRecognizer_create()
eigen_faces_model = cv2.face.EigenFaceRecognizer_create()


algorithm_selection = st.selectbox(
    "Select Algorithm for Training",
    ("None", "LBPH Algorithm","Eigen Faces Algorithm","Fisher Faces Algorithm")
)



def train_lbph(algorithm_name):
    print(algorithm_name)
    BASE_DIR = 'E:\\FINAL YEAR PROJECT\\data'
    folder_path = [os.path.join(BASE_DIR,folder) for folder in os.listdir(BASE_DIR)]

    file_path = [os.path.join(folder,file) for folder in folder_path for file in os.listdir(folder)]

    faces = []
    ids = []

    imageLocation = st.empty()
    for image_path in file_path:
        import time
        time.sleep(1)
        # image = Image.open(image_path).convert("L")
        # image = np.array(image,dtype='uint8')
        image = cv2.imread(image_path)
        image = cv2.resize(image,(224,224))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        imageLocation.image(image)
        faces.append(image)
        print(os.path.split(image_path)[1].split('.')[1])
        ids.append(int(os.path.split(image_path)[1].split('.')[1]))
    
    faces = np.array(faces)
    ids = np.array(ids)
    print(ids)

    ############# TRAINING LBPH MODEL ###################33
    # model = model_func()
    if algorithm_name == 'LBPH Algorithm':
        model_face.train(faces,ids)
        model_face.save("E:/FINAL YEAR PROJECT/face recognition/outputs/classifier.yaml")
        st.success("Training is Completed.")
    elif algorithm_name == 'Eigen Faces Algorithm':
        eigen_faces_model.train(faces,ids)
        eigen_faces_model.save("E:/FINAL YEAR PROJECT/face recognition/outputs/eigen_face_classifier.yaml")
        st.success("Training is Completed.")
    elif algorithm_name == 'Fisher Faces Algorithm':
        fisher_faces_model.train(faces,ids)
        fisher_faces_model.save("E:/FINAL YEAR PROJECT/face recognition/outputs/fisher_face_classifier.yaml")
        st.success("Training is Completed.")

if algorithm_selection != 'None':
    value = st.button('Train Model',on_click=train_lbph,args=[algorithm_selection])