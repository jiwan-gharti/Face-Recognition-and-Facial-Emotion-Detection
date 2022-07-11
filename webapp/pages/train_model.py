import os
import cv2
import pandas as pd
import streamlit as st 
from sqlite.database import (
    delete,
    add_data,
    edit_data,
    view_all_data, 
    get_single_data,
)
from validation.validations import (
    age_validation, 
    name_validation, 
    gender_validation
)




face_detection = cv2.CascadeClassifier('E:/FINAL YEAR PROJECT/haar cascade files/haarcascade_frontalface_default.xml')

file_path = f"E:/FINAL YEAR PROJECT/data"


st.subheader("Train Face Recognition Model")

st.sidebar.subheader("Train Model")
option = st.sidebar.selectbox(
    'Select CRUD option.',
    ('Insert', 'Read', 'Update','Delete')
)
if option == 'Insert':
    st.text("*** ALL FIELDS ARE MANDETORY. ***")
    name = st.text_input('Full Name*',on_change=name_validation,key='name')
    age = st.text_input('Age*',on_change=age_validation, key='age')
    gender = st.radio("Gender*",('Male', 'Female'),on_change=gender_validation)

    if name and age and gender:
        save_btn = st.button('Save',on_click=add_data,args=(name,age,gender))
        if save_btn:
            st.success("Successfully Added Data: {}".format(name))

if option == 'Read':
    st.subheader("View Items")
    result = view_all_data()
    df =pd.DataFrame(result,columns=['ID','NAME','AGE','GENDER'])
    st.dataframe(df,width=2000)

if option == 'Update':
    selected_student = st.selectbox(
    'Search Box',
    tuple([f'{data[0]}. {data[1]}' for data in view_all_data()])
    )

    if selected_student:
        st.subheader("Update/Rewite your data.")
        id = selected_student.split('.')[0]
        data = get_single_data(id)

        name = st.text_input('Full Name*',on_change=name_validation,key='name',value=data[1])
        age = st.text_input('Age*',on_change=age_validation,key='age',value=data[2])
        gender_selection = ['Male' if data[3] == 'Male' else 'Female']
        gender = st.selectbox("Gender*",tuple(gender_selection+["Female" if "Male" in gender_selection else 'Male']),on_change=gender_validation)

        st.button("Update",on_click=edit_data(name,age,gender))
        i= 0
        run = st.checkbox('Add Image for Face recognition.', key='12')
        FRAME_WINDOW_CAPTURE = st.image([])
        cam = cv2.VideoCapture(0)

        captuer_image = st.button('Capture Image')
        student_name = selected_student.split('.')[1]
        image_directory = f'{file_path}/{student_name}'
        if not os.path.exists(image_directory):
            os.mkdir(image_directory)

        while run:
            res, frame = cam.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame,1)


            if i < 100:
                if captuer_image:
                    faces = face_detection.detectMultiScale(frame,1.1,5)
                    if len(faces) != 0:
                        for (x,y,w,h) in faces:
                            roi_color = frame[y:y+h, x:x+w]
                            roi_color = cv2.resize(roi_color,(450,450), interpolation=cv2.INTER_AREA)
                            roi_gray = cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)
                            cv2.imwrite(f'{file_path}/{student_name}/image.{id}.{i}.png',roi_gray)
                            print("save save save save")
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)
                        i += 1
                    else:
                        cv2.putText(frame,'No Face',(0, 0),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            
            else:
                frame = cv2.putText(frame,"thank you!",(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            FRAME_WINDOW_CAPTURE.image(frame)


if option == "Delete":
    selected_student = st.selectbox(
    'Search Box',
    tuple([f'{data[0]}. {data[1]}' for data in view_all_data()])
    )
    if selected_student:
        st.subheader("Delete your data.")
        id = selected_student.split('.')[0]
        data = get_single_data(id)
        st.write(data)
        name = st.text_input('Full Name*',on_change=name_validation,key='name',value=data[1])
        st.button("Delete",on_click=delete(int(data[0])))








