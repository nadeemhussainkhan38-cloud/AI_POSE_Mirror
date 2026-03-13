import streamlit as st
import cv2
import mediapipe as mp

st.set_page_config(page_title="AI Mirror Assistant", layout="centered")

st.title("🤖 AI Mirror Assistant")

st.write("Move your body and the AI will copy your pose.")

start = st.button("Start Camera")

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()
face = mp_face.FaceMesh()

FRAME_WINDOW = st.image([])

if start:

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            st.write("Camera not working")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result = pose.process(rgb)
        face_result = face.process(rgb)

        # Body Skeleton
        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Eye / Face Mesh
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_TESSELATION
                )

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
