import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.set_page_config(page_title="Virtual Mouse", page_icon="🖱️", layout="centered")

st.title("🖱️ Virtual Mouse using Hand Tracking (Demo Version)")
st.write("""
This Streamlit app demonstrates how hand landmarks can be used to control a cursor-like movement.
While full mouse control (pyautogui) isn't supported on Streamlit Cloud, 
you can still visualize your hand-tracking functionality in real-time.
""")

# Initialize MediaPipe Hand module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create a sidebar for settings
st.sidebar.title("⚙️ Settings")
max_num_hands = st.sidebar.slider("Max number of hands", 1, 2, 1)
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5)
run = st.sidebar.checkbox("Run Webcam")

# Start webcam only if the user clicks the checkbox
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

if run:
    st.sidebar.success("Webcam is running. Show your hand to control the virtual cursor!")
    with mp_hands.Hands(max_num_hands=max_num_hands,
                        min_detection_confidence=detection_confidence,
                        min_tracking_confidence=tracking_confidence) as hands:

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    # Example: Draw a virtual pointer on the index fingertip
                    index_finger = hand_landmarks.landmark[8]
                    cx, cy = int(index_finger.x * w), int(index_finger.y * h)
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1)

                    # Optional: Display coordinates
                    cv2.putText(frame, f"Pointer: ({cx},{cy})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.warning("👆 Click 'Run Webcam' in the sidebar to start tracking!")

camera.release()
st.sidebar.info("App created by Prachi Priyadarshini ✨")
