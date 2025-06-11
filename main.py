# Importing required libraries
import cv2                         # For capturing video and image processing
import mediapipe as mp             # For hand tracking
import pyautogui                   # For controlling mouse and taking screenshots
import random                      # For generating random screenshot names
import util                        # Custom utility functions (angles, distance)
from pynput.mouse import Button, Controller  # For simulating mouse clicks

# Initializing mouse controller and getting screen size
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Setting up Mediapipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,            # Real-time processing
    model_complexity=1,                 # Balance between speed and accuracy
    min_detection_confidence=0.7,       # Minimum confidence for detection
    min_tracking_confidence=0.7,        # Minimum confidence for tracking
    max_num_hands=1                     # Detect only one hand
)

# Helper function to get index finger tip coordinates
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None

# Function to move the mouse pointer based on finger position
def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * 1.5 * screen_height)  # Adjusted for more accuracy
        pyautogui.moveTo(x, y)

# Gesture recognition functions
def is_left_click(landmark_list, thumb_index_dist):
    # Gesture: Bent index, extended middle finger, thumb away
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmark_list, thumb_index_dist):
    # Gesture: Bent middle, extended index, thumb away
    return (
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 50
    )

def is_double_click(landmark_list, thumb_index_dist):
    # Gesture: Both index and middle bent, thumb away
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50
    )

def is_screenshot(landmark_list, thumb_index_dist):
    # Gesture: Both fingers bent and thumb close
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )

# Function to handle all gesture-based operations
def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        # Gesture: Move mouse
        if thumb_index_dist < 50 and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)

        # Gesture: Left click
        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gesture: Right click
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Gesture: Double click
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Gesture: Screenshot
        elif is_screenshot(landmark_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# Main function to run the webcam loop
def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the video
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe
            processed = hands.process(frameRGB)

            # Extract landmarks
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Run the main function
if __name__ == '__main__':
    main()
