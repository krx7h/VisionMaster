import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Cooldowns (in seconds)
cooldowns = {
    "jump": 0.7,
    "slide": 0.7,
    "left": 0.7,
    "right": 0.7
}
last_action_time = {key: 0 for key in cooldowns}
last_direction = "center"
neutral_zone = 0.05  # range around center to not trigger anything

# Previous nose for movement detection
prev_nose = None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        nose = lm[mp_pose.PoseLandmark.NOSE.value]
        current_time = time.time()

        if prev_nose:
            dx = nose.x - prev_nose.x
            dy = nose.y - prev_nose.y

            # Reset to center if face is steady
            if abs(dx) < neutral_zone and abs(dy) < neutral_zone:
                last_direction = "center"

            # LEFT
            elif dx < -neutral_zone and last_direction == "center" and current_time - last_action_time["left"] > cooldowns["left"]:
                print("Move Left")
                pyautogui.press('right')
                last_action_time["left"] = current_time
                last_direction = "left"

            # RIGHT
            elif dx > neutral_zone and last_direction == "center" and current_time - last_action_time["right"] > cooldowns["right"]:
                print("Move Right")
                pyautogui.press('left')
                last_action_time["right"] = current_time
                last_direction = "right"

            # UP (Jump)
            elif dy < -neutral_zone and last_direction == "center" and current_time - last_action_time["jump"] > cooldowns["jump"]:
                print("Jump")
                pyautogui.press('up')
                last_action_time["jump"] = current_time
                last_direction = "up"

            # DOWN (Slide)
            elif dy > neutral_zone and last_direction == "center" and current_time - last_action_time["slide"] > cooldowns["slide"]:
                print("Slide")
                pyautogui.press('down')
                last_action_time["slide"] = current_time
                last_direction = "down"

        # Update nose position
        prev_nose = nose

    cv2.imshow("Face Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
