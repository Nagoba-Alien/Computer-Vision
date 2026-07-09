import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(1)  # Change if needed on your system

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0
    h, w, _ = frame.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            pts = []
            for lm in hand_lms.landmark:
                pts.append((int(lm.x * w), int(lm.y * h)))

            label = handedness.classification[0].label

            # Thumb
            if label == "Right":
                if pts[4][0] < pts[3][0]:
                    finger_count += 1
            else:  # Left hand
                if pts[4][0] > pts[3][0]:
                    finger_count += 1

            # Index, middle, ring, pinky
            for tip in (8, 12, 16, 20):
                if pts[tip][1] < pts[tip - 2][1]:
                    finger_count += 1

            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        f"Fingers: {finger_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
