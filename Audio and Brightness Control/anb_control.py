import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

vol = 50.0
bright = 50.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for lmks, handedness in zip(result.multi_hand_landmarks,
                                    result.multi_handedness):

            h, w, _ = frame.shape

            thumb = lmks.landmark[4]
            index = lmks.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            dist = np.hypot(x2 - x1, y2 - y1)
            value = np.interp(dist, [20, 180], [0, 100])

            # Exponential smoothing
            if handedness.classification[0].label == "Right":
                vol = 0.8 * vol + 0.2 * value
            else:
                bright = 0.8 * bright + 0.2 * value

            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            draw.draw_landmarks(frame, lmks, mp_hands.HAND_CONNECTIONS)

    # Volume bar (left side of screen)
    cv2.rectangle(frame, (30, 100), (60, 400), (255, 255, 255), 2)
    vfill = int(np.interp(vol, [0, 100], [400, 100]))
    cv2.rectangle(frame, (30, vfill), (60, 400), (0, 255, 0), -1)
    cv2.putText(frame, f"Vol {vol:.0f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Brightness bar (right side of screen)
    cv2.rectangle(frame, (580, 100), (610, 400), (255, 255, 255), 2)
    bfill = int(np.interp(bright, [0, 100], [400, 100]))
    cv2.rectangle(frame, (580, bfill), (610, 400), (0, 255, 255), -1)
    cv2.putText(frame, f"Bri {bright:.0f}%", (500, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Virtual Controls", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
